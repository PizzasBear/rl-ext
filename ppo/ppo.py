import rlext
import numpy as np
import numpy.typing as npt

import gym
import gym.spaces
import pybullet_envs as _

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from dataclasses import dataclass
from datetime import datetime
from itertools import count
from typing import Union

DEVICE          = torch.device('cuda')
BUFFER_SIZE     = 256
NUM_ENVS        = 12
PPO_EPOCHS      = 10
BATCH_SIZE      = 64
PPO_EPSILON     = 0.2
DISCOUNT        = 0.99
GAE_LAMBDA      = 0.95
NORM_ADVANTAGE  = True
POLICY_LR       = 3e-4
VALUE_LR        = 1e-3
ENTROPY_COEF    = 1e-3
LOG_DIR: str = 'lunar-lander/'
SAVE_PATH: str = 'lunar-lander1.pt.tar'

def make_env():
    return gym.wrappers.NormalizeObservation(gym.vector.make("LunarLander-v2", NUM_ENVS, continuous=True, new_step_api=True, autoreset=True), new_step_api=True)

def make_play_env():
    return gym.wrappers.NormalizeObservation(gym.make("LunarLander-v2", continuous=True, new_step_api=True, autoreset=True, render_mode='human'), new_step_api=True)

TOTAL_BUFFER_SIZE = BUFFER_SIZE * NUM_ENVS

@torch.no_grad()
def xavier_linear(in_features: int, out_features: int, device: torch.device = None, activation: str = 'linear') -> nn.Linear:
    linear = nn.Linear(in_features, out_features, device=device)
    nn.init.xavier_uniform_(linear.weight, gain=nn.init.calculate_gain(activation))
    return linear

class Policy(nn.Module):
    net: nn.Sequential
    loc_layer: nn.Linear
    scale_layer: nn.Linear
    input_size: int
    output_size: int

    @torch.no_grad()
    def __init__(self, input_size: int, output_size: int, device: torch.device = None):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.net = nn.Sequential(
            xavier_linear(self.input_size, 64, device=device, activation='tanh'),
            nn.Tanh(),
            xavier_linear(64, 64, device=device, activation='tanh'),
            nn.Tanh(),
        )
        self.loc_layer = xavier_linear(64, self.output_size, device=device)
        self.scale_layer = xavier_linear(64, self.output_size, device=device)

        # self.loc_layer.weight.div_(100)
        # self.loc_layer.bias.div_(100)
        # self.scale_layer.weight.div_(10)
        # self.scale_layer.bias.div_(10)

    def forward(self, x: torch.Tensor) -> Normal:
        x = self.net(x)
        return Normal(self.loc_layer(x), F.softplus(self.scale_layer(x)))

    @torch.no_grad()
    def act(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist: Normal = self(x)
        sample: torch.Tensor = dist.sample()
        tanh_sample: torch.Tensor = sample.tanh()
        return sample, tanh_sample, tanh_normal_log_prob(dist, sample, tanh_sample)

class Value(nn.Module):
    net: nn.Sequential
    input_size: int

    def __init__(self, input_size: int, device: torch.device = DEVICE):
        super().__init__()

        self.input_size = input_size
        self.net = nn.Sequential(
            nn.Linear(self.input_size, 256, device=device),
            nn.Tanh(),
            nn.Linear(256, 256, device=device),
            nn.Tanh(),
            nn.Linear(256, 1, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)

@dataclass
class Buffer:
    states: npt.NDArray[np.float32]
    values: npt.NDArray[np.float32]
    atanh_actions: npt.NDArray[np.float32]
    log_probs: npt.NDArray[np.float32]
    rewards: npt.NDArray[np.float32]
    terminated: npt.NDArray[np.bool_]
    truncated: npt.NDArray[np.bool_]


def tanh_normal_log_prob(dist: Normal, sample: torch.Tensor, tanh_sample: torch.Tensor) -> torch.Tensor:
    return dist.log_prob(sample).sum(1) - (1 + 1e-6 - tanh_sample.square()).log().sum(1)

def map_actions(action: npt.NDArray[np.float32], action_space: gym.spaces.Box) -> npt.NDArray[np.float32]:
    return (action * (action_space.high - action_space.low) + action_space.high + action_space.low) / 2

@torch.no_grad()
def all_finite(x: Union[torch.Tensor, npt.NDArray, float]) -> bool:
    if isinstance(x, torch.Tensor):
        return np.isfinite(x.sum().cpu().numpy())
    else:
        return np.isfinite(np.sum(x))

def ppo_step(
    t: int,
    writer: SummaryWriter,
    policy_net: Policy,
    value_net: Value,
    policy_opt: optim.Adam,
    value_opt: optim.Adam,
    buffer: Buffer,
    device: torch.device = None,
):
    mean_ppo_loss = 0.
    mean_value_loss = 0.
    mean_value_err = 0.
    mean_entropy = 0.
    mean_value = 0.
    mean_return = 0.

    buf_states = torch.tensor(buffer.states, device=device).view(TOTAL_BUFFER_SIZE, -1)
    buf_atanh_actions = torch.tensor(buffer.atanh_actions, device=device).view(TOTAL_BUFFER_SIZE, -1)
    buf_log_probs = torch.tensor(buffer.log_probs, device=device).view(-1)

    for _ in range(PPO_EPOCHS):
        buf_returns, buf_advantages = rlext.par_gae(
            buffer.values,
            buffer.rewards,
            buffer.terminated,
            buffer.truncated,
            lambda_=GAE_LAMBDA,
            discount=DISCOUNT,
            normalize_advantages=NORM_ADVANTAGE,
        )
        mean_return += buf_returns.sum() / BATCH_SIZE

        buf_returns = torch.tensor(buf_returns, device=device).view(-1)  # .clip(-1e+4, 1e+4)
        buf_advantages = torch.tensor(buf_advantages, device=device).view(-1)

        for indices in torch.randperm(TOTAL_BUFFER_SIZE, device=device).view(-1, BATCH_SIZE):
            states = buf_states[indices, :]
            atanh_actions = buf_atanh_actions[indices, :]
            old_action_log_probs = buf_log_probs[indices]
            returns = buf_returns[indices]
            advantages = buf_advantages[indices]

            actions = atanh_actions.tanh()

            policy_dist: Normal = policy_net(states)

            action_log_probs = tanh_normal_log_prob(policy_dist, atanh_actions, actions)
            prob_ratio = (action_log_probs - old_action_log_probs).exp()
            ppo_loss = -torch.minimum(prob_ratio * advantages, prob_ratio.clip(1 / (1 + PPO_EPSILON), 1 + PPO_EPSILON) * advantages).mean()

            new_atanh_actions = policy_dist.rsample()
            entropy = -tanh_normal_log_prob(policy_dist, new_atanh_actions, new_atanh_actions.tanh()).mean()

            policy_loss = ppo_loss - ENTROPY_COEF * entropy

            policy_opt.zero_grad()
            policy_loss.backward()
            policy_opt.step()

            values: torch.Tensor = value_net(states)
            value_err = values.detach() - returns
            value_loss = F.mse_loss(values, returns)

            value_opt.zero_grad()
            value_loss.backward()
            value_opt.step()

            buffer.values.reshape(-1)[indices.cpu().numpy()] = values.detach().cpu().numpy()

            mean_ppo_loss += ppo_loss.detach().cpu().numpy()
            mean_value_err += value_err.detach().mean().cpu().numpy()
            mean_value_loss += value_loss.detach().cpu().numpy()
            mean_entropy += entropy.detach().cpu().numpy()
            mean_value += values.detach().mean().cpu().numpy()

    d = TOTAL_BUFFER_SIZE * PPO_EPOCHS // BATCH_SIZE
    writer.add_scalar("loss/ppo_loss", mean_ppo_loss / d, t)
    writer.add_scalar("loss/value_err", mean_value_err / d, t)
    writer.add_scalar("loss/value_loss", mean_value_loss / d, t)
    writer.add_scalar("entropy", mean_entropy / d, t)
    writer.add_scalar("debug/value", mean_value / d, t)
    writer.add_scalar("debug/return", mean_return / d, t)

def main():
    rewards: npt.NDArray[np.float32]
    terminated: npt.NDArray[np.bool_]
    truncated: npt.NDArray[np.bool_]

    env = make_env()
    assert isinstance(env.action_space, gym.spaces.Box)
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.dtype == np.float32
    assert env.action_space.dtype == np.float32

    assert NUM_ENVS == env.action_space.shape[0] == env.observation_space.shape[0]
    _, action_size = env.action_space.shape
    _, state_size = env.observation_space.shape

    policy_net = Policy(state_size, action_size, device=DEVICE)
    value_net = Value(state_size, device=DEVICE)

    policy_opt = optim.Adam(policy_net.parameters(), lr=POLICY_LR)
    value_opt = optim.Adam(value_net.parameters(), lr=VALUE_LR)

    buffer = Buffer(
        states          = np.zeros((NUM_ENVS, BUFFER_SIZE, state_size), dtype=np.float32),
        values          = np.zeros((NUM_ENVS, BUFFER_SIZE + 1), dtype=np.float32),
        atanh_actions   = np.zeros((NUM_ENVS, BUFFER_SIZE, action_size), dtype=np.float32),
        log_probs       = np.zeros((NUM_ENVS, BUFFER_SIZE), dtype=np.float32),
        rewards         = np.zeros((NUM_ENVS, BUFFER_SIZE), dtype=np.float32),
        terminated      = np.zeros((NUM_ENVS, BUFFER_SIZE), dtype=np.bool_),
        truncated       = np.zeros((NUM_ENVS, BUFFER_SIZE), dtype=np.bool_),
    )

    fitness: npt.NDArray[np.float32] = np.zeros(env.num_envs, dtype=np.float32)
    states: npt.NDArray[np.float32] = env.reset().astype(np.float32)

    log_dir: str = 'data/logs/' + LOG_DIR + datetime.now().strftime('%Y%m%d-%H%M%S')
    writer = SummaryWriter(log_dir)

    for tt in count(1):
        t = tt * TOTAL_BUFFER_SIZE
        with torch.no_grad():
            for i in range(BUFFER_SIZE):
                with torch.no_grad():
                    states_tch = torch.tensor(states, device=DEVICE)
                    atanh_actions_tch, actions_tch, log_probs_tch = policy_net.act(states_tch)
                    values: npt.NDArray[np.float32] = value_net(states_tch).cpu().numpy()

                atanh_actions: npt.NDArray[np.float32] = atanh_actions_tch.cpu().numpy()
                actions: npt.NDArray[np.float32] = actions_tch.cpu().numpy()
                log_probs: npt.NDArray[np.float32] = log_probs_tch.cpu().numpy()

                buffer.states[:, i]         = states
                buffer.values[:, i]         = values
                buffer.atanh_actions[:, i]  = atanh_actions
                buffer.log_probs[:, i]      = log_probs

                states, rewards, terminated, truncated, _ = env.step(map_actions(actions, env.action_space))
                states = states.astype(np.float32)
                fitness += rewards

                buffer.rewards[:, i]        = rewards
                buffer.terminated[:, i]     = terminated
                buffer.truncated[:, i]      = truncated

                done: bool
                for j, done in enumerate(terminated | truncated):
                    if done:
                        writer.add_scalar("fitness", fitness[j], t + NUM_ENVS * i + j)
                        fitness[j] = 0

            buffer.values[:, BUFFER_SIZE] = value_net(torch.tensor(states, device=DEVICE)).cpu().numpy()

        ppo_step(t, writer, policy_net, value_net, policy_opt, value_opt, buffer, device=DEVICE)
        if not t % 8:
            torch.save({
                'policy_net': policy_net.state_dict(),
                'value_net': value_net.state_dict(),
                'policy_opt': policy_opt.state_dict(),
                'value_opt': value_opt.state_dict(),
            }, SAVE_PATH)


if __name__ == '__main__':
    main()
