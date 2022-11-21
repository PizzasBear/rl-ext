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

NUM_ENVS                = 24
BUFFER_SIZE             = 3 * 995_328 // NUM_ENVS
BUFFER_MIN_LEN          = 1024
ENV_STEPS_PER_TIMESTEP  = 4
EPOCHS                  = 24
BATCH_SIZE              = 128
TARGET_SMOOTHING_COEF   = 5e-3
DISCOUNT                = 0.99
POLICY_LR               = 3e-4
QNET_LR                 = 3e-4
ALPHA_LR                = 3e-4

RESTORE: bool = False
# LOG_DIR: str = 'lunar-lander/'
# SAVE_PATH: str = 'lunar-lander.2.pt.tar'
# LOG_DIR: str = 'half-cheetah/'
# SAVE_PATH: str = 'half-cheetah.2.pt.tar'
LOG_DIR: str = 'walker2d/'
SAVE_PATH: str = 'walker2d.2.pt.tar'
# LOG_DIR: str = 'hopper/'
# SAVE_PATH: str = 'hopper.1.pt.tar'

device = torch.device('cuda')
rng = np.random.default_rng()

def make_env():
    return gym.wrappers.NormalizeObservation(
        # gym.vector.make("LunarLander-v2", NUM_ENVS, continuous=True, new_step_api=True, autoreset=True),
        # gym.vector.make("HalfCheetah-v4", NUM_ENVS, new_step_api=True, autoreset=True),
        gym.vector.make("Walker2d-v4", NUM_ENVS, new_step_api=True, autoreset=True),
        # gym.vector.make("Hopper-v4", NUM_ENVS, new_step_api=True, autoreset=True),
        new_step_api=True,
    )

def make_play_env():
    return gym.wrappers.NormalizeObservation(
        # gym.make("LunarLander-v2", continuous=True, new_step_api=True, autoreset=True, render_mode='human'),
        # gym.make("HalfCheetah-v4", new_step_api=True, autoreset=True, render_mode='human'),
        gym.make("Walker2d-v4", new_step_api=True, autoreset=True, render_mode='human'),
        # gym.make("Hopper-v4", new_step_api=True, autoreset=True, render_mode='human'),
        new_step_api=True,
    )

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
            xavier_linear(self.input_size, 128, device=device, activation='tanh'),
            nn.Tanh(),
            xavier_linear(128, 128, device=device, activation='tanh'),
            nn.Tanh(),
        )
        self.loc_layer = xavier_linear(128, self.output_size, device=device)
        self.scale_layer = xavier_linear(128, self.output_size, device=device)

        # self.loc_layer.weight.div_(100)
        # self.loc_layer.bias.div_(100)
        # self.scale_layer.weight.div_(10)
        # self.scale_layer.bias.div_(10)

    def forward(self, x: torch.Tensor) -> Normal:
        x = self.net(x)
        return Normal(self.loc_layer(x), F.softplus(self.scale_layer(x)))

    @torch.no_grad()
    def act(self, x: torch.Tensor) -> torch.Tensor:
        dist: Normal = self(x)
        sample: torch.Tensor = dist.sample()
        tanh_sample: torch.Tensor = sample.tanh()
        return tanh_sample
        # return sample, tanh_sample, tanh_normal_log_prob(dist, sample, tanh_sample)

class QNet(nn.Module):
    net: nn.Sequential
    state_size: int
    action_size: int

    def __init__(self, state_size: int, action_size: int, device: torch.device = None):
        super().__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.net = nn.Sequential(
            nn.Linear(self.state_size + self.action_size, 256, device=device),
            nn.ReLU(),
            nn.Linear(256, 256, device=device),
            nn.ReLU(),
            nn.Linear(256, 1, device=device),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat((states, actions), -1)
        return self.net(x).squeeze(-1)

@dataclass
class Buffer:
    states: npt.NDArray[np.float32]
    actions: npt.NDArray[np.float32]
    rewards: npt.NDArray[np.float32]
    term_mask: npt.NDArray[np.bool_]
    trunc_mask: npt.NDArray[np.bool_]
    capacity: int
    start: int
    size: int

    def __len__(self):
        return self.size

    def push_idx(self) -> int:
        free_idx = (self.start + self.size) % self.capacity
        if self.size < self.capacity:
            self.size += 1
        else:
            self.start = (self.start + 1) % self.capacity
        return free_idx

@dataclass
class SacData(nn.Module):
    policy_net: Policy
    qnet1: QNet
    qnet2: QNet
    alpha: nn.Parameter

    target_qnet1: QNet
    target_qnet2: QNet

    policy_opt: optim.Adam
    qnet1_opt: optim.Adam
    qnet2_opt: optim.Adam
    alpha_opt: optim.Adam

    target_entropy: float

    def __new__(cls, *__args__, **__kwargs__):
        slf = object.__new__(cls)
        super(cls, slf).__init__()
        return slf

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

@torch.no_grad()
def update_target(net: nn.Module, target: nn.Module, tau: float = TARGET_SMOOTHING_COEF):
    for param, target_param in zip(net.parameters(), target.parameters()):
        target_param.copy_(tau * param + (1-tau) * target_param)

def sac_step(
    t: int,
    writer: SummaryWriter,
    sac_data: SacData,
    buffer: Buffer,
    device: torch.device = None,
):
    sum_policy_loss = 0.
    sum_qnet1_loss = 0.
    sum_qnet2_loss = 0.
    sum_qnet1_err = 0.
    sum_qnet2_err = 0.
    sum_entropy = 0.
    sum_value = 0.

    env_indices = rng.integers(NUM_ENVS, size=EPOCHS * BATCH_SIZE)
    indices = rng.integers(buffer.start, buffer.start + len(buffer) - 1, size=EPOCHS*BATCH_SIZE) % buffer.capacity

    for (
        states,
        actions,
        rewards,
        term_mask,
        trunc_mask,
        next_states,
    ) in zip(
        torch.tensor(buffer.states[env_indices, indices], device=device).view(EPOCHS, BATCH_SIZE, -1),
        torch.tensor(buffer.actions[env_indices, indices], device=device).view(EPOCHS, BATCH_SIZE, -1),
        torch.tensor(buffer.rewards[env_indices, indices], device=device).view(EPOCHS, BATCH_SIZE),
        torch.tensor(buffer.term_mask[env_indices, indices], device=device).view(EPOCHS, BATCH_SIZE),
        torch.tensor(buffer.trunc_mask[env_indices, indices], device=device).view(EPOCHS, BATCH_SIZE),
        torch.tensor(buffer.states[env_indices, (indices + 1) % buffer.capacity], device=device).view(EPOCHS, BATCH_SIZE, -1),
    ):
        next_policy_dist: Normal = sac_data.policy_net(next_states)
        next_atanh_actions: torch.Tensor = next_policy_dist.rsample()
        next_actions = next_atanh_actions.tanh()
        next_log_prob = tanh_normal_log_prob(next_policy_dist, next_atanh_actions, next_actions)

        alpha = F.softplus(sac_data.alpha)
        alpha_next_log_prob = alpha.detach() * next_log_prob

        with torch.no_grad():
            next_target_value = torch.minimum(
                sac_data.target_qnet1(next_states, next_actions),
                sac_data.target_qnet2(next_states, next_actions),
            ) - alpha_next_log_prob
            target_qs = rewards + DISCOUNT * term_mask * next_target_value

        num_untrunced = trunc_mask.sum()

        qnet1_err = trunc_mask * (sac_data.qnet1(states, actions) - target_qs)
        qnet1_loss = qnet1_err.square().sum() / num_untrunced / 2

        qnet2_err = trunc_mask * (sac_data.qnet2(states, actions) - target_qs)
        qnet2_loss = qnet2_err.square().sum() / num_untrunced / 2

        next_value = torch.minimum(
            sac_data.qnet1(next_states, next_actions),
            sac_data.qnet2(next_states, next_actions),
        ) - alpha_next_log_prob
        policy_loss = -next_value.mean()

        entropy = -next_log_prob.detach().mean()
        alpha_loss = alpha * (entropy - sac_data.target_entropy)

        sac_data.policy_opt.zero_grad()
        policy_loss.backward()
        sac_data.policy_opt.step()

        sac_data.qnet1_opt.zero_grad()
        qnet1_loss.backward()
        sac_data.qnet1_opt.step()

        sac_data.qnet2_opt.zero_grad()
        qnet2_loss.backward()
        sac_data.qnet2_opt.step()

        sac_data.alpha_opt.zero_grad()
        alpha_loss.backward()
        sac_data.alpha_opt.step()

        update_target(sac_data.qnet1, sac_data.target_qnet1)
        update_target(sac_data.qnet2, sac_data.target_qnet2)

        sum_policy_loss += policy_loss.detach().cpu().numpy()
        sum_qnet1_err += qnet1_err.detach().mean().cpu().numpy()
        sum_qnet2_err += qnet2_err.detach().mean().cpu().numpy()
        sum_qnet1_loss += qnet1_loss.detach().cpu().numpy()
        sum_qnet2_loss += qnet2_loss.detach().cpu().numpy()
        sum_entropy += entropy.detach().cpu().numpy()
        sum_value += next_value.detach().mean().cpu().numpy()

    writer.add_scalar("loss/policy_loss", sum_policy_loss, t)
    writer.add_scalar("loss/qnet1_loss", sum_qnet1_loss / EPOCHS, t)
    writer.add_scalar("loss/qnet2_loss", sum_qnet2_loss / EPOCHS, t)
    writer.add_scalar("loss/qnet1_err", sum_qnet1_err / EPOCHS, t)
    writer.add_scalar("loss/qnet2_err", sum_qnet2_err / EPOCHS, t)
    writer.add_scalar("metric/temperature", F.softplus(sac_data.alpha.detach()).cpu().numpy(), t)
    writer.add_scalar("metric/entropy", sum_entropy / EPOCHS, t)
    writer.add_scalar("metric/value", sum_value / EPOCHS, t)

def main():
    # torch.autograd.set_detect_anomaly(True)

    rewards: npt.NDArray[np.float32]
    terminated: npt.NDArray[np.bool_]
    truncated: npt.NDArray[np.bool_]

    env = make_env()
    assert isinstance(env.action_space, gym.spaces.Box)
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.dtype == np.float32 or env.observation_space.dtype == np.float64
    assert env.action_space.dtype == np.float32

    assert NUM_ENVS == env.action_space.shape[0] == env.observation_space.shape[0]
    _, action_size = env.action_space.shape
    _, state_size = env.observation_space.shape

    alpha = nn.Parameter(torch.tensor(-1., dtype=torch.float32, device=device))
    policy_net = Policy(state_size, action_size, device=device)
    qnet1 = QNet(state_size, action_size, device=device)
    qnet2 = QNet(state_size, action_size, device=device)
    sac_data = SacData(
        policy_net=policy_net,
        qnet1=qnet1,
        qnet2=qnet2,
        alpha=alpha,

        target_qnet1=QNet(state_size, action_size, device=device),
        target_qnet2=QNet(state_size, action_size, device=device),

        policy_opt=optim.Adam(policy_net.parameters(), lr=POLICY_LR),
        qnet1_opt=optim.Adam(qnet1.parameters(), lr=QNET_LR),
        qnet2_opt=optim.Adam(qnet2.parameters(), lr=QNET_LR),
        alpha_opt=optim.Adam([alpha], lr=ALPHA_LR),

        target_entropy=-action_size,
    )
    if RESTORE:
        state_dict = torch.load(SAVE_PATH, map_location=device)
        sac_data.load_state_dict(state_dict['data'])
        env.obs_rms.mean = state_dict['obs_rms.mean']
        env.obs_rms.var = state_dict['obs_rms.var']
        env.obs_rms.count = state_dict['obs_rms.count']
    else:
        sac_data.target_qnet1.load_state_dict(qnet1.state_dict())
        sac_data.target_qnet2.load_state_dict(qnet2.state_dict())

    buffer = Buffer(
        states      = np.zeros((NUM_ENVS, BUFFER_SIZE, state_size), dtype=np.float32),
        actions     = np.zeros((NUM_ENVS, BUFFER_SIZE, action_size), dtype=np.float32),
        rewards     = np.zeros((NUM_ENVS, BUFFER_SIZE), dtype=np.float32),
        term_mask  = np.zeros((NUM_ENVS, BUFFER_SIZE), dtype=np.bool_),
        trunc_mask   = np.zeros((NUM_ENVS, BUFFER_SIZE), dtype=np.bool_),

        capacity    = BUFFER_SIZE,
        start       = 0,
        size        = 0,
    )

    fitness: npt.NDArray[np.float32] = np.zeros(env.num_envs, dtype=np.float32)
    states: npt.NDArray[np.float32] = env.reset().astype(np.float32)

    log_dir: str = 'data/logs/' + LOG_DIR + datetime.now().strftime('%Y%m%d-%H%M%S')
    writer = SummaryWriter(log_dir)

    print()
    print("-" * 80)
    print("Starting now")
    for t in count(step=NUM_ENVS * ENV_STEPS_PER_TIMESTEP):
        for _ in range(ENV_STEPS_PER_TIMESTEP):
            t += NUM_ENVS

            with torch.no_grad():
                states_tch = torch.tensor(states, device=device)
                actions_tch = policy_net.act(states_tch)
                actions: npt.NDArray[np.float32] = actions_tch.cpu().numpy()

            i = buffer.push_idx()
            buffer.states[:, i]     = states
            buffer.actions[:, i]    = actions

            states, rewards, terminated, truncated, _ = env.step(map_actions(actions, env.action_space))
            states = states.astype(np.float32)
            fitness += rewards

            buffer.rewards[:, i]    = rewards
            buffer.term_mask[:, i]  = ~terminated
            buffer.trunc_mask[:, i] = ~truncated

            done: bool
            for i, done in enumerate(terminated | truncated):
                t += 1
                if done:
                    writer.add_scalar("metric/fitness", fitness[i], t)
                    fitness[i] = 0

        if BUFFER_MIN_LEN < len(buffer):
            sac_step(t, writer, sac_data, buffer, device=device)
            if not t % 48:
                torch.save(
                    {
                        'data': sac_data.state_dict(),
                        'obs_rms.mean': env.obs_rms.mean,
                        'obs_rms.var': env.obs_rms.var,
                        'obs_rms.count': env.obs_rms.count,
                    },
                    SAVE_PATH,
                )


if __name__ == '__main__':
    main()
