import gym
import gym.spaces
import torch
import numpy as np
from itertools import count

import ppo

@torch.no_grad()
def main():
    device = torch.device('cpu')

    env = ppo.make_play_env()
    assert isinstance(env.action_space, gym.spaces.Box)
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.dtype == np.float32
    assert env.action_space.dtype == np.float32

    action_size, = env.action_space.shape
    state_size, = env.observation_space.shape

    policy_net = ppo.Policy(state_size, action_size, device=device).eval()
    value_net = ppo.Value(state_size, device=device).eval()

    state_dict = torch.load(ppo.SAVE_PATH, map_location=device)
    policy_net.load_state_dict(state_dict['policy_net'])
    value_net.load_state_dict(state_dict['value_net'])

    fitness = 0.
    state = env.reset().astype(np.float32)
    for t in count():
        state_tch = torch.tensor(state, device=device).unsqueeze(0)
        _, action_tch, log_prob_tch = policy_net.act(state_tch)
        value_tch: torch.Tensor = value_net(state_tch)

        prob: float = log_prob_tch.squeeze().cpu().exp().numpy()
        value: float = value_tch.squeeze().cpu().numpy()

        truncated: bool
        state, reward, terminated, truncated, _ = env.step(action_tch.squeeze(0).cpu().numpy()) # type: ignore
        state = state.astype(np.float32)

        fitness += reward

        if not t % 8:
            print(f"prob={prob} reward_sum={fitness} expected_value={value} expected_discounted_fitness={fitness + value}", end="\r")

        if terminated or truncated:
            print(f"{'TERMINATED' if terminated else 'TRUNCATED'} END: fitness={fitness}", " " * 100)
            print()
            fitness = 0.

            state_dict = torch.load(ppo.SAVE_PATH, map_location=device)
            policy_net.load_state_dict(state_dict['policy_net'])
            value_net.load_state_dict(state_dict['value_net'])

if __name__ == '__main__':
    main()
