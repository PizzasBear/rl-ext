import gym
import gym.spaces
import numpy as np
import torch
import torch.nn as nn
from time import sleep
from torch.distributions import Normal
from itertools import count

import sac

class SacPlayData(nn.Module):
    def __init__(self, state_size: int, action_size: int, device: torch.device = None):
        super().__init__()
        self.policy_net = sac.Policy(state_size, action_size, device=device)
        self.qnet1 = sac.QNet(state_size, action_size, device=device)
        self.qnet2 = sac.QNet(state_size, action_size, device=device)
        self.alpha = nn.Parameter(torch.tensor(-1., dtype=torch.float32, device=device))

@torch.no_grad()
def main():
    device = torch.device('cpu')

    env = sac.make_play_env()
    assert isinstance(env.action_space, gym.spaces.Box)
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.dtype == np.float32 or env.observation_space.dtype == np.float64
    assert env.action_space.dtype == np.float32

    action_size, = env.action_space.shape
    state_size, = env.observation_space.shape

    sac_data = SacPlayData(state_size, action_size, device=device).eval()
    state_dict = torch.load(sac.SAVE_PATH, map_location=device)
    missing_keys, _ = sac_data.load_state_dict(state_dict['data'], strict=False)
    assert not len(missing_keys)
    env.obs_rms.mean = state_dict['obs_rms.mean']
    env.obs_rms.var = state_dict['obs_rms.var']
    env.obs_rms.count = state_dict['obs_rms.count']

    fitness = 0.
    state = env.reset().astype(np.float32)
    for t in count():
        state_tch = torch.tensor(state, device=device).unsqueeze(0)

        policy_dist: Normal = sac_data.policy_net(state_tch)
        atanh_action_tch: torch.Tensor = policy_dist.sample()
        action_tch: torch.Tensor = atanh_action_tch.tanh()
        log_prob_tch = sac.tanh_normal_log_prob(policy_dist, atanh_action_tch, action_tch)

        value_tch: torch.Tensor = torch.minimum(
            sac_data.qnet1(state_tch, action_tch),
            sac_data.qnet2(state_tch, action_tch),
        ) - sac_data.alpha * log_prob_tch

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

            try:
                state_dict = torch.load(sac.SAVE_PATH, map_location=device)
                missing_keys, _ = sac_data.load_state_dict(state_dict['data'], strict=False)
                if missing_keys:
                    raise RuntimeError(f"missing keys {repr(missing_keys)}")
                env.obs_rms.mean = state_dict['obs_rms.mean']
                env.obs_rms.var = state_dict['obs_rms.var']
                env.obs_rms.count = state_dict['obs_rms.count']
            except Exception as err:
                print(f"Failed loading \"{sac.SAVE_PATH}\" with {repr(err)}")
                print(f"  state_dict.keys()={state_dict.keys()}")
                if 'data' in state_dict.keys():
                    print(f"  state_dict['data'].keys()={state_dict['data'].keys()}")
                sleep(0.05)
        # sleep(1 / 60)

if __name__ == '__main__':
    main()
