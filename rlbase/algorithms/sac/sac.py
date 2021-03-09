import torch
import numpy as np


import copy


class ReplayBuffer:
    """
        Memory for transitions
    """

    def __init__(self, size=int(1e5), *, act_dim: int, obs_dim: int, device: object):
        self.obs = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs_n = np.zeros([size, obs_dim], dtype=np.float32)
        self.act = np.zeros([size,  act_dim], dtype=np.float32)
        self.rewards = np.zeros([size], dtype=np.float32)
        self.dones = np.zeros([size], dtype=np.float32)

        self.idx = 0
        self.max_size = size
        self.size = 0
        self.device = device

    def store(self, obs, act, obs_n, rew, done):
        """
            Store transition in buffer
        """
        idx = self.idx
        self.obs[idx] = obs
        self.obs_n[idx] = obs_n
        self.act[idx] = act
        self.dones[idx] = done
        self.rewards[idx] = rew

        self.idx = (self.idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """
            Samples a batch of transitions

            Returns: dict of transitions
        """

        idx = np.random.randint(0, self.size, size=batch_size)

        samples = {'obs': self.obs[idx],
                   'obs_n': self.obs_n[idx],
                   'act': self.act[idx],
                   'rew': self.rewards[idx],
                   'dones': self.dones[idx]
                   }
        samples = {
            k: torch.from_numpy(v).to(
                self.device) for k, v in samples.items()
        }

        return samples


def sac(env, ac_kwargs={}, actor_critic=None, memory_size: int = int(1e6),
        seed: int = 1, **args):
    """
        Soft Actor Critic

        Args
        ----
        ac_kwargs (dict):  Actor critic parameters

        memory_size (int): Replay buffer transition count limit

    """

    np.random.seed(seed)
    torch.manual_seed(seed)

    device_nm = args.get('device') or (
        'cuda' if torch.cuda.is_available else 'cpu')
    device = torch.device(device_nm)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Init pi and Qs parameters
    actor_critic = actor_critic(obs_dim, act_dim, **ac_kwargs).to(device)
    ac_target = copy.deepcopy(actor_critic).to(device)
