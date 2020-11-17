import copy

import numpy as np
import torch

import core


class ReplayBuffer:
    """
        Memory for transitions
    """

    def __init__(self, size=int(1e4), *, act_dim: int, obs_dim: int, device: object):
        self.obs = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs_n = np.zeros([size, obs_dim], dtype=np.float32)
        self.act = np.zeros([size,  act_dim], dtype=np.float32)
        self.rewards = np.zeros([size], dtype=np.float32)
        self.dones = np.zeros([size], dtype=np.float32)

        self.idx = 0
        self.size = size
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

        self.idx = (self.idx + 1) % self.size

    def sample(self, batch_size):
        """
            Samples a batch of transitions

            Returns: dict of transitions
        """

        idx = np.random.randint(self.size, size=batch_size)

        samples = {'obs': self.obs[idx],
                   'obs_n': self.obs_n[idx],
                   'act': self.act[idx],
                   'rew': self.rewards[idx],
                   'dones': self.dones[idx]
                   }
        samples = {
            k: torch.from_numpy(v).to(
                self.device) for k, v in samples
        }

        return samples


def ddpg(env, ac_kwargs={}, memory_size=int(1e5), actor_critic=core.MLPActorCritic):
    """
        Args
        ---

        env: Gym env
        ac_kwargs (dict): Actor Critic module parameters
        memory_size (int): Replay buffer limit for transition
            storage
    """

    device = torch.device('cpu' if not torch.cuda.is_available else 'gpu')

    act_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    act_limit = env.action_space.high[0]

    rep_buffer = ReplayBuffer(size=memory_size,
                              act_dim=act_dim,
                              obs_dim=obs_dim, device=device)
    actor_critic = actor_critic(obs_dim,
                                act_dim, act_limit,
                                **ac_kwargs)
    q, pi = actor_critic.q, actor_critic.pi

    ac_target = copy.deepcopy(actor_critic)
    q_target, pi_target = ac_target.q, ac_target.pi

    for param in q_target:
        param.grad = None

    for param in pi_target:
        param.grad = None
