import numpy as np
import torch


class ReplayBuffer:
    """
        Transitions buffer
        Stores transitions for a single episode
    """

    def __init__(self, act_dim, obs_dim, max_size=1000000):

        self.rewards = np.zeros([size], dtype=np.float32)
        self.actions = np.zeros([size, act_dim], dtype=np.float32)
        self.states = np.zeros([size, obs_dim], dtype=np.float32)
        self.n_states = np.zeros([size, obs_dim], dtype=np.float32)

        self.dones = np.zeros([size], dtype=np.float32)
        self.log_prob = np.zeros([size], dtype=np.float32)

        self.ptr, self.size = 0, 0
        self.max_size = max_size

    def store(self, act, states, n_states,  rew, dones, log_p):
        """
            Store transitions
        """
        idx = self.ptr % self.max_size

        self.rewards[idx] = rew
        self.actions[idx] = act
        self.states[idx] = states
        self.n_states[idx] = n_states
        self.log_prob[idx] = log_p
        self.dones[idx] = dones

        self.ptr += 1
        self.size = min(self.ptr + 1, self.max_size)

    def sample_recent(self, batch_size):
        """
            Returns recent transitions of size batch_size
            in order: act, rew, obs, n_obs, dones, log_p
        """
        assert self.ptr >= batch_size

        return (
            torch.as_tensor(
                self.actions[-batch_size:], dtype=torch.float32),
            torch.as_tensor(
                self.rewards[-batch_size:], dtype=torch.float32),
            torch.as_tensor(
                self.states[-batch_size:], dtype=torch.float32),
            torch.as_tensor(
                self.n_states[-batch_size:], dtype=torch.float32),
            torch.as_tensor(
                self.dones[-batch_size:], dtype=torch.float32),
            torch.as_tensor(
                self.log_prob[-batch_size:], dtype=torch.float32)
        )

    def sample_random(self, batch_size, itr_limit=20):
        """
            Randomly sample trajectories upto the most recent
            itr_limit iterations

            in order: act, rew, obs, n_obs, dones, log_p
        """

        lowest_iter = itr_limit * batch_size
        low_ = 0
        if self.ptr > lowest_iter:
            low_ = lowest_iter
        idx = np.random.randint(
            low=self.ptr - low_, high=self.ptr, size=batch_size)

        return (
            torch.as_tensor(
                self.actions[idx], dtype=torch.float32),
            torch.as_tensor(
                self.rewards[idx], dtype=torch.float32),
            torch.as_tensor(
                self.states[idx], dtype=torch.float32),
            torch.as_tensor(
                self.n_states[idx], dtype=torch.float32),
            torch.as_tensor(
                self.dones[idx], dtype=torch.float32),
            torch.as_tensor(
                self.log_prob[idx], dtype=torch.float32)
        )
