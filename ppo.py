import numpy as np
import torch

from .core import discounted_cumsum


class ReplayBuffer:
    """
        Transitions buffer
        Stores transitions for a single episode
    """

    def __init__(self, size=4000, gamma=.98, lamda=.95):
        self.size = size
        self.gamma = gamma
        self.lamda = lamda

        self.rewards = []
        self.actions = []
        self.states = []
        self.next_states = []

        self.log_prob = []
        self.adv = []
        self.vals = []

        self.ptr, self.eps_end_ptr = 0, 0

    def store(self, act, states, values, rew):
        """
            Store transitions
        """
        idx = self.ptr % self.size

        self.rewards[idx] = rew
        self.actions[idx] = act
        self.states[idx] = states
        self.vals[idx] = values

        self.ptr += 1

    def get(self):
        """
            Returns episode transitions
        """
        assert self.ptr >= self.size

        self.ptr = 0
        self.eps_end_ptr = 0
        return torch.as_tensor(self.actions), torch.as_tensor(self.rewards), \
            torch.as_tensor(self.states), torch.as_tensor(
            self.adv), torch.as_tensor(self.log_prob)

    def end_eps(self, value=0):
        """
            Calculates the adv once the agent
            encounters an end state

            value: value of that state -> zero if the agent
            died or the value function if the episode was terminated
        """
        idx = slice(self.eps_end_ptr, self.ptr)

        rew = np.append(self.rewards, value)
        vals = np.append(self.vals, value)

        # GAE
        deltas = rew[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv[idx] = discounted_cumsum(deltas, self.gamma * self.lamda)

        # Reward to go
        self.rewards[idx] = discounted_cumsum(rew, self.gamma)[:-1]

        self.eps_end_ptr = self.ptr
