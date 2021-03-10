import torch
import torch.nn as nn

import numpy as np


def count(module: object):
    """
        Returns a count of the total number of parameters
        in a module
    """

    return np.sum([np.prod(p.shape) for p in module.parameters()])


class MLPActor(nn.Module):
    """
        Policy: Selects actions given states
    """

    def __init__(self, obs_dim: int, act_dim: int,  hidden_sizes: list,  activation: object):
        super(MLPActor, self).__init__()

        self.pi = mlp(obs_dim, hidden_sizes + [act_dim],
                      activation=activation, output_activation=nn.Tanh)

    def forward(self, state):
        return self.pi(state)


class MLPCritic(nn.Module):
    """
        Q(s, a): State-Action value function
    """

    def __init__(self, obs_dim: int, act_dim: int,  hidden_sizes: list,  activation: object):
        self.q = mlp(obs_dim + act_dim,  hidden_sizes + [1],  activation)

    def forward(self, obs, action):
        obs_act = torch.cat([obs, action], dim=-1)

        return self.q(obs_act).squeeze(-1)


class MLPActorCritic(nn.module):
    def __init__(self, obs_dim: int, act_dim: int, activation: object = nn.ReLU,
                 hidden_sizes: list = [64, 64],
                 size: int = 2):
        super(MLPActorCritic, self).__init__()

        self.q_1 = MLPCritic(obs_dim, act_dim, hidden_sizes, activation)
        self.q_2 = MLPCritic(obs_dim, act_dim, hidden_sizes, activation)

        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        """
            Select action for given observation with
            the current policy
        """
        with torch.no_grad():
            return self.pi(obs).numpy().cpu()
