import torch
import torch.nn as nn
from torch import distributions

import numpy as np


def count(module: object):
    """
        Returns a count of the total number of parameters
        in a module
    """

    return np.sum([np.prod(p.shape) for p in module.parameters()])


def mlp(x, hidden_layers, activation=nn.Tanh, size=2, output_activation=nn.Identity):
    """
        Multi-layer perceptron
    """
    net_layers = []

    if len(hidden_layers[:-1]) < size:
        hidden_layers[:-1] *= size

    for size in hidden_layers[:-1]:
        layer = nn.Linear(x, size)
        net_layers.append(layer)
        net_layers.append(activation())
        x = size

    net_layers.append(nn.Linear(x, hidden_layers[-1]))
    net_layers += [output_activation()]

    return nn.Sequential(*net_layers)


class MLPActor(nn.Module):
    """
        Policy: Selects actions given states
        u_theta(s) + sigma_theta(s) * noise
    """

    def __init__(self, obs_dim: int, act_dim: int,  hidden_sizes: list,  activation: object):
        super(MLPActor, self).__init__()

        self.logits = mlp(obs_dim, hidden_sizes + [act_dim],
                          activation=nn.Tanh)

        self.log_std = mlp(obs_dim, hidden_sizes + [act_dim])
        self.squash_f = nn.Tanh()
        self._act_dim = act_dim

    def sample_policy(self, state):
        """Return a new policy from the given state"""
        # TODO Comput noise
        mu = self.logits(state)
        noise = np.random.normal(size=self._act_dim)
        pi = distributions.Normal(loc=mu,
                                  scale=torch.exp(self.sigma * noise))

        return pi

    def forward(self, obs):
        pi_new = self.pi.sample_policy(obs)
        act = pi_new.sample()

        return self.squash_f(act)

    @classmethod
    def log_p(cls, pi, act):
        """Compute log probabilities of the policy w.r.t actions"""
        return pi.log_prob(act).sum(axis=-1)


class MLPCritic(nn.Module):
    """
        Q(s, a): State-Action value function
    """

    def __init__(self, obs_dim: int, act_dim: int,  hidden_sizes: list,  activation: object):
        super(MLPCritic, self).__init__()
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
            act = self.pi(obs)

        return act.numpy().cpu()
