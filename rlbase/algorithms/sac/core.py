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


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class MLPActor(nn.Module):
    """
        Policy: Selects actions given states
        u_theta(s) + sigma_theta(s) * noise
    """

    def __init__(self, obs_dim: int, act_dim: int,  hidden_sizes: list,  activation: object, act_limit: float):
        super(MLPActor, self).__init__()

        self.logits = mlp(obs_dim, hidden_sizes,
                          activation=nn.ReLU)

        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)

        self.squash_f = torch.tanh
        self._act_dim = act_dim
        self.act_limit = act_limit

    def sample_policy(self, state, mean_act: bool = False):
        """Return a new policy from the given state
           At test time, we use the mean action,
           instead of sampling from the distribution.

           The spinning up docs say this improves performance
           over the original stochastic policy
        """
        act_pred = self.logits(state)
        mu = self.mu_layer(act_pred)

        if mean_act:
            return mu
        log_std = self.log_std_layer(act_pred)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        pi = distributions.Normal(loc=mu,
                                  scale=torch.exp(
                                      log_std))

        return pi

    def forward(self, obs, mean_act=False, return_pi=False):
        """Sample action

           mean_act (bool): If True, sample action from normal
            distribution without stochasticity
        """

        if mean_act:
            action = self.sample_policy(obs, mean_act)
        else:
            pi_new = self.sample_policy(obs)
            action = pi_new.rsample()

        if return_pi:
            log_pi = self.log_p(pi_new, action)
        else:
            log_pi = None

        action = self.squash_f(action)
        action = self.act_limit * action

        return action, log_pi

    def log_p(self, pi, act):
        """Compute log probabilities of the policy w.r.t actions"""
        # log_mu - \sum_i=1^D[1 - tanh^2(mu_i)]
        log_pi = pi.log_prob(act).sum(axis=-1)
        log_pi -= (2*(np.log(2) - act -
                   nn.functional.softplus(-2 * act))).sum(axis=-1)

        return log_pi


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


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, act_limit: float, activation: object = nn.ReLU,
                 hidden_sizes: list = [64, 64],
                 size: int = 2):
        super(MLPActorCritic, self).__init__()

        self.q_1 = MLPCritic(
            obs_dim, act_dim, hidden_sizes, activation)
        self.q_2 = MLPCritic(
            obs_dim, act_dim, hidden_sizes, activation)

        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes,
                           activation, act_limit)

    def act(self, obs, mean_act=False):
        """
            Select action for given observation with
            the current policy
        """
        with torch.no_grad():
            act = self.pi(obs, mean_act)

            act, log_pi = act
            act = act.cpu().numpy()
            return act, log_pi
