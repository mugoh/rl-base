import torch
from torch import nn
import torch.nn.functional as F

import numpy as np



def mlp(x, hidden_layers, activation=nn.tanh, size=2, output_activation=nn.Identity):
    """
        Multi-layer perceptron
    """
    net_layers = []

    if len(hidden_layers) < size:
        hidden_layers *= size

    for layer in hidden_layers[:-1]:
        x = nn.Linear(x, layer)
        net_layers.append(x, activation())

    net_layers.append(nn.Linear(x, hidden_layers[-1], output_activation()))

    return nn.Sequential(*net_layers)


class MLPGaussianPolicy(nn.Module):
    """
        Gaussian Policy for stochastic actions
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=F.tanh):
        super(MLPGaussianPolicy, self).__init__()

        self.logits = mlp(obs_dim, hidden_sizes + [act_dim], activation)
        self.log_std = nn.Parameter(torch.as_tensor(np.zeros(act_dim)))

    def sample_action(self, obs):
        """
            Creates a normal distribution representing
            the current policy
            Returns an action on the policy given the observation
        """
        mu = self.logits(obs)
        pi = torch.normal(mu, torch.exp(self.log_std))

        return pi.sample()

    @property
    def log_p(self, pi, a):
        """
            The log probability of taken action
            a in policy pi
        """
        return pi.log_prob(a).sum(axis=-1) # Sum needed for Torch normal distr.

