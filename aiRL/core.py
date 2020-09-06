"""
    g_args: hidden_layers=[32, 1], activation=nn.Identity
    h_args: hidden_layers=[32, 32, 1], activation=nn.ReLu
"""
import torch.nn as nn
import torch

import numpy as np


def count(module):
    """
        Returns a count of the parameters
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


class Discriminator(nn.Module):
    """
        Disctimates between expert data and samples from
        learned policy.
        It recovers the advantage f_theta_phi  used in training the
        policy.

        The Discriminator has:

        g_theta(s): A state-only dependant reward function
            This allows extraction of rewards that are disentangled
            from the dynamics of the environment in which they were trained

        h_phi(s): The shaping term
                  Mitigates unwanted shaping on the reward term g_theta(s)

        f_theta_phi = g_theta(s) + gamma * h_phi(s') - h_phi(s)
        (Essentially an advantage estimate)
    """

    def __init__(self, obs_dim, gamma=.99, **args):
        super(Discriminator, self).__init__()
        self.gamma = gamma

        # *g(s) = *r(s) + const
        #  g(s) recovers the optimal reward function +  c
        self.g_theta = torch.squeeze(mlp(obs_dim, **args['g_args']))

        # *h(s) = *V(s) + const (Recovers optimal value function + c)
        self.h_phi = torch.squeeze(mlp(**args['h_args']))

    def forward(self, data):
        """
            Returns the estimated reward function / Advantage
            estimate
        """
        obs, obs_n = data
        f_thet_phi = self.g_theta(obs) + self.gamma * \
            self.h_phi(obs_n) - self.h_phi(obs)

        return f_thet_phi
