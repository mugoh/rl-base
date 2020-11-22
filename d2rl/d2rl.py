"""
    D2RL Actor and Critic Architectures
"""
from typing import Optional

import torch
import torch.nn as nn


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


class PolicyMLP(nn.Module):
    """
        Actor Class
    """

    def __init__(self, obs_dim: int, act_dim: int, *,
                 hidden_sizes: list, activation: Optional[object] = nn.ReLU,
                 output_activation: Optional[object] = nn.Identity,
                 size: Optional[int] = 4):
        super(PolicyMLP, self).__init__()

        self.pi = mlp(obs_dim, hidden_layers + [act_dim], activation,
                      size, output_activation)

    def forward(self, obs):
        """
            Predict action for given observation
        """
        # Don't concat last two layers
        # Remeber each layer has an activation function
        # So count = 4
        for layer in self.pi[:-4]:
            x = layer(obs)
            x = torch.cat([x, obs], dim=-1)

        for layer in self.pi[:-4]:
            x = layer(x)

        act = x
        return act


class MLPQFunction(nn.Module):
    """
        Q Function (D2RL)

        The action-value funtion
    """

    def __init__(self, obs_dim: int, act_dim: int, *,
                 hidden_sizes: list, activation: Optional[object] = nn.ReLU,
                 output_activation: Optional[object] = nn.Identity,
                 size: Optional[int] = 4):
        super(MLPQFunction, self).__init__()

        self.q = mlp(act_dim + obs_dim, hidden_sizes + [1],
                     activation, size, output_activation)

    def forward(self, obs, act):
        inpt = torch.cat([obs, act], dim=-1)

        for layer in self.q[:-4]:
            x = layer(inpt)
            x = torch.cat([x, inpt], dim=-1)

        # No concat for last 2 layers
        for layer in self.q[-4:]:
            x = layer(x)

        out = x.unsqueeze()

        return out


class MLPActorCritic(nn.module):
    def __init__(self, obs_dim: int, act_dim: int,
                 hidden_sizes: Optional[list] = [64] * 5,
                 activation: Optional[object] = nn.ReLU,
                 output_activation: Optional[object] = nn.Identity,
                 size: Optional[int] = 5):
        self.q = MLPQFunction(obs_dim, act_dim,
                              hidden_sizes=hidden_sizes,
                              activation=activation,
                              size=size)
        self.pi = PolicyMLP(obs_dim, act_dim,
                            hidden_sizes=hidden_sizes,
                            activation=activation,
                            output_activation=nn.Tanh,
                            size=size)

    def act(self, obs):
        """
            Predict action for observation
        """
        with torch.no_grad():
            act = self.pi(obs)

        return act
