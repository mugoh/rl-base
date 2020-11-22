"""
    D2RL Actor and Critic Architectures
"""
from typing import Optional

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
        for layer in self.pi[-1]:
            x = layer(obs)
            obs = torch.cat([x, obs], dim=-1)

        act = self.pi[-1](obs)

        return act
