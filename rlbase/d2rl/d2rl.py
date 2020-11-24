"""
    D2RL Actor and Critic Architectures
"""
from typing import Optional, TypeVar, Iterable, Tuple

import torch
import torch.nn as nn

T = TypeVar('T')


def d2rl_mlp(x, hidden_layers, activation=nn.Tanh, size=2, output_activation=nn.Identity):
    """
        Multi-layer perceptron

        The architecure is created by concatenating
        the output of each FC with the input.

        The result is then fed into the next layer
        and the process repeated
    """
    net_layers = []

    if len(hidden_layers[:-1]) < size:
        hidden_layers[:-1] *= size

    input_d = x

    for size in hidden_layers[:-1]:
        layer = nn.Linear(x, size)
        net_layers.append(layer)
        net_layers.append(activation())
        x = size + input_d

    net_layers.append(nn.Linear(hidden_layers[-2], hidden_layers[-1]))
    net_layers += [output_activation()]

    return nn.Sequential(*net_layers)


class PolicyMLP(nn.Module):
    """
        Actor Class
    """

    def __init__(self, obs_dim: int, act_dim: int, act_limit: int, *,
                 hidden_sizes: list, activation: Optional[object] = nn.ReLU,
                 output_activation: Optional[object] = nn.Identity,
                 size: Optional[int] = 4):
        super(PolicyMLP, self).__init__()

        self.pi = d2rl_mlp(obs_dim, hidden_sizes + [act_dim], activation,
                           size, output_activation)
        self.act_limit = act_limit

    def forward(self, obs):
        """
            Predict action for given observation
        """
        # Don't concat last two layers
        # Remeber each layer has an activation function
        # So count = 4

        in_put = obs.clone().detach()

        for fc_layer, activation_f in group_layers(self.pi[:-4]):
            x = fc_layer(obs)
            x = activation_f(x)
            x = torch.cat([x, in_put], dim=-1)

            obs = x

        for layer in self.pi[-4:]:
            x = layer(x)

        act = x

        return act * self.act_limit


class MLPQFunction(nn.Module):
    """
        Q Function (D2RL)

        The action-value funtion
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 *,
                 hidden_sizes: list, activation: Optional[object] = nn.ReLU,
                 output_activation: Optional[object] = nn.Identity,
                 size: Optional[int] = 4):
        super(MLPQFunction, self).__init__()

        self.q = d2rl_mlp(act_dim + obs_dim, hidden_sizes + [1],
                          activation, size, output_activation)

    def forward(self, obs, act):
        inpt = torch.cat([obs, act], dim=-1)

        inpt_d = inpt.clone().detach()

        for fc_layer, activation_f in group_layers(self.q[:-4]):
            x = activation_f(fc_layer(inpt))
            x = torch.cat([x, inpt_d], dim=1)

            inpt = x

        # No concat for last 2 layers
        for layer in self.q[-4:]:
            x = layer(x)

        out = x.squeeze()

        return out


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, act_limit: int,
                 hidden_sizes: Optional[list] = [64, 64, 64, 64, 64],
                 activation: Optional[object] = nn.ReLU,
                 output_activation: Optional[object] = nn.Identity,
                 size: Optional[int] = 5):
        super(MLPActorCritic, self).__init__()

        self.q = MLPQFunction(obs_dim, act_dim,
                              hidden_sizes=hidden_sizes,
                              activation=activation,
                              size=size)
        self.pi = PolicyMLP(obs_dim, act_dim, act_limit,
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


def group_layers(iterable: Iterable[T], n=2) -> Iterable[Tuple[T, ...]]:
    """
        Create iteration for D2RL forward pass.

        This returns an iterator that gives every two
        sequential elements grouped together. It's
        useful for isolating Activation layers
        from FCs in a Sequential model.

        s -> (s0, s1, s1, s3) -> ((s0, s1), (s2, s3))

        s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), ...
    """
    return zip(*[iter(iterable)] * n)
