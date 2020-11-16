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


class MLPActor(nn.Module):
    """
        Policy
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list, activation, act_limit: float):
        super(MLPActor, self).__init__()

        self.pi = mlp(
            obs_dim,
            hidden_sizes + [act_dim],
            activation=activation,
            output_activation=nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        return self.pi(obs) * self.act_limit
