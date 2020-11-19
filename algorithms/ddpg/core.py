import torch.nn as nn
import torch


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
        Policy: Selects an action for each observation
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


class MLPQ(nn.Module):
    """
        Q Function: Action value function
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list, activation: object):
        super(MLPQ, self).__init__()

        self.q = mlp([obs_dim + act_dim], hidden_sizes + [1], activation)

    def forward(self, obs, act):
        inpt = torch.cat([obs, act], dim=-1)

        return self.q(inpt).squeeze(-1)


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, act_limit: float, activation: object = nn.ReLU, hidden_sizes: list = [64, 64]):
        super(MLPActorCritic, self).__init__()
        self.q = MLPQ(obs_dim, act_dim, hidden_sizes, activation=activation)
        self.pi = MLPActor(obs_dim,
                           act_dim,
                           act_limit=act_limit,
                           activation=activation,
                           hidden_sizes=hidden_sizes)

    def act(self, obs):
        """
            Predict action for given observation
        """

        with torch.no_grad():
            return self.pi(obs).numpy()
