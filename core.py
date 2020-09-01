import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from gym.spaces import Discrete



def mlp(x, hidden_layers, activation=nn.Tanh, size=2, output_activation=nn.Identity):
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

class MLPCritic(nn.Module):
    """
        Agent Critic
        Estmates value function
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, size=2):
        super(MLPCritic, self).__init__()

        self.net = mlp(obs_dim, hidden_sizes + [1], activation=activation, size=size)
    def forward(self, obs):
        """
            Get value function estimate
        """
        return torch.squeeze(self.net(obs), axis=-1)



class Actor(nn.Module):
    def __init__(self, **args):
        super(Actor, self).__init__()
    def forward(self, obs, ac=None):
        """
            Gives policy under current observation
            and optionally action log prob from that
            policy
        """
        pi = self.sample_action(obs)
        log_p = None

        if ac is not None:
            log_p = self.log_p(pi, ac)
        return pi, log_p


class MLPGaussianPolicy(Actor):
    """
        Gaussian Policy for stochastic actions
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.Tanh, size=2):
        super(MLPGaussianPolicy, self).__init__()

        self.logits = mlp(obs_dim, hidden_sizes + [act_dim], activation, size=size)
        self.log_std = nn.Parameter(torch.as_tensor(np.zeros(act_dim)))

    def sample_action(self, obs):
        """
            Creates a normal distribution representing
            the current policy which
            if sampled, returns an action on the policy given the observation
        """
        mu = self.logits(obs)
        pi = torch.normal(mu, torch.exp(self.log_std))

        return pi

    @classmethod
    def log_p(cls, pi, a):
        """
            The log probability of taken action
            a in policy pi
        """
        return pi.log_prob(a).sum(axis=-1) # Sum needed for Torch normal distr.


class CategoricalPolicy(Actor):
    """
        Categorical Policy for discrete action spaces
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.Tanh, size=2):
        super(CategoricalPolicy, self).__init__()

        self.logits = mlp(obs_dim, hidden_sizes + [act_dim], activation, size=size)
    def sample_action(self, obs):
        """
            Get new policy
        """
        logits = self.logits(obs)
        pi = torch.distributions.Categorical(logits)

        return torch.distributions.Categorical(logits)

    @classmethod
    def log_p(cls, p, a):
        """
            Log probabilities of actions w.r.t pi
        """

        return p.log_prob(a)


class MLPActor(nn.Module):
    """
        Agent actor Net
    """
    def __init__(self, obs_space, act_space, hidden_size=[32, 32], activation=nn.Tanh, size=2):
        super(Actor, self).__init__()

        obs_dim = obs_space.shape[0]

        discrete = True if isinstance(act_space, Discrete) else False
        act_dim = act_space.n if discrete else act_space.shape[0]


        if discrete:
            self.pi = CategoricalPolicy(obs_dim, act_dim, hidden_size, size=size, activation=activation)
        else:
            self.pi = MLPGaussianPolicy(obs_dim, act_dim, hidden_size,activation=activation, size=size)


            self.v =MLPCritic(obs_dim, act_dim, hidden_sizes=hidden_size, size=size, activation=activation)

    def step(self, obs):
        """
            Get value function estimate and action sample from pi
        """
        with torch.no_grad():
            pi_new  = self.sample_action()
            a = pi_new.sample()

            v = self.v(obs)
            log_p = self.pi.log_p(pi_new, a)

        return a.numpy(), v.numpy(), log_p.numpy()
