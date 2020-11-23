"""
    g_args: hidden_layers=[32, 1], activation=nn.Identity
    h_args: hidden_layers=[32, 32, 1], activation=nn.ReLu

    TOLOG: diff: f*(s, a) and log*pi(a|s). Should be equal
"""
import torch.nn as nn
import torch

import numpy as np

from gym.spaces import Discrete

EPS = 1e-8


def count(module):
    """
        Returns a count of the parameters
        in a module
    """
    return np.sum([np.prod(p.shape) for p in module.parameters()])


def mlp(x,
        hidden_layers,
        activation=nn.Tanh,
        size=2,
        output_activation=nn.Identity):
    """
        Multi-layer perceptron
    """
    net_layers = []

    if len(hidden_layers[:-1]) < size:
        hidden_layers[:-1] *= size

    for size in hidden_layers[:-1]:
        layer = nn.Linear(x, size)
        net_layers.append(layer)

        # For discriminator
        if activation.__name__ == 'ReLU':
            net_layers.append(activation(inplace=True))
        elif activation.__name__ == 'LeakyReLU':
            net_layers.append(activation(.2, inplace=True))
        else:
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
        self.g_theta = mlp(obs_dim, **args['g_args'])

        # *h(s) = *V(s) + const (Recovers optimal value function + c)
        self.h_phi = mlp(obs_dim, **args['h_args'])
        self.sigmoid = nn.Sigmoid()

    def forward(self, *data):
        """
            Returns the estimated reward function / Advantage
            estimate. Given by:

            f(s, a, s') = g(s) + gamma * h(s') - h(s)


            Parameters
            ----------
            data    | [obs, obs_n, dones]
        """
        obs, obs_n, dones = data
        g_s = torch.squeeze(self.g_theta(obs), axis=-1)

        shaping_term = self.gamma * \
            (1 - dones) * self.h_phi(obs_n).squeeze() - \
            self.h_phi(obs).squeeze(-1)

        f_thet_phi = g_s + shaping_term

        return f_thet_phi

    def discr_value(self, log_p, *data):
        """
            Calculates the disctiminator output
                D = exp(f(s, a, s')) / [exp(f(s, a, s')) + pi(a|s)]
        """

        adv = self(*data)

        exp_adv = torch.exp(adv)
        value = exp_adv / (exp_adv + torch.exp(log_p) + EPS)
        # value2 = adv / (adv + log_p + EPS)

        return self.sigmoid(value)


class Actor(nn.Module):
    def __init__(self, **args):
        super(Actor, self).__init__()

    def forward(self, obs, ac=None):
        """
            Gives policy for given observations
            and optionally actions log prob under that
            policy
        """
        pi = self.sample_policy(obs)
        log_p = None

        if isinstance(self, CategoricalPolicy):
            ac = ac.unsqueeze(-1)

        if ac is not None:
            log_p = self.log_p(pi, ac)
        return pi, log_p


class MLPGaussianPolicy(Actor):
    """
        Gaussian Policy for stochastic actions
    """
    def __init__(self,
                 obs_dim,
                 act_dim,
                 hidden_sizes,
                 activation=nn.Tanh,
                 size=2):
        super(MLPGaussianPolicy, self).__init__()

        self.logits = mlp(obs_dim,
                          hidden_sizes + [act_dim],
                          activation,
                          size=size)
        log_std = -.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))

    def sample_policy(self, obs):
        """
            Creates a normal distribution representing
            the current policy which
            if sampled, returns an action on the policy given the observation
        """
        mu = self.logits(obs)
        pi = torch.distributions.Normal(loc=mu, scale=torch.exp(self.log_std))

        return pi

    @classmethod
    def log_p(cls, pi, a):
        """
            The log probability of taken action
            a in policy pi
        """
        return pi.log_prob(a).sum(
            axis=-1)  # Sum needed for Torch normal distr.


class CategoricalPolicy(Actor):
    """
        Categorical Policy for discrete action spaces
    """
    def __init__(self,
                 obs_dim,
                 act_dim,
                 hidden_sizes,
                 activation=nn.Tanh,
                 size=2):
        super(CategoricalPolicy, self).__init__()

        self.logits = mlp(obs_dim,
                          hidden_sizes + [act_dim],
                          activation,
                          size=size)

    def sample_policy(self, obs):
        """
            Get new policy
        """
        logits = self.logits(obs)
        pi = torch.distributions.Categorical(logits=logits)

        return pi

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
    def __init__(self,
                 obs_space,
                 act_space,
                 hidden_size=[32, 32],
                 activation=nn.Tanh,
                 size=2,
                 **args):
        super(MLPActor, self).__init__()

        obs_dim = obs_space.shape[0]

        discrete = True if isinstance(act_space, Discrete) else False
        act_dim = act_space.n if discrete else act_space.shape[0]

        if discrete:
            self.pi = CategoricalPolicy(obs_dim,
                                        act_dim,
                                        hidden_size,
                                        size=size,
                                        activation=activation)
        else:
            self.pi = MLPGaussianPolicy(obs_dim,
                                        act_dim,
                                        hidden_size,
                                        activation=activation,
                                        size=size)

        self.disc = Discriminator(obs_dim, **args)

    def step(self, obs):
        """
            Get distribution under current obs and action sample from pi
        """
        with torch.no_grad():
            pi_new = self.pi.sample_policy(obs)
            a = pi_new.sample()

            log_p = self.pi.log_p(pi_new, a)

        return a.numpy(), log_p.numpy()
