import numpy as np
import torch
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import gym
import time

import core
from core import dcum2 as discounted_cumsum


class ReplayBuffer:
    """
        Transitions buffer
        Stores transitions for a single episode
    """

    def __init__(self, act_dim, obs_dim, size=4000, gamma=.98, lamda=.95):
        self.size = size
        self.gamma = gamma
        self.lamda = lamda

        self.rewards = np.zeros([size], dtype=np.float32)
        self.actions = np.zeros([size, act_dim], dtype=np.float32)
        self.states = np.zeros([size, obs_dim], dtype=np.float32)

        self.log_prob = np.zeros([size], dtype=np.float32)
        self.adv = np.zeros([size], dtype=np.float32)
        self.vals = np.zeros([size], dtype=np.float32)

        self.ptr, self.eps_end_ptr = 0, 0

    def store(self, act, states, values, rew, log_p):
        """
            Store transitions
        """
        idx = self.ptr % self.size

        self.rewards[idx] = rew
        self.actions[idx] = act
        self.states[idx] = states
        self.vals[idx] = values
        self.log_prob[idx] = log_p

        self.ptr += 1

    def get(self):
        """
            Returns episode transitions
        """
        assert self.ptr >= self.size

        self.ptr = 0
        self.eps_end_ptr = 0
        return torch.as_tensor(self.actions, dtype=torch.float32), torch.as_tensor(self.rewards, dtype=torch.float32), \
            torch.as_tensor(self.states, dtype=torch.float32), torch.as_tensor(
            self.adv, dtype=torch.float32), torch.as_tensor(self.log_prob, dtype=torch.float32)

    def end_eps(self, value=0):
        """
            Calculates the adv once the agent
            encounters an end state

            value: value of that state -> zero if the agent
            died or the value function if the episode was terminated
        """
        idx = slice(self.eps_end_ptr, self.ptr)

        rew = np.append(self.rewards[idx], value)
        vals = np.append(self.vals[idx], value)

        # GAE
        deltas = rew[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv[idx] = discounted_cumsum(deltas, self.gamma * self.lamda)

        # Reward to go
        self.rewards[idx] = discounted_cumsum(rew, self.gamma)[:-1]

        self.eps_end_ptr = self.ptr


def ppo(env, actor_class=core.MLPActor, gamma=.98, lamda=.95, epoch_size=40, steps_per_epoch=5000, max_eps_len=1000, clip_ratio=.2, **args):
    """
    actor_args: hidden_size(list), size(int)-network size, pi_lr,
     v_lr
    """

    obs_space = env.observation_space
    act_space = env.action_space

    act_dim = act_space.shape[0] if not isinstance(act_space,
                                                   gym.spaces.Discrete) else act_space.n
    obs_dim = obs_space.shape[0]

    actor = actor_class(obs_space=obs_space,
                        act_space=act_space, **args['ac_args'])
    pi, v_func = actor.pi, actor.v

    memory = ReplayBuffer(act_dim, obs_dim, steps_per_epoch,
                          lamda=lamda, gamma=gamma)

    pi_optimizer = optim.Adam(pi.parameters(), args.get('pi_lr') or 1e-4)
    v_optimizer = optim.Adam(v_func.parameters(), args.get('v_lr') or 1e-3)

    pi_losses, v_losses = [], []  # Hold epoch losses for logging
    pi_kl = []  # kls for logging
    v_logs = []

    logger = SummaryWriter(log_dir='data/')

    def compute_pi_loss(log_p_old, adv_b, act_b, obs_b):
        """
            Pi loss
        """

        nonlocal pi

        # returns new_pi_normal_distribution, logp_act
        _, log_p_ = pi(obs_b, act_b)
        log_p_= log_p_.type(torch.float32)  # From torch.float64

        pi_ratio = torch.exp(log_p_ - log_p_old)
        min_adv = torch.where(adv_b >= 0, (1 + clip_ratio)
                              * adv_b, (1-clip_ratio) * adv_b)


        pi_loss = -torch.mean(torch.min(pi_ratio * adv_b, min_adv))

        # pi = pi_new

        return pi_loss, (log_p_old - log_p_).mean().item()

    def compute_v_loss(data):
        """
            Value function loss
        """
        obs_b, rew_b = data['obs_b'], data['rew_b']

        v_pred = v_func(obs_b)
        v_loss = torch.mean((v_pred - rew_b) ** 2)

        return v_loss

    def update(epoch):
        """
            Update the policy and value function from loss
        """
        data = memory.get()
        act_b, rew_b, obs_b, adv_b, log_p_old = data

        pi_optimizer.zero_grad()
        pi_loss, kl = compute_pi_loss(
            log_p_old=log_p_old, obs_b=obs_b, adv_b=adv_b, act_b=act_b)

        pi_loss.backward()
        pi_optimizer.step()

        v_optimizer.zero_grad()
        v_loss = compute_v_loss({'obs_b': obs_b, 'rew_b': rew_b})

        v_loss.backward()
        v_optimizer.step()

        pi_loss = pi_loss.item()
        v_loss = v_loss.item()
        v_losses.append(v_loss)

        pi_losses.append(pi_loss)
        pi_kl.append(kl)

        logger.add_scalar('loss/pi', pi_loss, epoch)
        logger.add_scalar('loss/v', v_loss, epoch)
        logger.add_scalar('Kl', kl, epoch)

    start_time = time.time()
    obs = env.reset()
    eps_len, eps_ret = 0, 0

    for t in range(epoch_size):
        eps_len_logs, eps_ret_log = [], []
        for step in range(steps_per_epoch):
            a, v, log_p = actor.step(torch.as_tensor(obs, dtype=torch.float32))

            # log v
            v_logs.append(v)
            obs_n, rew, done, _ = env.step(a)

            eps_len += 1
            eps_ret += rew

            memory.store(a, obs, values=v, log_p=log_p, rew=eps_ret)

            obs = obs_n

            terminal = done or eps_len == max_eps_len

            if terminal or step == steps_per_epoch - 1:
                # terminated by max episode steps
                if not done:
                    last_v = v_func(torch.as_tensor(obs, dtype=torch.float32))
                else:  # Agent terminated episode
                    last_v = 0

                if terminal:
                    # only log these for terminals
                    eps_len_logs += [eps_len]
                    eps_ret_log += [eps_ret]


                memory.end_eps(value=last_v)
                obs = env.reset()
                eps_len, eps_ret = 0, 0


        update(t)

        # Print info for each epoch: loss_pi, loss_v, kl
        # time, v at traj collection, eps_len, epoch_no,
        # eps_ret: min, max, av
        RunTime = time.time() - start_time
        AverageEpisodeLen = np.mean(eps_len_logs)

        logger.add_scalar('AvEpsLen', AverageEpisodeLen, t)
        # MaxEpisodeLen = np.max(eps_len_logs)
        # MinEpsiodeLen = np.min(eps_len_logs)
        AverageEpsReturn = np.mean(eps_ret_log)
        MaxEpsReturn = np.max(eps_ret_log)
        MinEpsReturn = np.min(eps_ret_log)

        logger.add_scalar('EpsReturn/Max', MaxEpsReturn, t)
        logger.add_scalar('EpsReturn/Min', MinEpsReturn, t)
        logger.add_scalar('EpsReturn/Average', AverageEpsReturn, t)


        Pi_Loss = pi_losses[t]
        V_loss = v_losses[t]
        Kl = pi_kl[t]

        print('\n', t)
        print('', '-' * 35)
        print('AverageEpsReturn: ', AverageEpsReturn)
        print('MinEpsReturn: ', MinEpsReturn)
        print('MaxEpsReturn: ', MaxEpsReturn)
        print('KL: ', Kl)
        print('AverageEpsLen: ', AverageEpisodeLen)
        print('Pi_loss: ', Pi_Loss)
        print('V_loss: ', V_loss)
        print('Run time: ', RunTime)
        print('\n\n\n')


def main():
    """
        Ppo runner
    """

    env = gym.make('HalfCheetah-v3')

    ac_args = {
        'hidden_size': [32, 32],
        'size': 2
    }

    args = {'ac_args': ac_args, 'pi_lr': 1e-4, 'v_lr': 1e-3}

    ppo(env, **args)


if __name__ == '__main__':
    main()
