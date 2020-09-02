import numpy as np
import torch
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import gym
import time

from .core import discounted_cumsum


class ReplayBuffer:
    """
        Transitions buffer
        Stores transitions for a single episode
    """

    def __init__(self, size=4000, gamma=.98, lamda=.95):
        self.size = size
        self.gamma = gamma
        self.lamda = lamda

        self.rewards = []
        self.actions = []
        self.states = []
        self.next_states = []

        self.log_prob = []
        self.adv = []
        self.vals = []

        self.ptr, self.eps_end_ptr = 0, 0

    def store(self, act, states, values, rew):
        """
            Store transitions
        """
        idx = self.ptr % self.size

        self.rewards[idx] = rew
        self.actions[idx] = act
        self.states[idx] = states
        self.vals[idx] = values

        self.ptr += 1

    def get(self):
        """
            Returns episode transitions
        """
        assert self.ptr >= self.size

        self.ptr = 0
        self.eps_end_ptr = 0
        return torch.as_tensor(self.actions), torch.as_tensor(self.rewards), \
            torch.as_tensor(self.states), torch.as_tensor(
            self.adv), torch.as_tensor(self.log_prob)

    def end_eps(self, value=0):
        """
            Calculates the adv once the agent
            encounters an end state

            value: value of that state -> zero if the agent
            died or the value function if the episode was terminated
        """
        idx = slice(self.eps_end_ptr, self.ptr)

        rew = np.append(self.rewards, value)
        vals = np.append(self.vals, value)

        # GAE
        deltas = rew[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv[idx] = discounted_cumsum(deltas, self.gamma * self.lamda)

        # Reward to go
        self.rewards[idx] = discounted_cumsum(rew, self.gamma)[:-1]

        self.eps_end_ptr = self.ptr


def ppo(env, actor_class=core.MLPActor, gamma=.98, lamda=.95, epoch_size=4000, steps_per_epoch=1000, clip_ratio=.2, **args):
    """
    actor_args: hidden_size(list), size(int)-network size, pi_lr,
     v_lr
    """

    obs_space = env.observation_space
    act_space = env.action_space

    actor = actor_class(**args['ac_args'])
    pi, v_func = actor.pi, actor.v

    memory = ReplayBuffer(epoch_size, lamda=lamda, gamma=gamma)

    pi_optimizer = optim.adam(pi.parameters(), args.get('pi_lr') or 1e-4)
    v_optimizer = optim.adam(v_func.parameters(), args.get('v_lr') or 1e-3)

    pi_losses, v_losses = [], []  # Hold epoch losses for logging
    pi_kl = []  # kls for logging
    v_logs = []

    logger = SummaryWriter(log_dir='.data/')

    def compute_pi_loss(log_p_old, adv_b, act_b, obs_b):
        """
            Pi loss
        """

        global pi

        pi_new, log_p_ = pi(obs_b, act_b)

        pi_ratio = torch.exp(log_p_ - log_p_old)
        min_adv = torch.where(adv_b >= 0, (1 + clip_ratio)
                              * adv_b, (1-clip_ratio) * adv_b)

        pi_loss = -torch.mean(torch.min(pi_ratio * adv_b, min_adv))

        pi = pi_new

        return pi_loss, (pi_new - pi).mean().item()

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
        v_loss = compute_v_loss(data)

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
        eps_len_logs = [], eps_ret_log = []
        for step in range(steps_per_epoch):
            a, v, log_p = pi.step(obs)

            # log v
            v_logs.append(v)
            obs_n, rew, done, _ = env.step(a)

            eps_len += 1
            eps_ret += rew

            memory.store(a, obs, values=v, rew=eps_ret)

            if done or step == steps_per_epoch - 1:
                # terminated by max episode steps
                if not done:
                    last_v = v_func(obs)
                else:  # Agent terminated episode
                    last_v = 0

                    # only log these for terminals
                    eps_len_logs += [eps_len]
                    eps_ret_log += [eps_ret]

                memory.end_eps(value=last_v)
                obs_n = env.reset()
                eps_len, eps_ret = 0, 0

            obs = obs_n
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

        print('\n', '-' * 15)
        print('AverageEpsReturn: ', AverageEpsReturn)
        print('MinEpsReturn: ', MinEpsReturn)
        print('MaxEpsReturn: ', MaxEpsReturn)
        print('KL: ', Kl)
        print('AverageEpsLen: ', AverageEpisodeLen)
        print('Pi loss: ', Pi_Loss)
        print('V loss: ', V_loss)
        print('Run time: ', RunTime)


def main():
    """
        Ppo runner
    """

    env = gym.make('HalfCheetah-v3')

    ac_args = {
        'hidden_size': [32, 32],
        'size': 2, 'pi_lr': 1e-4, 'v_lr': 1e-3
    }

    args = {'ac_args': ac_args}

    ppo(env, **args)
