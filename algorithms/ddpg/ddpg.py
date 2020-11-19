import copy
import time
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import optim

import core
from typing import Optional


class ReplayBuffer:
    """
        Memory for transitions
    """

    def __init__(self, size=int(1e4), *, act_dim: int, obs_dim: int, device: object):
        self.obs = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs_n = np.zeros([size, obs_dim], dtype=np.float32)
        self.act = np.zeros([size,  act_dim], dtype=np.float32)
        self.rewards = np.zeros([size], dtype=np.float32)
        self.dones = np.zeros([size], dtype=np.float32)

        self.idx = 0
        self.size = size
        self.device = device

    def store(self, obs, act, obs_n, rew, done):
        """
            Store transition in buffer
        """
        idx = self.idx
        self.obs[idx] = obs
        self.obs_n[idx] = obs_n
        self.act[idx] = act
        self.dones[idx] = done
        self.rewards[idx] = rew

        self.idx = (self.idx + 1) % self.size

    def sample(self, batch_size):
        """
            Samples a batch of transitions

            Returns: dict of transitions
        """

        idx = np.random.randint(self.size, size=batch_size)

        samples = {'obs': self.obs[idx],
                   'obs_n': self.obs_n[idx],
                   'act': self.act[idx],
                   'rew': self.rewards[idx],
                   'dones': self.dones[idx]
                   }
        samples = {
            k: torch.from_numpy(v).to(
                self.device) for k, v in samples
        }

        return samples


def ddpg(env, ac_kwargs={}, memory_size=int(1e5), actor_critic=core.MLPActorCritic,
         steps_per_epoch: int = 5000, epochs: int = 100, max_eps_len: int = 1000,
         pi_lr: float = 1e-4, q_lr: float = 1e-3,
         act_noise_std: float = .1, exploration_steps: int = 10000,
         update_frequency: int = 50, start_update: int = 10000,
         batch_size: int = 128, gamma: float = .99,
         ** args):
    """
        Args
        ---
        ac_kwargs (dict): Actor Critic module parameters

        memory_size (int): Replay buffer limit for transition
            storage

        steps_per_epoch (int): Number of steps interact with the
            environment per episode

        epochs (int): Number of updates to perform on the agent

        max_eps_len (int): Maximum length to have for each episode
            before it is terminated

        pi_lr(float): Policy learning rate

        q_lr(float): Q-function learning rate

        act_noise_std (float): Stddev for noise added to actions at
            training time

        exploration_steps (int): Number of steps to select actions
            at random uniform before turning to the policy.
            Policy is deterministic

        update_frequency (int): Interval of steps at which to update
            the actor

        start_update (int): The number of steps to interact with
            the environment before starting updates.
            This is meant to collect enough transitions in the
            replay buffer.

        batch_size (int): SGD mini batch size

        gamma (float): Rewards decay factor

        polyyak (float): Factor for interpolation during updating
            of target network
    """

    device = torch.device('cpu' if not torch.cuda.is_available else 'gpu')

    act_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    act_limit = env.action_space.high[0]

    rep_buffer = ReplayBuffer(size=memory_size,
                              act_dim=act_dim,
                              obs_dim=obs_dim, device=device)
    actor_critic = actor_critic(obs_dim,
                                act_dim, act_limit,
                                **ac_kwargs)
    q, pi = actor_critic.q, actor_critic.pi

    ac_target = copy.deepcopy(actor_critic)
    q_target, pi_target = ac_target.q, ac_target.pi

    q_optim = optim.Adam(q.parameters(), lr=q_lr)
    pi_optim = optim.Adam(pi.parameters(), lr=pi_lr)

    q_loss_f = torch.nn.MSELoss()

    for param in q_target.parameters():
        param.grad = None

    for param in pi_target.parameters():
        param.grad = None

    run_t = time.strftime('%Y-%m-%d-%H-%M-%S')
    path = os.path.join(
        'data', env.unwrapped.spec.id + args.get('env_name', '') + '_' + run_t)

    logger = SummaryWriter(log_dir=path)

    def encode_action(action):
        """
            Add Gaussian noise to action
        """

        epsilon = np.random.rand(act_dim) * act_noise_std

        return np.clip(action + epsilon, -act_limit, act_limit)

    def compute_pi_loss(data):
        """
            Policy Loss function
        """
        states = data['obs']
        return -(q(states, pi(states))).mean()

    def compute_q_loss(data, gamma: float):
        """
            Q function loss
        """
        rew = data['rew']
        dones = data['dones']
        n_states = data['obs_n']
        states = data['obs']
        target = rew + gamma * (1 - dones) * \
            q_target(n_states, pi_target(n_states))
        loss = q_loss_f(q(states), target)

        return loss

    def zero_optim(optimizer, set_none: Optional[bool] = True):
        """
            Set Grads to None
        """
        if not set_none:
            optimizer.zero_grad()
            return
        for group in optimizer.param_groups:
            for param in group['params']:
                param.grad = None

    def update(n_epoch, main_args=args):
        """
            Policy and Q function update
        """

        data = rep_buffer.sample(main_args['batch_size'])

        # update Q
        zero_optim(q_optim)
        q_loss = compute_q_loss(data, main_args['gamma'])
        q_loss.backward()
        q_optim.step()

        # update pi

        zero_optim(pi_optim)
        pi_loss = compute_pi_loss(data)
        pi_loss.backward()
        pi_optim.step()

        logger.add_scalar('Loss/q', q_loss, n_epoch)
        logger.add_scalar('Loss/pi', pi_loss, n_epoch)

        polyyak = main_args['polyyak']
        phi_params = actor_critic.parameters()
        phi_target_params = ac_target.parameters()

        # update target
        for param, target_param in zip(
            phi_params, phi_target_params
        ):
            # p(target) + (1 - p)param
            target_param.mul_(polyyak)
            target_param.add(param.mul(1-polyyak))

    start_time = time.time()
    obs = env.reset()
    eps_len, eps_ret = 0, 0

    for epoch in range(epochs):

        eps_len_logs, eps_ret_logs = [], []
        for t in range(steps_per_epoch):

            # Total steps ran
            steps_run = epoch * steps_per_epoch
            if steps_run + t <= exploration_steps:
                act = env.action_space.sample()
            else:
                act = encode_action(actor_critic.act(obs))

            obs_n, rew, done, _ = env.step(act)

            rep_buffer.store(obs, act, obs_n, rew, done)
            obs = obs_n

            eps_len += 1
            eps_ret += rew

            terminal = done or eps_len == max_eps_len

            if terminal:
                eps_len_logs.append(eps_len)
                eps_ret_logs.append(eps_ret)

                obs, eps_ret, eps_len = env.reset(), 0, 0

        # perform update
        if steps_run > start_update and not steps_run % update_frequency:
            # update
            update(epoch)

        l_t = epoch  # log_time, start at 0

        logs = dict(RunTime=time.time() - start_time,
                    AverageEpisodeLen=np.mean(eps_len_logs),

                    # MaxEpisodeLen = np.max(eps_len_logs)
                    # MinEpsiodeLen = np.min(eps_len_logs)
                    AverageEpsReturn=np.mean(eps_ret_logs),
                    MaxEpsReturn=np.max(eps_ret_logs),
                    MinEpsReturn=np.min(eps_ret_logs)
                    )

        logger.add_scalar('AvEpsLen', logs['AverageEpisodeLen'], l_t)
        logger.add_scalar('EpsReturn/Max', logs['MaxEpsReturn'], l_t)
        logger.add_scalar('EpsReturn/Min', logs['MinEpsReturn'], l_t)
        logger.add_scalar('EpsReturn/Average', logs['AverageEpsReturn'], l_t)

        if t == 0:
            first_run_ret = logs['AverageEpsReturn']
        logs['FirstEpsReturn'] = first_run_ret

        for k, v in logs.items():
            print(k, v)
