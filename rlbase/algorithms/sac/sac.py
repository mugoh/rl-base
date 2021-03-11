"""SAC"""

import copy
import time
import os

import torch
import torch.nn as nn
import numpy as np


from torch.utils.tensorboard import SummaryWriter


import core


class ReplayBuffer:
    """
        Memory for transitions
    """

    def __init__(self, size=int(1e5), *, act_dim: int, obs_dim: int, device: object):
        self.obs = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs_n = np.zeros([size, obs_dim], dtype=np.float32)
        self.act = np.zeros([size,  act_dim], dtype=np.float32)
        self.rewards = np.zeros([size], dtype=np.float32)
        self.dones = np.zeros([size], dtype=np.float32)

        self.idx = 0
        self.max_size = size
        self.size = 0
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

        self.idx = (self.idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """
            Samples a batch of transitions

            Returns: dict of transitions
        """

        idx = np.random.randint(0, self.size, size=batch_size)

        samples = {'obs': self.obs[idx],
                   'obs_n': self.obs_n[idx],
                   'act': self.act[idx],
                   'rew': self.rewards[idx],
                   'dones': self.dones[idx]
                   }
        samples = {
            k: torch.from_numpy(v).to(
                self.device) for k, v in samples.items()
        }

        return samples


def sac(env, ac_kwargs={}, actor_critic=None, memory_size: int = int(1e6),
        exploration_steps: int = 10000, max_eps_len: int = 1000,
        start_update: int = 3000,
        steps_per_epoch: int = 5000,
        epochs: int = 100,
        batch_size: int = 64, alpha: float = .05,
        seed: int = 1, **args):
    """
        Soft Actor Critic

        Args
        ----
        ac_kwargs (dict):  Actor critic parameters

        memory_size (int): Replay buffer transition count limit

        steps_per_epoch (int): Number of steps to interact with the
            env per episode

        epochs (int): Updates to perform on the policy

        exploration_steps: Number of steps to perform random action
            selection, before starting to query the policy for actions

        max_eps_len (int): Maximum possible length for each episode

        steps_per_epoch (int): Number of environment steps to take before
            updating the policy and q parameters

        epochs (int): Number of updates to perform

        start_update (int): Number of transitions to store in the replay
            buffer before it can be sampled for updates

        batch_size (int):Number of transitions to be used in updates

        alpha (float): Entropy regulation coefficient.
            It is used for exploration-exploitation trade-off.
            A high value of alpha results in higher exploration

    """

    np.random.seed(seed)
    torch.manual_seed(seed)

    device_nm = args.get('device') or (
        'cuda' if torch.cuda.is_available else 'cpu')
    device = torch.device(device_nm)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    r_buffer = ReplayBuffer(
        size=buffer_size, act_dim=act_dim, obs_dim=obs_dim, device=device)
    # Init pi and Qs parameters
    actor_critic = actor_critic(obs_dim, act_dim, **ac_kwargs).to(device)
    ac_target = copy.deepcopy(actor_critic).to(device)

    q_loss_f = nn.MSELoss()

    print(f'Param counts: {core.count(actor_critic)}\n')

    run_t = time.strftime('%Y-%m-%d-%H-%M-%S')
    path = os.path.join(
        'data', env.unwrapped.spec.id + args.get('env_name', '') + '_' + run_t)

    logger = SummaryWriter(log_dir=path)

    def zero_optim(optimizer, set_none: Optional[bool] = False):
        """
            Set Grads to None
        """
        if not set_none:
            optimizer.zero_grad()
            return
        for group in optimizer.param_groups:
            for param in group['params']:
                param.grad = None

    def compute_pi_loss(data):
        """Returns policy loss"""
        ...

    def compute_q_loss(data, gamma: float):
        """Returns Q loss"""
        rew = data['rew']
        dones = data['dones']
        obs_n = data['obs_n']
        obs = data['obs']
        act = data['act']

        # target = r + gamma(1 - d)[Q_t(s', a') + alpha * H(pi(a_tilde, xi))]

        # Min Q target
        act_tilde = actor_critic.pi(obs_n)
        q1_pred = ac_target.q_1(obs_n, act_tilde)
        q2_pred = ac_target.q_2(obs_n, act_tilde)

        q_pred = min(q1_target, q2_target)

        target = rew + gamma * (1 - dones) * q_pred - \
            (alpha * actor_critic.pi.log_p(act_tilde))

        q1_loss = q_loss_f(ac_target.q_1(obs, act), target)
        q2_loss = q_loss_f(ac_target.q_2(obs, act), target)

        return q1_loss, q2_loss

    def update(data):
        """Updates the policy and Q functions"""
        q1_loss, q2_loss = compute_q_loss(data)

    eps_len, eps_ret = 0

    update_frequency = steps_per_epoch // epochs

    start_time = time.time()
    obs = env.reset()

    for epoch in range(epochs):
        eps_len_logs, eps_ret_logs = [], []

        for t in range(steps_per_epoch):
            steps_run = (epoch * steps_per_epoch) + t + 1

            if steps_run <= exploration_steps:
                act = env.action_space.sample()

            else:
                act =  # get action from policy

            obs_n, rew, done, _ = env.step(act)
            r_buffer.store(obs, act, obs_n, rew, done)

            obs = obs_n

            eps_len += 1
            eps_ret += rew

            terminal = done or eps_len == max_eps_len

            if terminal:
                # append logs
                eps_len_logs.append(eps_len)
                eps_ret_logs.append(rew)

                # reset state
                obs, eps_ret, eps_len = env.reset(), 0, 0

            # update
            if steps_run >= start_update and not steps_run % update_frequency:
                data = r_buffer.sample(batch_size)
                update(data)
