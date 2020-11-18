import copy
import time
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import optim

import core


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
         act_noise_std: float = .1, exploration_steps: int = 10000 ** args):
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

    start_time = time.time()
    obs = env.reset()
    eps_len, eps_ret = 0, 0

    for epoch in range(epochs):

        eps_len_logs, eps_ret_logs = [], []
        for t in range(steps_per_epoch):

            # Total steps ran
            if (epoch * steps_per_epoch) + t <= exploration_steps:
                act = env.action_space.sample()
            else:
                act = encode_action(actor_critic.act(obs))
        l_t = t  # log_time, start at 0

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
