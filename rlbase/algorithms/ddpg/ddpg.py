import copy
import time
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch import nn


import core
from typing import Optional

import gym

from rlbase.d2rl.d2rl import MLPActorCritic


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


def ddpg(env, ac_kwargs={}, actor_critic=core.MLPActorCritic,  memory_size=int(1e6),
         steps_per_epoch: int = 5000, epochs: int = 100, max_eps_len: int = 1000,
         pi_lr: float = 1e-3, q_lr: float = 1e-3, seed=0,
         act_noise_std: float = .1, exploration_steps: int = 10000,
         update_frequency: int = 50, start_update: int = 1000,
         batch_size: int = 128, gamma: float = .99,
         eval_episodes: int = 5, polyyak: float = .995,
         ** args):
    """
        Deep Deterministic Policy Gradients (DDPG)

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

        eval_episodes (int): Number of episodes to evaluate the agent
    """

    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(args.get('device'))if args.get('device') else\
        torch.device('cpu' if not torch.cuda.is_available else 'cuda')

    act_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    act_limit = env.action_space.high[0]

    rep_buffer = ReplayBuffer(size=memory_size,
                              act_dim=act_dim,
                              obs_dim=obs_dim, device=device)
    if args.get('load_model'):
        ckpt = load()
        actor_critic = ckpt['ac'].to(device)
        ac_target = ckpt['ac_target'].to(device)
        pi_optim = ckpt['pi_optim']
        q_optim = ckpt['q_optim']
        epoch_checkpoint = ckpt['epoch']
        q, pi = actor_critic.q, actor_critic.pi
    else:

        actor_critic = actor_critic(obs_dim,
                                    act_dim, act_limit,
                                    **ac_kwargs).to(device)
        q, pi = actor_critic.q, actor_critic.pi
        ac_target = copy.deepcopy(actor_critic).to(device)

        q_optim = optim.Adam(q.parameters(), lr=q_lr)
        pi_optim = optim.Adam(pi.parameters(), lr=pi_lr)
        epoch_checkpoint = 0

    q_target, pi_target = ac_target.q, ac_target.pi

    q_loss_f = torch.nn.MSELoss()

    for param in q_target.parameters():
        param.requires_grad = False

    for param in pi_target.parameters():
        param.requires_grad = False

    run_t = time.strftime('%Y-%m-%d-%H-%M-%S')
    path = os.path.join(
        'data', env.unwrapped.spec.id + args.get('env_name', '') + '_' + run_t)

    logger = SummaryWriter(log_dir=path)

    q_losses, pi_losses = [], []

    def encode_action(action):
        """
            Add Gaussian noise to action
        """

        epsilon = np.random.rand(act_dim) * act_noise_std

        return np.clip(action + epsilon, -act_limit, act_limit)

    def save(epoch: int, path: str = 'model.pt'):
        torch.save({
            'epoch': epoch,
            'ac': actor_critic.state_dict(),
            'ac_target': ac_target.state_dict(),
            'pi_optim': pi_optim.state_dict(),
            'q_optim': q_optim.state_dict()
        }, path)

    def load(path: str = 'model.pt'):
        """
            Load saved model

            Returns: (dict) Checkpoint key, value pairs
                epoch, ac, ac_target, pi_optim, q_optim
        """
        ckpt = torch.load(path)

        return ckpt

    def eval_agent(epoch, kwargs=args):
        """
            Evaluate agent
        """
        # env = kwargs['test_env']
        print(f'\n\nEvaluating agent\nEpisodes [{eval_episodes}]')
        all_eps_len, all_eps_ret = [], []
        for eps in range(eval_episodes):

            eps_len, eps_ret, obs = 0, 0, env.reset()
            for _ in range(steps_per_epoch):
                act = actor_critic.act(
                    torch.from_numpy(obs).float().to(device))

                obs_n, rew, done, _ = env.step(act)

                obs = obs_n
                eps_len += 1
                eps_ret += rew

                if done or eps_len == max_eps_len:
                    all_eps_len.append(eps_len)
                    all_eps_ret.append(eps_ret)

                    break

            logger.add_scalar('Evaluation/Return', eps_ret, epoch)
            logger.add_scalar('Evaluation/EpsLen', eps_len, epoch)

        return np.mean(all_eps_len), np.mean(all_eps_ret)

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
        actions = data['act']

        q_pred = q(states, actions)
        with torch.no_grad():
            target = rew + gamma * (1 - dones) * \
                q_target(n_states, pi_target(n_states))
        loss = q_loss_f(q_pred, target)

        return loss

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

    def update(n_epoch, data):
        """
            Policy and Q function update
        """

        # update Q
        zero_optim(q_optim)
        q_loss = compute_q_loss(data, gamma)
        q_loss.backward()
        q_optim.step()

        # update pi
        for p in q.parameters():
            p.requires_grad = False

        zero_optim(pi_optim)
        pi_loss = compute_pi_loss(data)
        pi_loss.backward()
        pi_optim.step()

        nonlocal q_losses, pi_losses

        q_losses += [q_loss.item()]
        pi_losses += [pi_loss.item()]

        logger.add_scalar('Loss/q', q_loss.item(), n_epoch)
        logger.add_scalar('Loss/pi', pi_loss.item(), n_epoch)

        for p in q.parameters():
            p.requires_grad = True
        # update target
        with torch.no_grad():
            phi_params = actor_critic.parameters()
            phi_target_params = ac_target.parameters()
            for param, target_param in zip(
                phi_params, phi_target_params
            ):
                # p(target) + (1 - p)param
                target_param.data.mul_(polyyak)
                target_param.data.add_(param.data * (1-polyyak))

    start_time = time.time()
    obs = env.reset()
    eps_len, eps_ret = 0, 0

    for epoch in range(epoch_checkpoint, epochs):

        eps_len_logs, eps_ret_logs = [], []
        for t in range(steps_per_epoch):

            # Total steps ran
            steps_run = (epoch * steps_per_epoch) + t + 1
            if steps_run <= exploration_steps:
                act = env.action_space.sample()
            else:
                obs = torch.from_numpy(obs).float().to(device)
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
            if steps_run >= start_update and not steps_run % update_frequency:
                data = rep_buffer.sample(batch_size)
                # Keep ratio of env interactions to n_updates = 1
                for _ in range(update_frequency):
                    update(epoch, data)

        l_t = epoch  # log_time, start at 0

        logs = dict(Epoch=epoch,
                    AverageEpisodeLen=np.mean(eps_len_logs),

                    # MaxEpisodeLen = np.max(eps_len_logs)
                    # MinEpsiodeLen = np.min(eps_len_logs)
                    AverageEpsReturn=np.mean(eps_ret_logs),
                    MaxEpsReturn=np.max(eps_ret_logs),
                    MinEpsReturn=np.min(eps_ret_logs),
                    RunTime=time.time() - start_time
                    )

        if args.get('evaluate_agent'):
            eval_eps_len, eval_eps_ret = eval_agent(epoch)
            logs['EvalAvEpsLength'] = eval_eps_len
            logs['EvalAvReturn'] = eval_eps_ret

        logger.add_scalar('AvEpsLen', logs['AverageEpisodeLen'], l_t)
        logger.add_scalar('EpsReturn/Max', logs['MaxEpsReturn'], l_t)
        logger.add_scalar('EpsReturn/Min', logs['MinEpsReturn'], l_t)
        logger.add_scalar('EpsReturn/Average', logs['AverageEpsReturn'], l_t)

        logger.add_scalar('Loss/Av-q', np.mean(q_losses), l_t)
        logger.add_scalar('Loss/Av-pi', np.mean(pi_losses), l_t)

        # Reset loss logs for next udpate
        q_losses, pi_losses = [], []

        if t == 0:
            first_run_ret = logs['AverageEpsReturn']
            logs['FirstEpsReturn'] = first_run_ret

        print('\n\n')
        print('-' * 15)
        for k, v in logs.items():
            print(k, v)

        # Save model

        if not epoch % args.get('save_frequency', 50) or epoch == epochs - 1:
            save(epoch)


def main():
    """
        DDPG run
    """
    en_nm = 'InvertedPendulum-v2'
    env = gym.make(en_nm)
    test_env = gym.make(en_nm)

    ac_kwargs = {
        'hidden_sizes': [128, 128, 128, 128, 128]
    }
    agent_args = {
        'env_name': 'HCv2',
        'actor_critic': MLPActorCritic
    }
    train_args = {
        'eval_episodes': 5,
        'seed': 0,
        'save_frequency': 120,
        'load_model': False,
        'device': 'cpu',
        'max_eps_len': 150,
        'test_env': test_env,
        'evaluate_agent': False,
        'q_lr': 1e-4,
        'pi_lr': 1e-4,
        'exploration_steps': 10000,
        'steps_per_epoch': 1000,
        'batch_size': 100
    }

    args = {'ac_kwargs': ac_kwargs, **agent_args, **train_args}

    ddpg(env,  **args)


if __name__ == '__main__':
    main()
