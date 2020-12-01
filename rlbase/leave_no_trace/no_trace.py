"""
    Learning to Reset Agent
"""

import copy
import time
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch import nn

import rlbase.algorithms.ddpg.core as core
from rlbase.algorithms.ddpg.ddpg import ReplayBuffer

from typing import Optional, Any, Mapping, List
from dataclasses import dataclass
import click

import gym


@dataclass
class Agent:
    env: object
    ac_kwargs: dict = {}
    actor_critic_class: object = core.MLPActorCritic
    memory_size: int = int(1e6)
    steps_per_epoch: int = 500
    epochs: int = 100
    max_eps_len: int = 150
    pi_lr: float = 1e-4
    q_lr: float = 1e-4
    seed: int = 0
    act_noise_std: float = .1
    exploration_steps: int = 1000
    update_frequency: int = 50
    start_update: int = 1000
    batch_size: int = 128
    gamma: float = .99
    eval_episodes: int = 5
    polyyak: float = .995
    s_reset: float = .7
    q_min: float = 10.
    n_resets: int = 1
    epochs_per_policy: int = 1

    args: Mapping[Any, Any]

    """
        DDPG agent

        Args
        ---
        ac_kwargs (dict): Actor Critic module parameters

        memory_size (int): Replay buffer limit for transition
            storage

        steps_per_epoch (int): Number of steps interact with the
            environment per episode

        epochs (int): Number of updates to perform on the agents
            These are updates for both the forward and the reset
            policy

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

        q_min (float): Minimum Q-value for a state to be considered
            safe. If less than this value, the control is trasnffered
            to the reset policy

        n_resets (int): Number of soft reset attempts before making
            a hard reset.

            Increasing `n_resets` decreases the no. of hard resets
            but may lead to the agent making unnessary reset attempts
            on a state an early reset can't be done

            In some environments, increasing it also lowers the
            cumulutave reward

        s_reset (float): The minimum reset policy reward for a state
            to be considered in the safe set s_reset

        epochs_per_policy (int): Number of episode updates to run
            for each policy before switching to the other.
            Defaults to 1 each

    """

    def __init__(self, env,  **args):

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        device = torch.device(args.get('device')) if args.get('device') else \
            torch.device('cpu' if not torch.cuda.is_available else 'cuda')
        self.args = args
        ac_kwargs = args['ac_kwargs']

        env = self.env
        act_dim = env.action_space.shape[0]
        obs_dim = env.observation_space.shape[0]
        act_limit = env.action_space.high[0]

        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.act_limit = act_limit

        self.rep_buffer = ReplayBuffer(size=self.memory_size,
                                       act_dim=act_dim,
                                       obs_dim=obs_dim, device=device)
        self.rep_buffer_reset = ReplayBuffer(size=self.memory_size,
                                             act_dim=act_dim,
                                             obs_dim=obs_dim, device=device)

        self.actor_critic = self.actor_critic_class(obs_dim,
                                                    act_dim, act_limit,
                                                    **ac_kwargs).to(device)
        self.actor_critic_reset = self.actor_critic_class(obs_dim,
                                                          act_dim, act_limit,
                                                          **ac_kwargs).to(device)
        self.ac_target = copy.deepcopy(self.actor_critic).to(device)
        self.ac_target_reset = copy.deepcopy(
            self.actor_critic_reset).to(device)

        pi_lr, q_lr = self.pi_lr, self.q_lr

        q_optim = optim.Adam(self.actor_critic.q.parameters(), lr=q_lr)
        pi_optim = optim.Adam(self.actor_critic.pi.parameters(), lr=pi_lr)

        q_reset_optim = optim.Adam(
            self.actor_critic_reset.q.parameters(), lr=q_lr)
        pi_reset_optim = optim.Adam(
            self.actor_critic_reset.pi.parameters(), lr=pi_lr)

        self.optimizers = {'q_optim': q_optim, 'pi_optim': pi_optim,
                           'pi_reset_optim': pi_reset_optim, 'q_reset_optim': q_reset_optim}

        self.q_loss_f = torch.nn.MSELoss()

        print(f'param counts: {core.count(self.actor_critic)}, ' +
              f'reset:{core.count(self.actor_critic_reset)}\n')

        for param in self.ac_target.parameters():
            param.requires_grad = False
        for param in self.ac_target_reset.parameters():
            param.requires_grad = False

        run_t = time.strftime('%Y-%m-%d-%H-%M-%S')
        path = os.path.join(
            'data', env.unwrapped.spec.id + args.get('env_name', '') + '_' + run_t)

        self.logger = SummaryWriter(log_dir=path)

        self.q_losses, self.pi_losses = [], []

    def encode_action(self, action):
        """
            Add Gaussian noise to action
        """

        epsilon = np.random.rand(self.act_dim) * self.act_noise_std
        act_limit = self.act_limit

        return np.clip(action + epsilon, -act_limit, act_limit)

    def eval_agent(self, epoch):
        """
            Evaluate agent
        """
        # env = kwargs['test_env']
        print(f'\n\nEvaluating agent\nEpisodes [{self.eval_episodes}]')
        all_eps_len, all_eps_ret = [], []
        for _ in range(self.eval_episodes):

            eps_len, eps_ret, obs = 0, 0, self.env.reset()
            for _ in range(self.steps_per_epoch):
                act = self.actor_critic.act(
                    torch.from_numpy(obs).float().to(device))

                obs_n, rew, done, _ = self.env.step(act)

                obs = obs_n
                eps_len += 1
                eps_ret += rew

                if done or eps_len == self.max_eps_len:
                    all_eps_len.append(eps_len)
                    all_eps_ret.append(eps_ret)

                    break

            self.logger.add_scalar('Evaluation/Return', eps_ret, epoch)
            self.logger.add_scalar('Evaluation/EpsLen', eps_len, epoch)

        return np.mean(all_eps_len), np.mean(all_eps_ret)

    def compute_pi_loss(self, data, ac: object):
        """
            Policy Loss function

            ac: Actor critic to compute loss for
        """
        states = data['obs']
        q = ac.q
        pi = ac.pi

        return -(q(states, pi(states))).mean()

    def save(self, epoch: int, model_data: dict, path: str = 'model.pt'):
        """
            Saves the model checkpoint
            Pass the model data in the format:

              {
                'epoch': epoch,
                'ac': actor_critic.state_dict(),
                'ac_target': ac_target.state_dict(),
                'pi_optim': pi_optim.state_dict(),
                'q_optim': q_optim.state_dict()
             }
        """
        torch.save(model_data, path)

    def load(self, path: str = 'model.pt'):
        """
            Load saved model

            Returns: (dict) Checkpoint key, value pairs
                epoch, ac, ac_target, pi_optim, q_optim
        """
        ckpt = torch.load(path)

        return ckpt

    def compute_q_loss(self, data, gamma: float, ac: List[object]):
        """
            Q function loss

            ac: [Ac, Ac_target] Actor critic to compute loss for

        """
        rew = data['rew']
        dones = data['dones']
        n_states = data['obs_n']
        states = data['obs']
        actions = data['act']

        q = ac[0].q
        pi_target = ac[1].pi
        q_target = ac[1].q

        q_pred = q(states, actions)
        with torch.no_grad():
            target = rew + gamma * (1 - dones) * \
                q_target(n_states, pi_target(n_states))
        loss = self.q_loss_f(q_pred, target)

        return loss

    def zero_optim(self, optimizer, set_none: Optional[bool] = False):
        """
            Set Grads to None
        """
        if not set_none:
            optimizer.zero_grad()
            return
        for group in optimizer.param_groups:
            for param in group['params']:
                param.grad = None

    def update(self, n_epoch, data, ac_items: List[object], pi_optim: object, q_optim: object, name: str):
        """ Policy and Q function update

            ac: [ac, ac_target] Current actor critic
        """

        ac, ac_target = ac_items
        # update Q
        self.zero_optim(q_optim)
        q_loss = self.compute_q_loss(data, self.gamma, ac_items)
        q_loss.backward()
        q_optim.step()

        # update pi
        for p in ac.q.parameters():
            p.requires_grad = False

        self.zero_optim(pi_optim)
        pi_loss = self.compute_pi_loss(data, ac)
        pi_loss.backward()
        pi_optim.step()

        self.q_losses += [q_loss.item()]
        self.pi_losses += [pi_loss.item()]

        self.logger.add_scalar(f'Loss/q-{name}', q_loss.item(), n_epoch)
        self.logger.add_scalar('Loss/pi-{name}', pi_loss.item(), n_epoch)

        for p in ac.q.parameters():
            p.requires_grad = True
        # update target
        with torch.no_grad():
            phi_params = ac.parameters()
            phi_target_params = ac_target.parameters()
            for param, target_param in zip(
                phi_params, phi_target_params
            ):
                # p(target) + (1 - p)param
                target_param.data.mul_(self.polyyak)
                target_param.data.add_(param.data * (1-self.polyyak))

    def train_agent(self):
        """ Trains the forward and reset policies

        """

        for epoch in range(self.epochs):
            # Train forward policy
            self.run_training_loop([self.actor_critic, self.ac_target],
                                   policy_name='forward',
                                   global_epoch=epoch)

            # Train reset policy

            self.run_training_loop([self.actor_critic_reset, self.ac_target_reset],
                                   'reset',
                                   epoch)

    def run_training_loop(self, ac_items: List[object], policy_name: str,
                          global_epoch: int):
        """ Trains the agent by running the specified
            number of steps and agent udpates

            Args:
                ac_items (list): Actor critic and Target for the current policy
                policy_name (str): The current policy (forward or reset)
        """
        start_time = time.time()
        obs = self.env.reset()
        eps_len, eps_ret = 0, 0
        hard_reset_count, soft_reset_count = 0, 0

        actor_critic, _ = ac_items

        q_optim = 'q_reset_optim' if 'reset' in policy_name else 'q_optim'
        pi_optim = 'pi_reset_optim' if 'reset' in q_optim else 'pi_optim'

        q_optim = self.optimizers['q_optim']
        pi_optim = self.optimizers['pi_optim']

        eps_len_logs, eps_ret_logs = [], []
        hard_reset_logs, soft_reset_logs = [], []

        to_reset = 'forward' in policy_name

        for epoch in range(self.epochs_per_policy):

            for t in range(self.steps_per_epoch):

                # Total steps ran
                steps_run = (global_epoch * self.steps_per_epoch) + t + 1
                random_policy = steps_run <= self.exploration_steps
                if random_policy:
                    act = self.env.action_space.sample()
                else:
                    obs = torch.from_numpy(obs).float().to(self.device)
                    act = self.encode_action(actor_critic.act(obs))

                    # If this is the forward policy
                    # and Q < Q min
                    # switch to reset policy

                if to_reset and not random_policy:

                    for attempt in range(self.n_resets):
                        rst_q_value = self.actor_critic_reset.q(obs, act)

                        if rst_q_value < self.q_min:
                            # soft reset
                            act = self.actor_critic_reset.act(obs)
                        else:
                            soft_reset_count += 1
                            break
                        if attempt == self.n_resets - 1:
                            hard_reset_count += 1
                            hard_reset = True

                obs_n, rew, done, _ = self.env.step(act)

                if 'reset' in policy_name:
                    # Train reset policy
                    self.rep_buffer_reset.store(obs, act, obs_n, rew, done)

                else:
                    self.rep_buffer.store(obs, act, obs_n, rew, done)

                obs = obs_n

                eps_len += 1
                eps_ret += rew

                terminal = done or eps_len == self.max_eps_len

                if terminal or hard_reset:
                    eps_len_logs.append(eps_len)
                    eps_ret_logs.append(eps_ret)

                    hard_reset_logs.append(hard_reset_count)
                    soft_reset_logs.append(soft_reset_count)

                    obs, eps_ret, eps_len = self.env.reset(), 0, 0
                    hard_reset_count, soft_reset_count = 0, 0

                # perform update
                if steps_run >= self.start_update and not steps_run % self.update_frequency:
                    memory_ = self.rep_buffer.sample if 'forward' in policy_name\
                        else self.rep_buffer_reset.sample
                    data = memory_(self.batch_size)

                    # Keep ratio of env interactions to n_updates = 1
                    for _ in range(self.update_frequency):
                        self.update(epoch, data, ac_items,
                                    q_optim=q_optim,
                                    pi_optim=pi_optim,
                                    name=policy_name)

            l_t = global_epoch  # log_time, start at 0

            logs = dict(Epoch=l_t,
                        AverageEpisodeLen=np.mean(eps_len_logs),

                        # MaxEpisodeLen = np.max(eps_len_logs)
                        # MinEpsiodeLen = np.min(eps_len_logs)
                        AverageEpsReturn=np.mean(eps_ret_logs),
                        HardResets=hard_reset_logs,
                        SoftResets=soft_reset_logs,
                        MaxEpsReturn=np.max(eps_ret_logs),
                        MinEpsReturn=np.min(eps_ret_logs),
                        RunTime=time.time() - start_time
                        )

            if self.args.get('evaluate_agent'):
                eval_eps_len, eval_eps_ret = self.eval_agent(epoch)
                logs[f'EvalAvEpsLength-{policy_name}'] = eval_eps_len
                logs[f'EvalAvReturn-{policy_name}'] = eval_eps_ret
            if 'forward' in policy_name:
                self.logger.add_scalar(
                    'HardResetCount', logs['HardResets'], l_t)
                self.logger.add_scalar(
                    'SoftResetCount', logs['SoftResets'], l_t)

            self.logger.add_scalar(
                f'AvEpsLen-{policy_name}', logs['AverageEpisodeLen'], l_t)
            self.logger.add_scalar(
                f'EpsReturn/Max-{policy_name}',
                logs['MaxEpsReturn'], l_t)
            self.logger.add_scalar(
                f'EpsReturn/Min-{policy_name}',
                logs['MinEpsReturn'], l_t)
            self.logger.add_scalar(f'EpsReturn/Average-{policy_name}',
                                   logs['AverageEpsReturn'], l_t)

            self.logger.add_scalar(
                f'Loss/Av-q-{policy_name}',
                np.mean(self.q_losses), l_t)
            self.logger.add_scalar(
                f'Loss/Av-pi-{policy_name}', np.mean(self.pi_losses), l_t)
            self.logger.flush()

            # Reset loss logs for next udpate
            self.q_losses, self.pi_losses = [], []

            if t == 0:
                first_run_ret = logs['AverageEpsReturn']
                logs['FirstEpsReturn'] = first_run_ret

            print('\n\n')
            print('-' * 15)
            print(policy_name, '\n')
            for k, v in logs.items():
                print(k, v)

            # Save model

            if not global_epoch % self.args.get('save_frequency', 50) \
                    or global_epoch == self.epochs_per_policy - 1:
                ...
                """data = {
                    'epoch': global_epoch,
                    'forward': [
                        self.actor_critic.state_dict(),
                        self.ac_target.state_dict()],
                    'reset': [
                        self.actor_critic_reset.state_dict(),
                        self.ac_target_reset.state_dict()],
                    'optims': {k: v.state_dict() for k, v in self.optimizers.items()}
                }
                self.save(global_epoch, data, path='no_trace_model.pt') """


def main():
    """
        Leave no Trace
    """
    env = gym.make('HalfCheetah-v2')

    ac_kwargs = {
        'hidden_sizes': [64, 64, 64, 64],
        'size': 4
    }
    agent_args = {
        'env_name': 'HCv2'
    }
    train_args = {
        'eval_episodes': 5,
        'seed': 0,
        'save_frequency': 120,
        'load_model': False,
        'device': 'cpu',
        'max_eps_len': 150,
        'evaluate_agent': False,
        'q_lr': 1e-4,
        'pi_lr': 1e-4,
        'exploration_steps': 1000,
        'steps_per_epoch': 500,
        'batch_size': 128,
    }

    all_args = {'ac_kwargs': ac_kwargs,  **agent_args, **train_args}
    agent = Agent(env, args=all_args)
    agent.train_agent()


if __name__ == '__main__':
    main()
