"""SAC"""

import copy
import time
import os

import torch
import torch.nn as nn
import numpy as np


from torch.utils.tensorboard import SummaryWriter

import gym

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


def sac(env, ac_kwargs={},
        actor_critic=core.MLPActorCritic, memory_size: int = int(1e6),
        exploration_steps: int = 10000, max_eps_len: int = 1000,
        start_update: int = 3000,
        steps_per_epoch: int = 5000,
        epochs: int = 100,
        batch_size: int = 64, alpha: float = .2,
        q_lr: float = 1e-3,
        pi_lr: float = 1e-4, gamma: float = .99,
        polyyak: float = .995,
        seed: int = 1, update_frequency: 50,
        evaluation_episodes=50, **args):
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

        q_lr (float): Q function learning rate

        pi_lr (float): Policy learning rate

        gamma (float): Reward discounting factor

        polyyak (float): Interpolation factor during updating of
            target network

        update_frequency ( int ): Numbeer of times to perform an update
            for each epoch
    """

    np.random.seed(seed)
    torch.manual_seed(seed)

    device_nm = args.get('device') or (
        'cuda' if torch.cuda.is_available else 'cpu')
    device = torch.device(device_nm)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    act_limit = env.action_space.high[0]

    r_buffer = ReplayBuffer(
        size=memory_size,
        act_dim=act_dim,
        obs_dim=obs_dim, device=device)
    # Init pi and Qs parameters
    actor_critic = actor_critic(
        obs_dim, act_dim, act_limit, **ac_kwargs).to(device)
    ac_target = copy.deepcopy(actor_critic).to(device)

    q_loss_f = nn.MSELoss()

    q_params = list(actor_critic.q_1.parameters()) + \
        list(actor_critic.q_2.parameters())
    q_optim = torch.optim.Adam(q_params, lr=q_lr)
    pi_optim = torch.optim.Adam(actor_critic.pi.parameters(), pi_lr)

    print(f'Param counts: {core.count(actor_critic)}\n')

    run_t = time.strftime('%Y-%m-%d-%H-%M-%S')
    path = os.path.join(
        'data', env.unwrapped.spec.id + args.get('env_name', '') + '_' + run_t)

    logger = SummaryWriter(log_dir=path)
    q_losses,  pi_losses = [], []

    def eval_agent(epoch, kwargs=args):
        """Evaluate the updated policy

          During evaluation, use the mean action
        """
        test_env = kwargs['test_env']
        eval_eps_lens = []
        eval_eps_rets = []

        for eps in range(evaluation_episodes):
            eps_length, eps_return, obsv = 0, 0, test_env.reset()

            for _ in range(kwargs.get('eval_steps_per_epoch', 200)):
                obsv = torch.from_numpy(obsv).float().to(device)
                act = actor_critic.act(obsv, mean_act=True)

                obsv_n, rew, done, _ = env.step(act)

                eps_length += 1
                eps_return += rew
                obsv = obsv_n

                if done or eps_length == kwargs.get('eval_steps_per_epoch', 100):
                    eval_eps_lens.append(eps_length)
                    eval_eps_rets.append(eps_return)
            logger.add_scalar('Evaluation/Return', np.mean(eps_return), epoch)
            logger.add_scalar('Evaluation/EpsLen', np.mean(eps_length), epoch)
        return np.mean(eval_eps_lens), np.mean(eval_eps_rets)

    def zero_optim(optimizer, set_none=False):
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
        obs = data['obs']

        act_tilde, pi = actor_critic.pi(obs, return_pi=True)
        qv_1 = actor_critic.q_1(obs, act_tilde)
        qv_2 = actor_critic.q_2(obs, act_tilde)

        #q_v = ([qv_1, qv_2])
        q_v = torch.min(qv_1, qv_2)

        pi_loss = q_v - alpha * actor_critic.pi.log_p(pi, act_tilde)

        return -pi_loss.mean()

    def compute_q_loss(data):
        """Returns Q loss"""
        rew = data['rew']
        dones = data['dones']
        obs_n = data['obs_n']
        obs = data['obs']
        act = data['act']

        # target = r + gamma(1 - d)[Q_t(s', a') + alpha * H(pi(a_tilde, xi))]

        # Min Q target
        q1_pred = ac_target.q_1(obs_n, act)
        q2_pred = ac_target.q_2(obs_n, act)

        with torch.no_grad():
            act_tilde, pi = actor_critic.pi(obs_n, return_pi=True)
            q1_target = ac_target.q_1(obs, act_tilde)
            q2_target = ac_target.q_2(obs, act_tilde)

            # q_target= min([q1_pred, q2_pred])
            q_target = torch.min(q1_target, q2_target)

            backup = rew + gamma * (1 - dones) * \
                (q_target - alpha * actor_critic.pi.log_p(pi, act_tilde))

        q1_loss = q_loss_f(ac_target.q_1(obs, act), backup)
        q2_loss = q_loss_f(ac_target.q_2(obs, act), backup)

        return q1_loss + q2_loss

    def update(data, n_epoch):
        """Updates the policy and Q functions"""

        torch.autograd.set_detect_anomaly(True)

        zero_optim(q_optim)
        q_loss = compute_q_loss(data)

        q_loss.backward()

        q_optim.step()

        nonlocal q_losses, pi_losses

        # update pi
        # Freeze Q-network to not waste resources computing grads
        for p in q_params:
            p.requires_grad = False

        zero_optim(pi_optim)
        pi_loss = compute_pi_loss(data)

        pi_loss.backward()
        pi_optim.step()

        q_losses.append(q_loss.item())
        pi_losses.append(pi_loss.item())

        logger.add_scalar('Loss/Q', q_loss.item())

        for p in q_params:
            p.requires_grad = True

        # update target

        with torch.no_grad():
            phi_params = actor_critic.parameters()
            phi_target_params = ac_target.parameters()

            for param, target_param in zip(phi_params, phi_target_params):

                # rho(target) + (1 - rho)param
                target_param.data.mul_(polyyak)
                target_param.data.add_(param.data.mul(1 - polyyak))

    eps_len, eps_ret = 0, 0

    start_time = time.time()
    obs = env.reset()

    for epoch in range(epochs):
        eps_len_logs, eps_ret_logs = [], []

        for t in range(steps_per_epoch):
            steps_run = (epoch * steps_per_epoch) + t + 1

            if steps_run <= exploration_steps:
                act = env.action_space.sample()

            else:
                obs = torch.from_numpy(obs).float().to(device)
                act = actor_critic.act(obs, mean_act=False)

            obs_n, rew, done, _ = env.step(act)
            r_buffer.store(obs, act, obs_n, rew, done)

            obs = obs_n

            eps_len += 1
            eps_ret += rew

            terminal = done or eps_len == max_eps_len

            if terminal:
                # append logs
                eps_len_logs.append(eps_len)
                eps_ret_logs.append(eps_ret)

                # reset state
                obs, eps_ret, eps_len = env.reset(), 0, 0

            # update
            if steps_run >= start_update and not steps_run % update_frequency:
                for _ in range(update_frequency):
                    data = r_buffer.sample(batch_size)
                    update(data, epoch)
        logs = dict(Epoch=epoch,
                    AverageEpisodeLength=np.mean(eps_len_logs),
                    AverageEpisodeReturn=np.mean(eps_ret_logs),
                    MinEpsReturn=np.min(eps_ret_logs),
                    MaxEpsReturn=np.max(eps_ret_logs),
                    RunTime=time.time()-start_time
                    )

        if args.get('evaluate_agent'):
            eval_eps_len, eval_eps_ret = eval_agent(epoch)
            logs['EvalAvEpsLength'] = eval_eps_len
            logs['EvalAvReturn'] = eval_eps_ret

        l_t = epoch
        logger.add_scalar('AvEpsLen', logs['AverageEpisodeLength'], l_t)
        logger.add_scalar('EpsReturn/Max', logs['MaxEpsReturn'], l_t)
        logger.add_scalar('EpsReturn/Min', logs['MinEpsReturn'], l_t)
        logger.add_scalar('EpsReturn/Average',
                          logs['AverageEpisodeReturn'], l_t)

        if l_t == 0:
            first_run_ret = logs['AverageEpisodeReturn']
            logs['FirstEpsReturn'] = first_run_ret

        q_l_mean = np.mean(q_losses)
        logs['QLossAvg'] = q_l_mean

        logger.add_scalar('Loss/Av-q', q_l_mean, l_t)
        logger.add_scalar('Loss/Av-pi', np.mean(pi_losses), l_t)
        logger.flush()

        q_losses,  pi_losses = [], []

        print('\n\n')
        print('-' * 15)
        for k, v in logs.items():
            print(k, v)


def main():
    env_name = 'HalfCheetah-v2'
    env = gym.make(env_name)

    eval_args = {
        'eval_steps_per_epoch': 200,
        'evaluation_episodes': 10,
        'test_env': gym.make(env_name),
        'evaluate_agent': True
    }
    ac_kwargs = {'hidden_sizes': [256, 256]}

    train_args = {
        'seed': 0,
        'device': 'cpu',
        'max_eps_len': 150,
        'evaluate_agent': False,
        'q_lr': 1e-3,
        'pi_lr': 1e-3,
        'gamma': .99,
        'exploration_steps': 10000,
        'steps_per_epoch': 5000,
        'batch_size': 128,
        'epochs': 100
    }
    all_args = {**train_args, **eval_args, 'ac_kwargs': ac_kwargs}
    sac(env, **all_args)


if __name__ == '__main__':
    main()
