import numpy as np
import torch


class ReplayBuffer:
    """
        Transitions buffer
        Stores transitions for a single episode
    """

    def __init__(self, act_dim, obs_dim, size=1000000):

        self.rewards = np.zeros([size], dtype=np.float32)
        self.actions = np.zeros([size, act_dim], dtype=np.float32)
        self.states = np.zeros([size, obs_dim], dtype=np.float32)
        self.n_states = np.zeros([size, obs_dim], dtype=np.float32)

        self.dones = np.zeros([size], dtype=np.float32)
        self.log_prob = np.zeros([size], dtype=np.float32)

        self.ptr, self.size = 0, 0
        self.max_size = size

    def store(self, act, states, n_states,  rew, dones, log_p):
        """
            Store transitions
        """
        idx = self.ptr % self.max_size

        self.rewards[idx] = rew
        self.actions[idx] = act
        self.states[idx] = states
        self.n_states[idx] = n_states
        self.log_prob[idx] = log_p
        self.dones[idx] = dones

        self.ptr += 1
        self.size = min(self.ptr + 1, self.max_size)

    def sample_recent(self, batch_size):
        """
            Returns recent transitions of size batch_size
            in order: act, rew, obs, n_obs, dones, log_p
        """
        assert self.ptr >= batch_size

        return (
            torch.as_tensor(
                self.actions[-batch_size:], dtype=torch.float32),
            torch.as_tensor(
                self.rewards[-batch_size:], dtype=torch.float32),
            torch.as_tensor(
                self.states[-batch_size:], dtype=torch.float32),
            torch.as_tensor(
                self.n_states[-batch_size:], dtype=torch.float32),
            torch.as_tensor(
                self.dones[-batch_size:], dtype=torch.float32),
            torch.as_tensor(
                self.log_prob[-batch_size:], dtype=torch.float32)
        )

    def sample_random(self, batch_size, itr_limit=20):
        """
            Randomly sample trajectories upto the most recent
            itr_limit iterations

            in order: act, rew, obs, n_obs, dones, log_p
        """

        lowest_iter = itr_limit * batch_size
        low_ = 0
        if self.ptr > lowest_iter:
            low_ = lowest_iter
        idx = np.random.randint(
            low=self.ptr - low_, high=self.ptr, size=batch_size)

        return (
            torch.as_tensor(
                self.actions[idx], dtype=torch.float32),
            torch.as_tensor(
                self.rewards[idx], dtype=torch.float32),
            torch.as_tensor(
                self.states[idx], dtype=torch.float32),
            torch.as_tensor(
                self.n_states[idx], dtype=torch.float32),
            torch.as_tensor(
                self.dones[idx], dtype=torch.float32),
            torch.as_tensor(
                self.log_prob[idx], dtype=torch.float32)

        )


def airl(env, actor=core.MLPActor, n_epochs=50, steps_per_epoch=5000, max_eps_len=1000, clip_ratio=.2, entropy_reg=.1, **args):
    """
        Learning Robust Rewards with Adversarial Inversere RL
        Algorithm used: Soft PPO

        args: buffer_size:int(1e6), disc_lr: 2e-4,
    """

    obs_space = env.observation_space
    act_space = env.action_space

    act_dim = act_space.shape[0] if not isinstance(act_space,
                                                   gym.spaces.Discrete) else act_space.n
    obs_dim = obs_space.shape[0]

    actor = actor_class(obs_space=obs_space,
                        act_space=act_space, **args['ac_args'])
    params = [core.count(module) for module in (actor.pi, actor.disc)]
    print(f'\nParameters\npi: {params[0]}  discr: { params[1] }')

    memory = ReplayBuffer(act_dim, obs_dim, max_size=args['buffer_size']
                          )

    pi_optimizer = optim.Adam(actor.pi.parameters(), args.get('pi_lr') or 1e-4)
    discr_optimizer = optim.Adam(
        actor.disc.parameters(), args.get('disc_lr') or 1e-4)

    run_t = time.strftime('%Y-%m-%d-%H-%M-%S')
    path = os.path.join('data', env.unwrapped.spec.id +
                        args.get('env_name', '') + '_' + run_t)

    logger = SummaryWriter(log_dir=path)

    def update(epoch, train_args=args):
        """
            Perfroms gradient update on pi and discriminator
        """
        data = memory.sample_random(steps_per_epoch)
        act, rew, obs, obs_n, dones, log_p = data

    start_time = time.time()
    obs = env.reset()
    eps_len, eps_ret = 0, 0

    for t in range(n_epochs):
        eps_len_logs, eps_ret_logs = [], []
        for step in range(steps_per_epoch):
            a, log_p = actor.step(torch.as_tensor(obs, dtype=torch.float32))

            obs_n, rew, done, _ = env.step(a)

            eps_len += 1
            eps_ret += rew

            memory.store(a, obs, obs_n,  rew, done, log_p)
            obs = obs_n

            terminal = done or eps_len == max_eps_len

            if terminal or step == steps_per_epoch - 1:

                if terminal:
                    # only log these for terminals
                    eps_len_logs += [eps_len]
                    eps_ret_log += [eps_ret]

                obs = env.reset()
                eps_len, eps_ret = 0, 0

        update(t + 1)

        # Perform logging
