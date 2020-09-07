import numpy as np
import torch

import torch.nn as nn

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

        lowest_iter = itr_limit * batch_size # Batch size makes 1 iter
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

        Args:
            entropy_reg (float): Entropy regularizer for Soft Ppo (SPPO)
                Denoted :math: `\alpha`

            max_kl (float): KL divergence regulator. Used for early stopping
                when the KL between the new and old policy exceeds this
                threshold we think is appropriate (.01 - .05)

            clip_ratio (float): Clips the old policy objective.
                Determines how far the new policy can go from the old
                policy while still improving the objective

            pi_lr (float): Learning rate for the policy

            disc_lr (float): Learning rate for the discriminator

            seed (int): Random seed generator

            real_label (int): Label for expert data (1)

            pi_label (int): Label for policy samples (0)
    """

    torch.manual_seed(args[ 'seed' ])
    np.random.seed(args[ 'seed' ])

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
        actor.disc.parameters(), args.get('disc_lr') or 2e-4)
    loss_criterion = nn.BCELoss()

    run_t = time.strftime('%Y-%m-%d-%H-%M-%S')
    path = os.path.join('data', env.unwrapped.spec.id +
                        args.get('env_name', '') + '_' + run_t)

    logger = SummaryWriter(log_dir=path)


    def compute_pi_loss(log_p_old, adv_b, act_b, obs_b):
        """
            Pi loss

            adv_b: Advantage estimate from the learned reward function
        """

        # returns new_pi_normal_distribution, logp_act
        pi_new, log_p_ = actor.pi(obs_b, act_b)
        log_p_ = log_p_.type(torch.float32)  # From torch.float64

        pi_ratio = torch.exp(log_p_ - log_p_old)


        # Soft PPO update - Encourages entropy in the policy

        # i.e. Act as randomly as possibly while maximizing the objective
        # Example case: pi might learn to take a certain action for a given
        # state every time because it has some good reward, but forgo
        # trying other actions which might have higher reward

        # A_old_pi(s, a) = A(s, a) - entropy_reg * log pi_old(a|s)
        adv_b = adv_b - entropy_reg * log_p_old

        min_adv = torch.where(adv_b >= 0, (1 + clip_ratio)
                              * adv_b, (1-clip_ratio) * adv_b)

        pi_loss = -torch.mean(torch.min(pi_ratio * adv_b, min_adv))
        kl = (log_p_old - log_p_).mean().item()
        entropy = pi_new.entropy().mean().item()

        return pi_loss, kl, entropy

    def compute_disc_loss(traj, label, d_l_args=args):
        """
            Disciminator loss

            log D_theta_phi (s, a, s') − log(1 − D_theta_phi (s, a, s'))
            (Minimize likelohood of policy samples while increase likelihood
            of expert demonstrations)

            Args:
                traj: (s, a, s') samples
                label: Label for expert data or pi samples
        """

        # obs, obs_n, dones
        # expert_data or pi_samples in traj

        output = actor.disc(*traj).view(-1)

        err_d = loss_criterion(output, label)

        # Average output across the batch of the Discriminator
        # For expert data,
        # Should start at ~ 1 and converge to 0.5

        # For sample data, should start at 0 and converge to 0.5
        d_x = output.mean().item()


        # Call err_demo.backward first!
        return  d_x, err_d




    def update(epoch, train_args=args):
        """
            Perfroms gradient update on pi and discriminator
        """
        data = memory.sample_random(steps_per_epoch)
        act, rew, obs, obs_n, dones, log_p = data

        batch_size = d_l_args['steps_per_epoch']
        real_label = d_l_args['real_label']
        pi_label = d_l_args['pi_label']

        # loss before update
        pi_loss_old, kl, entropy = compute_pi_loss(
            log_p_old=log_p_old, obs_b=obs_b, adv_b=adv_b, act_b=act_b)

        label = torch.full((batch_size,), real_label, dtype=torch.float32)

        demo_info = compute_disc_loss({}, label=label)
        pi_samples_info = compute_disc_loss({}, label.fill_(pi_label))



        av_demo_output_old, err_demo_old = demo_info
        av_pi_output_old, err_pi_samples_old = pi_samples_info

        for i in range(train_args['disc_train_n_iters']):
            # Train with expert demonstrations
            # log(D(s, a, s'))

            actor.disc.zero_grad()

            av_demo_output, err_demo = compute_disc_loss(
                {},
                                                         label.fill_(real_label))

            err_demo.backward()

            # Train with policy samples
            # - log(D(s, a, s'))
            label.fill_(pi_label)
            av_pi_output, err_pi_samples = compute_disc_loss({}, label)
            err_pi_samples = -err_pi_samples

            err_pi_samples.backward()

            discr_optimizer.step()

        for i in range(train_args['pi_train_n_iters']):
            ...


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
