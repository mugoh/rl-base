import click
import gym

from ppo import ppo


@click.command()
@click.option('--env',
              type=str,
              default='HalfCheetah-v2',
              help='Gym environment name')
@click.option('--env_name',
              type=str,
              default='',
              help="Name to assign log data")
@click.option('--n_epochs', '-ep',
              type=int,
              help='Number of iterations (updates) to run')
@click.option('--steps_per_epoch', '-spe',
              type=int,
              help='Number of env steps per update')
@click.option('--max_eps_len', '-ep_len', type=int)
@click.option('--pi_lr', '-plr', type=float,  help='Policy learning rate')
@click.option('--v_lr', '-vlr', type=float,  help='Value function learning rate')
@click.option('--seed', type=int, default=1)
@click.option('--max_kl',
              '-kl',
              type=float,
              help="KL target between new and old policies. Used to trigger early stopping")
@click.option('--min_expert_return',
              '-min_ret', type=float,
              help='Minimum average reward on an epoch for which to add to the expert data')
@click.option('--n_demo_itrs',
              '-n_demos',
              type=int,
              help="Number of epochs for which to collect the expert data.\n" +
              'This needs to be larger than `n_epochs`')
@click.option('--last_n_epochs',
              '-last_e',
              type=bool,
              default=True,
              help='Whether to collect the final `n_demo_itrs` for the specified `n_epochs`.\n' +
              'If True, the collected data doesn\'t have to be above the `min_ret` as ' +
              'long as its among the last n epochs')
def main(**args):
    """
        Ppo expert policy data collector
    """

    env = gym.make(args.pop('env'))

    ac_args = {'hidden_size': [64, 64], 'size': 2}
    train_args = {
        'pi_train_n_iters': 80,
        'v_train_n_iters': 80,
        'max_kl': .01,
        'max_eps_len': 150
    }
    agent_args = {
        'n_epochs': 250,
        'env_name': '',
        'steps_per_epoch': 10000
    }

    collect_expert_policy = {
        'min_expert_return': 130,
        'n_demo_itrs': 25,
        'last_n_epochs': False
    }

    all_args = {
        'ac_args': ac_args,
        'pi_lr': 2e-4,
        'v_lr': 1e-3,
        'gamma': .99,
        'lamda': .995,
        **agent_args,
        **train_args,
        **collect_expert_policy,

        **{k: v
           for k, v in args.items() if v}
    }

    ppo(env, **all_args)


if __name__ == '__main__':
    main()
