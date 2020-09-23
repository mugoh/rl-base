"""AIRL Runner"""
import click
import gym
from torch import nn

from adv_irl import airl


@click.command()
@click.option('--env',
              type=str,
              default='HalfCheetah-v2',
              help='Gym environment name')
@click.option('--env_name',
              type=str,
              default='',
              help="Name to assign log data")
@click.option('--expert_data_path',
              '-demos',
              type=str,
              default='.data/expert_data_13-08-2020_16-09-18.npz',
              help="Path to expert data file")
@click.option('--epochs', '-ep', type=int)
@click.option('--steps_per_epoch', '-spe', type=int, default=10000)
@click.option('--max_eps_len', '-ep_len', type=int)
@click.option('--pi_learning_rate', '-pi_lr', type=float)
@click.option('--disc_learning_rate', '-d_lr', type=float)
@click.option('--seed', type=int, default=1)
@click.option('--target_kl',
              '-kl',
              type=float,
              help="Maximum kl between new and old trained policies")
@click.option(
    '--kl_start',
    '-kl_start',
    type=int,
    help="Iteration at which to start checking for target KL divergence")
@click.option('--pi_train_n_iters',
              '-pi_updates',
              type=int,
              help='Number of policy updates to perform per epoch')
@click.option('--disc_train_n_iters',
              '-d_updates',
              type=int,
              help='Number of discriminator updates to perform per epoch')
@click.option('--entropy_reg', '-tm',
              type=float, help='Temperature for entopy regularization.\n' +
              'Between 0 and 1. Higher value encourages stochasticity')
def main(**args):
    """
        Adversarial Inverser RL runner
    """
    env = gym.make(args.pop('env'))

    ac_args = {'hidden_size': [64, 64], 'size': 2}

    # Discriminator approximators
    disc_args = {
        'g_args': {
            'hidden_layers': [32, 1],
            'size': 1,
            'activation': nn.Identity
        },
        'h_args': {
            'hidden_layers': [32, 32, 1],
            'size': 2,
            'activation': nn.LeakyReLU
        }
    }

    ac_args.update(**disc_args)

    train_args = {
        'pi_train_n_iters': 80,
        'disc_train_n_iters': 40,
        'max_kl': args.pop('target_kl') or 1.,
        'kl_start': 20,
        'entropy_reg': .1,
        'clip_ratio': .2,
        'max_eps_len': 150,
        'real_label': 1,
        'pi_label': 0
    }
    agent_args = {
        'n_epochs': args.pop('epochs') or 250,
        'env_name': '',  # 'b_10000_plr_.1e-4',
        'steps_per_epoch': 10000
    }

    all_args = {
        'ac_args': ac_args,
        'pi_lr': 2e-4,
        'disc_lr': 1e-4,
        'gamma': .99,
        'buffer_size': int(1e6),
        **agent_args,
        **train_args,
        **{k: v
           for k, v in args.items() if v}
    }

    airl(env, **all_args)


if __name__ == '__main__':
    main()
