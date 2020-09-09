"""AIRL Runner"""

from adv_irl import airl

from torch import nn

import click
import gym


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
@click.option('--epochs', '-ep', type=int, default=100)
@click.option('--steps_per_epoch', '-spe', type=int, default=1000)
@click.option('--max_eps_len', '-ep_len', type=int)
@click.option('--pi_learning_rate', '-pi_lr', type=float, default=1e-4)
@click.option('--disc_learning_rate', '-d_lr', type=float, default=2e-4)
@click.option('--seed', type=int, default=1)
def main(**args):
    """
        User interaction & entry
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
            'activation': nn.ReLU
        }
    }

    ac_args.update(**disc_args)

    train_args = {
        'pi_train_n_iters': 80,
        'disc_train_n_iters': 40,
        'max_kl': 5,
        'entropy_reg': .1,
        'clip_ratio': .2,
        'max_eps_len': 150,
        'real_label': 1,
        'pi_label': 0
    }
    agent_args = {
        'n_epochs': 100,
        'env_name': '',  # 'b_10000_plr_.1e-4',
        'steps_per_epoch': 10000
    }

    all_args = {
        'ac_args': ac_args,
        'pi_lr': 3e-4,
        'disc_lr': 2e-4,
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
