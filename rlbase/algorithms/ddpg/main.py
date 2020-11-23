
from d2rl import MLPActorCritic

from ddpg import ddpg


def main():
    """
        DDPG run
    """
    en_nm = 'InvertedPendulum-v2'
    env = gym.make(en_nm)
    test_env = gym.make(en_nm)

    ac_kwargs = {'hidden_sizes': [256, 256],
                 'actor_critic': MLPActorCritic
                 }
    agent_args = {'env_name': 'HCv2'}
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
        'batch_size': 128
    }

    args = {'ac_kwargs': ac_kwargs, **agent_args, **train_args}

    ddpg(env,  **args)


if __name__ == '__main__':
    main()
