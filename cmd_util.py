"""
Helpers for scripts like run_atari.py.
"""

import os
import random

# try:
#     from mpi4py import MPI
# except ImportError:
#     MPI = None
#
import gym
import numpy as np
from atari_wrappers import make_atari, wrap_deepmind
from vec_env.monitor import Monitor

from vec_env.dummy_vec_env import DummyVecEnv
from vec_env.subproc_vec_env import SubprocVecEnv


def set_global_seeds(i):
    try:
        import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0

    myseed = i + 1000 * rank if i is not None else None
    try:
        import tensorflow as tf
        tf.set_random_seed(myseed)
    except ImportError:
        pass
    np.random.seed(myseed)
    random.seed(myseed)


def make_vec_env(env_id, env_type, num_env, seed,
                 start_index=0,
                 logger_dir="tmp/",
                 force_dummy=False):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
    """
    mpi_rank = 0  # MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = seed + 10000 * mpi_rank if seed is not None else None

    def make_thunk(rank):
        return lambda: make_env(
            env_id=env_id,
            env_type=env_type,
            mpi_rank=mpi_rank,
            subrank=rank,
            seed=seed,
            logger_dir=logger_dir,
        )

    set_global_seeds(seed)
    if not force_dummy and num_env > 1:
        return SubprocVecEnv([make_thunk(i + start_index) for i in range(num_env)])
    else:
        return DummyVecEnv([make_thunk(i + start_index) for i in range(num_env)])


def make_env(env_id, env_type, mpi_rank=0, subrank=0, seed=None, logger_dir=None, ):
    if env_type == 'atari':
        env = make_atari(env_id)
    else:
        env = gym.make(env_id)

    env.seed(seed + subrank if seed is not None else None)
    env = Monitor(env, logger_dir and os.path.join(logger_dir, str(mpi_rank) + '.' + str(subrank)),
                  allow_early_resets=True)

    if env_type == 'atari':
        env = wrap_deepmind(env)

    return env


def test_env():

    #env = make_env("MontezumaRevengeDeterministic-v4", "atari")
    env = make_env("MontezumaRevengeNoFrameskip-v4", "atari")
    s = env.reset()
    print(s.shape)
    s_ram = env.unwrapped._get_ram()
    a = env.action_space.sample()
    env.step(a)
    s, r, d, info = env.step(a)
    print(s_ram.shape)

if __name__ == '__main__':
    test_env()

