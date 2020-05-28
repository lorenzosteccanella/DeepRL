import matplotlib
import os
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

import sys
# Parse command line arguments
args = sys.argv

import time
import random
import numpy as np
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import importlib.util
import tensorflow as tf

from Agents import HrlAgent

import gridenvs.examples



def make_env(env_id, wrapper, wrapper_params, rank, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    import gym

    def _init():
        env = gym.make(env_id)
        if wrapper is not False:
            env = wrapper(env, wrapper_params)
        env.seed(seed + rank)  # are u sure about different seeds?

        print("SEED MULTI ENVIRONMENT " + str(rank) + " : " + str(seed + rank))
        return env

    return _init

def moving_average(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'valid')

def has_method(o, name):
    return callable(getattr(o, name, None))

def run(variables):
    print(variables.FILE_NAME)

    start = time.time()

    epochs = []
    rewards = []
    n_steps = []
    epoch = 0
    nstep = 0

    # if variables.BufferFillAgent is not None:total_r
    #    print("RANDOM EXPLORATION TO FILL EXPERIENCE REPLAY BUFFER")
    #    while variables.randomAgent.exp < variables.RANDOM_EXPLORATION:
    #        epoch, nstep, reward = variables.env.run(variables.randomAgent)
    #        epochs.append(epoch)
    #        rewards.append(reward)

    #    variables.agent.buffer = variables.randomAgent.buffer

    #    variables.randomAgent = None

    print("START TO LEARN")
    while nstep < variables.NUMBER_OF_STEPS:#epoch < variables.NUMBER_OF_EPOCHS:
        epoch, nstep, reward = variables.env.run(variables.agent)
        print(epoch, nstep, reward, (sum(rewards[-100:]) / 100))
        epochs.append(epoch)
        rewards.append(reward)
        n_steps.append(nstep)

        if variables.SAVE_RESULT is not False:
            message = str(epoch) + " " + str(nstep) + " " + str(reward) + " " + str((sum(rewards[-100:]) / 100)) + "\n"
            variables.SAVE_RESULT.save_data("on_going_results", message)



    end = time.time()

    return epochs, rewards, n_steps


if __name__ == '__main__':
    for experiment in args[1::]:
        path_protocol = 'Protocols.' + experiment
        variables = importlib.import_module(path_protocol).variables()

        for seed in variables.seeds:
            print("\n" * 6)
            print("SEED : " + str(seed))
            variables.reset()
            if variables.SAVE_RESULT is not False:
                variables.SAVE_RESULT.set_seed(seed)
                parameters = vars(variables)
                variables.SAVE_RESULT.save_settings(parameters)
            tf.random.set_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

            if variables.multi_processing is False:
                variables.env.env.seed(seed)   # Should I set the seed of the environment as well?
            else:
                from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
                environment = SubprocVecEnv(
                    [make_env(variables.PROBLEM, variables.wrapper, variables.wrapper_params, i) for i in
                     range(variables.num_workers)])
                variables.env.set_env(environment)

            epochs, rewards, n_steps = run(variables)
            moving_average_reward = moving_average(rewards, 100)

            if variables.SAVE_RESULT is not False:
                message = [str(e) + " " + str(nstep) + " " + str(r) + "\n" for e, r, nstep in zip(epochs, moving_average_reward, n_steps)]
                variables.SAVE_RESULT.save_data(variables.FILE_NAME, message)
                variables.SAVE_RESULT.plot_reward_ep(variables.FILE_NAME, "reward-over-episodes", "episodes", "reward")
                variables.SAVE_RESULT.plot_reward_nstep(variables.FILE_NAME, "reward-over-nsteps", "nsteps", "reward")

                if isinstance(variables.agent, HrlAgent):
                    if isinstance(variables.agent, list):
                        variables.SAVE_RESULT.plot_success_rate_transitions(variables.agent[0].FILE_NAME + "Transitions_performance")
                    else:
                        variables.SAVE_RESULT.plot_success_rate_transitions(variables.agent.FILE_NAME + "Transitions_performance")

            if(has_method(variables, 'transfer_learning_test')):
                for _ in range(len(variables.TEST_TRANSFER_PROBLEM)):
                    tf.random.set_seed(seed)
                    random.seed(seed)
                    np.random.seed(seed)
                    if variables.multi_processing is False:
                        variables.env.env.seed(seed)  # Should I set the seed of the environment as well?
                    else:
                        from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
                        environment = SubprocVecEnv(
                            [make_env(variables.PROBLEM, variables.wrapper, variables.wrapper_params, i) for i in
                             range(variables.num_workers)])
                        variables.env.set_env(environment)
                    print("\n"*6)
                    variables.transfer_learning_test()
                    epochs, rewards, n_steps = run(variables)
                    moving_average_reward = moving_average(rewards, 100)
                    message = [str(e) + " " + str(nstep) + " " + str(r) + "\n" for e, r, nstep in zip(epochs, moving_average_reward, n_steps)]
                    if variables.SAVE_RESULT is not False:
                        variables.SAVE_RESULT.save_data(variables.TRANSFER_FILE_NAME, message)
                        variables.SAVE_RESULT.plot_reward_ep(variables.TRANSFER_FILE_NAME, "reward-over-episodes", "episodes", "reward")
                        variables.SAVE_RESULT.plot_reward_nstep(variables.TRANSFER_FILE_NAME, "reward-over-nsteps", "nsteps", "reward")
                        if isinstance(variables.agent, HrlAgent):
                            if isinstance(variables.agent, list):
                                variables.SAVE_RESULT.plot_success_rate_transitions(variables.agent[0].FILE_NAME + "Transitions_performance")
                            else:
                                variables.SAVE_RESULT.plot_success_rate_transitions(variables.agent.FILE_NAME + "Transitions_performance")


        variables.SAVE_RESULT.plot_multiple_seeds_reward_ep(variables.FILE_NAME, "reward-over-episodes", "episodes", "reward")
        variables.SAVE_RESULT.plot_multiple_seeds_reward_nstep(variables.FILE_NAME, "reward-over-nsteps", "nsteps", "reward")
        if isinstance(variables.agent, HrlAgent):
            variables.SAVE_RESULT.plot_multiple_seeds("Transitions_performance", "success rate of options' transitions", "number of options executed", "% of successful option executions")
        if (has_method(variables, 'transfer_learning_test')):
            for file_name_transfer in variables.TEST_TRANSFER_PROBLEM:
                variables.SAVE_RESULT.plot_multiple_seeds_reward_ep(variables.FILE_NAME + " - " + file_name_transfer, "reward-over-episodes", "episodes", "reward")
                variables.SAVE_RESULT.plot_multiple_seeds_reward_nstep(variables.FILE_NAME + " - " + file_name_transfer, "reward-over-nsteps", "nsteps", "reward")




        del variables
        print("FINISHED")

        # finally:
        #    AnaliseResults.save_data(variables.RESULTS_FOLDER, variables.FILE_NAME, [epochs, rewards])
        #    AnaliseResults.reward_over_episodes(epochs, rewards)
        #    variables.analyze_memory.plot_memory_distribution()
        #
        #    if variables.sess is not None:
        #        variables.sess.close()
        #    # agent.brain.model.save("Boh.h5")
    #
