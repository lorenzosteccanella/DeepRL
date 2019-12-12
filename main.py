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

    # if variables.BufferFillAgent is not None:total_r
    #    print("RANDOM EXPLORATION TO FILL EXPERIENCE REPLAY BUFFER")
    #    while variables.randomAgent.exp < variables.RANDOM_EXPLORATION:
    #        epoch, nstep, reward = variables.env.run(variables.randomAgent)
    #        epochs.append(epoch)
    #        rewards.append(reward)

    #    variables.agent.buffer = variables.randomAgent.buffer

    #    variables.randomAgent = None

    print("START TO LEARN")
    while epoch < variables.NUMBER_OF_EPOCHS:
        epoch, nstep, reward = variables.env.run(variables.agent)
        print(epoch, nstep, reward, (sum(rewards[-10:]) / 10))
        epochs.append(epoch)
        rewards.append(reward)
        n_steps.append(nstep)

    end = time.time()

    return epochs, rewards, n_steps



# try:

for experiment in args[1::]:
    path_protocol = 'Protocols.' + experiment
    variables = importlib.import_module(path_protocol).variables()

    for seed in variables.seeds:
        print("\n" * 6)
        variables.reset()
        variables.SAVE_RESULT.set_seed(seed)
        parameters = vars(variables)
        variables.SAVE_RESULT.save_settings(parameters)
        tf.set_random_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        #variables.env.env.seed(seed)   # Should I set the seed of the environment as well?

        epochs, rewards, n_steps = run(variables)
        moving_average_reward = moving_average(rewards, 10)

        message = [str(e) + " " + str(nstep) + " " + str(r) + "\n" for e, r, nstep in zip(epochs, moving_average_reward, n_steps)]
        variables.SAVE_RESULT.save_data(variables.FILE_NAME, message)
        variables.SAVE_RESULT.plot_results(variables.FILE_NAME, "reward-over-episodes", "episodes", "reward")
        if isinstance(variables.agent, HrlAgent):
            variables.SAVE_RESULT.plot_success_rate_transitions(variables.agent.FILE_NAME + "Transitions_performance")

        if(has_method(variables, 'transfer_learning_test')):
            for _ in range(len(variables.TEST_TRANSFER_PROBLEM)):
                print("\n"*6)
                variables.transfer_learning_test()
                epochs, rewards, n_steps = run(variables)
                moving_average_reward = moving_average(rewards, 10)
                message = [str(e) + " " + str(nstep) + " " + str(r) + "\n" for e, r, nstep in zip(epochs, moving_average_reward, n_steps)]
                variables.SAVE_RESULT.save_data(variables.TRANSFER_FILE_NAME, message)
                variables.SAVE_RESULT.plot_results(variables.TRANSFER_FILE_NAME, "reward-over-episodes", "episodes", "reward")
                if isinstance(variables.agent, HrlAgent):
                    variables.SAVE_RESULT.plot_success_rate_transitions(variables.agent.FILE_NAME + "Transitions_performance")


    variables.SAVE_RESULT.plot_multiple_seeds(variables.FILE_NAME, "reward-over-episodes", "episodes", "reward")
    #variables.SAVE_RESULT.plot_multiple_seeds("Transitions_performance", "success rate of options' transitions",
    #                                          "number of options executed", "% of successful option executions")
    if (has_method(variables, 'transfer_learning_test')):
        for file_name_transfer in variables.TEST_TRANSFER_PROBLEM:
            variables.SAVE_RESULT.plot_multiple_seeds(variables.FILE_NAME+" - "+ file_name_transfer, "reward-over-episodes", "episodes", "reward")



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




