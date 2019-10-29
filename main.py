
import sys
# Parse command line arguments
args = sys.argv

from Utils import AnaliseResults
import tensorflow as tf
import os
import time
import random
import numpy as np
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import importlib.util

import matplotlib

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')

import matplotlib.pyplot as plt



# Get the protocol info


# try:

for experiment in args[1::]:

    path_protocol = 'Protocols.' + experiment
    variables = importlib.import_module(path_protocol).variables()

    for seed in variables.seeds:
        tf.set_random_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        #variables.env.env.seed(seed)   # Should I set the seed of the environment as well?

        print(variables.FILE_NAME)

        start = time.time()

        epochs = []
        rewards = []
        epoch = 0

        #if variables.BufferFillAgent is not None:
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
            print(epoch, nstep, reward)  # (sum(rewards[-10:])/10))
            epochs.append(epoch)
            rewards.append(reward)

        end = time.time()

        # AnaliseResults.save_data(variables.RESULTS_FOLDER, variables.FILE_NAME, [epochs, rewards, (end - start)])

        del variables

        # finally:
        #    AnaliseResults.save_data(variables.RESULTS_FOLDER, variables.FILE_NAME, [epochs, rewards])
        #    AnaliseResults.reward_over_episodes(epochs, rewards)
        #    variables.analyze_memory.plot_memory_distribution()
        #
        #    if variables.sess is not None:
        #        variables.sess.close()
        #    # agent.brain.model.save("Boh.h5")
        #
        print("FINISHED")

        tf.keras.backend.clear_session()
