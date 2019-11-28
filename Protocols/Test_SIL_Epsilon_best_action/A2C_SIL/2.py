from Models.A2CSILnetworksEager import *
import tensorflow as tf
import os
from Environment import Environment
from Utils import AnaliseResults, Preprocessing, AnalyzeMemory
from Agents import A2CSILAgent
import gym
import Environments.Gridenv_envs.examples
from Utils import SaveResult

class variables():

    def __init__(self):

        tf.enable_eager_execution()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        self.seeds = range(3)
        self.PROBLEM = 'GE_MazeKeyDoor-v20'
        self.ACTION_SPACE = [0, 1, 2, 3, 4]
        self.GAMMA = 0.99
        self.LEARNING_RATE = 0.0001
        self.RESULTS_FOLDER = 'TEST_A2C_SIL/'
        self.SAVE_RESULT = SaveResult(self.RESULTS_FOLDER)
        self.FILE_NAME = 'Key_Door_A2C_SIL'
        self.BATCH_SIZE = 6
        self.SIL_BATCH_SIZE = 64
        self.IMITATION_BUFFER_SIZE = self.SIL_BATCH_SIZE * 1000
        self.IMITATION_LEARNING_STEPS = 4
        self.WEIGHT_MSE = 0.5
        self.WEIGHT_SIL_MSE = 0.01
        self.WEIGHT_CE_EXPLORATION = 0.0001


        self.NUMBER_OF_EPOCHS = 4000

        self.preprocess = None

        environment = gym.make(self.PROBLEM)

        self.number_of_stacked_frames = 1

        preprocessing = Preprocessing(84, 84, 3, self.number_of_stacked_frames)

        display_env = False

        if display_env:
            from Utils import ShowRenderHRL
            rendering = ShowRenderHRL
        else:
            rendering = False

        self.env = Environment(environment, preprocessing=preprocessing, rendering_custom_class=rendering)

    def reset(self):
        self.env.close()

        # Just to be sure that we don't have some others graph loaded
        tf.reset_default_graph()

        shared_conv_layers = SharedConvLayers()

        self.a2cDNN_SIL = A2CSILEagerSync(30, len(self.ACTION_SPACE), CriticNetwork, ActorNetwork,
                                          self.LEARNING_RATE, self.WEIGHT_MSE, self.WEIGHT_SIL_MSE,
                                          self.WEIGHT_CE_EXPLORATION, shared_conv_layers)

        self.randomAgent = None

        self.agent = A2CSILAgent(self.ACTION_SPACE, self.a2cDNN_SIL, self.GAMMA, self.BATCH_SIZE, self.SIL_BATCH_SIZE,
                                 self.IMITATION_BUFFER_SIZE, self.IMITATION_LEARNING_STEPS)