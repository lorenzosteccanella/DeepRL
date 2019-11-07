from Models.A2CnetworksEager import *
import tensorflow as tf
import os
from Environment import Environment
from Utils import AnaliseResults, Preprocessing, AnalyzeMemory
from Agents import A2CAgent
import gym
import gridenvs.examples
from Utils import SaveResult

class variables():

    def __init__(self):

        tf.enable_eager_execution()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        self.seeds = [1]
        self.PROBLEM = 'CartPole-v0'
        self.ACTION_SPACE = [0, 1]
        self.GAMMA = 0.99
        self.LEARNING_RATE = 0.001
        self.RESULTS_FOLDER = 'A2C/'
        self.SAVE_RESULT = SaveResult(self.RESULTS_FOLDER)
        self.FILE_NAME = 'cartpole_A2C'
        self.BATCH_SIZE = 32
        self.WEIGHT_MSE = 0.5
        self.WEIGHT_CE_EXPLORATION = 0.0001

        self.NUMBER_OF_EPOCHS = 1000

        self.preprocess = None

        self.env = Environment(gym.make(self.PROBLEM), self.preprocess)



    def reset(self):
        self.env.close()

        # Just to be sure that we don't have some others graph loaded
        tf.reset_default_graph()

        self.SharedDenseLayers = SharedDenseLayers(30)

        self.a2cDNN = A2CEagerSync(30, len(self.ACTION_SPACE), CriticNetwork, ActorNetwork,
                                   self.LEARNING_RATE, self.WEIGHT_MSE, self.WEIGHT_CE_EXPLORATION, self.SharedDenseLayers)

        self.randomAgent = None

        self.agent = A2CAgent(self.ACTION_SPACE, self.a2cDNN, self.GAMMA, self.BATCH_SIZE)