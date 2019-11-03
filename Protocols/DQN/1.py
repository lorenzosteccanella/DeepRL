from Models.QnetworksEager import *
import tensorflow as tf
import os
from Environment import Environment
from Utils import PrioritizedExperienceReplay, ExperienceReplay, AnaliseResults, Preprocessing, UpdateWeightsModels,UpdateWeightsEager, SoftUpdateWeightsEager, AnalyzeMemory, Tools4DQN
from Agents import DQNAgent
import gym
import gridenvs.examples
from Utils import SaveResult


class variables():

    def __init__(self):
        tf.enable_eager_execution()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        self.seeds = [1]
        self.MEMORY_CAPACITY = 40000
        self.RANDOM_EXPLORATION = 0
        self.PROBLEM = 'CartPole-v0'
        self.LAMBDA = 0.0005
        self.UPDATE_TARGET_FREQ = 1
        self.ACTION_SPACE = [0, 1]
        self.GAMMA = 0.95
        self.BATCH_SIZE = 32
        self.MIN_EPSILON = 0.01

        self.NUMBER_OF_EPOCHS = 200

        self.RESULTS_FOLDER = 'DQN/'
        self.SAVE_RESULT = SaveResult(self.RESULTS_FOLDER)
        self.FILE_NAME = 'cartpole_DQN_experience_replay'

        self.preprocess = None

        self.analyze_memory = AnalyzeMemory()

        self.env = Environment(gym.make(self.PROBLEM), self.preprocess)

        self.stateDimension = (4,)
        self.buffer_memory = ExperienceReplay(self.MEMORY_CAPACITY)

        # Just to be sure that we don't have some others graph loaded
        tf.reset_default_graph()

        self.MainQN = QnetworkEager(30, len(self.ACTION_SPACE), DenseModel)
        self.TargetQN = QnetworkEager(30, len(self.ACTION_SPACE), DenseModel)

        self.upd_target = SoftUpdateWeightsEager(weights=self.MainQN, model=self.TargetQN, tau=0.08)

        self.BufferFillAgent = None
        self.agent = DQNAgent(self.ACTION_SPACE, self.stateDimension, self.buffer_memory, self.MainQN, self.TargetQN,
                              self.LAMBDA, self.UPDATE_TARGET_FREQ, self.upd_target, self.GAMMA, self.BATCH_SIZE, self.MIN_EPSILON, self.analyze_memory)

    def reset(self):
        self.env.close()

        # Just to be sure that we don't have some others graph loaded
        tf.reset_default_graph()

        self.MainQN = QnetworkEager(30, len(self.ACTION_SPACE), DenseModel)
        self.TargetQN = QnetworkEager(30, len(self.ACTION_SPACE), DenseModel)

        self.upd_target = SoftUpdateWeightsEager(weights=self.MainQN, model=self.TargetQN, tau=0.08)

        self.BufferFillAgent = None
        self.agent = DQNAgent(self.ACTION_SPACE, self.stateDimension, self.buffer_memory, self.MainQN, self.TargetQN,
                              self.LAMBDA, self.UPDATE_TARGET_FREQ, self.upd_target, self.GAMMA, self.BATCH_SIZE, self.MIN_EPSILON, self.analyze_memory)

