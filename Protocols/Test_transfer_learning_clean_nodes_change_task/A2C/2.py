from Models.A2CnetworksEager import *
import tensorflow as tf
import os
from Agents import A2CAgent
import gym
import gridenvs.examples
from Utils import SaveResult, Preprocessing, LoadEnvironment

class variables():

    def __init__(self):

        self.index_execution = 0

        tf.enable_eager_execution()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        self.seeds = range(3)
        self.ACTION_SPACE = [0, 1, 2, 3, 4]
        self.GAMMA = 0.99
        self.LEARNING_RATE = 0.0001
        self.RESULTS_FOLDER = os.path.basename(os.path.dirname(os.path.dirname(__file__))) + 'TEST_A2C_grid18/'
        self.SAVE_RESULT = SaveResult(self.RESULTS_FOLDER)
        self.FILE_NAME = 'Key_Door_A2C'
        self.BATCH_SIZE = 6
        self.WEIGHT_MSE = 0.5
        self.WEIGHT_CE_EXPLORATION = 0.01

        self.NUMBER_OF_EPOCHS = 2000

        self.preprocess = None

        self.PROBLEM = 'GE_MazeKeyDoor18key1color1-v0'
        self.TEST_TRANSFER_PROBLEM = ['GE_MazeKeyDoor18pick_up_objects1color1-v0']

        self.load_environment = LoadEnvironment()
        self.env = None

    def reset(self):
        self.index_execution = 0

        if self.env is not None:
            self.env.close()

        self.number_of_stacked_frames = 1
        preprocessing = Preprocessing(84, 84, 3, self.number_of_stacked_frames)
        environment = gym.make(self.PROBLEM)
        self.env = self.load_environment.Load(environment, preprocessing, False)

        # Just to be sure that we don't have some others graph loaded
        tf.reset_default_graph()

        shared_conv_layers = SharedConvLayers()

        self.a2cDNN = A2CEagerSync(30, len(self.ACTION_SPACE), CriticNetwork, ActorNetwork,
                                   self.LEARNING_RATE, self.WEIGHT_MSE, self.WEIGHT_CE_EXPLORATION, shared_conv_layers)

        self.agent = A2CAgent(self.ACTION_SPACE, self.a2cDNN, self.GAMMA, self.BATCH_SIZE)

    def transfer_learning_test(self):

        if self.env is not None:
            self.env.close()

        self.number_of_stacked_frames = 1
        preprocessing = Preprocessing(84, 84, 3, self.number_of_stacked_frames)
        environment = gym.make(self.TEST_TRANSFER_PROBLEM[self.index_execution])
        self.env = self.load_environment.Load(environment, preprocessing, False)

        self.TRANSFER_FILE_NAME = self.FILE_NAME + " - " + self.TEST_TRANSFER_PROBLEM[self.index_execution]

        self.index_execution += 1
