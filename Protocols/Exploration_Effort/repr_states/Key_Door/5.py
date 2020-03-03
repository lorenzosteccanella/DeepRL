from Models.ExplorationEffortnetworksEager import *
import tensorflow as tf
import os
from Environment import Environment
from Agents import ExplorationEffortRepresentativeStates
import gym
import gridenvs.examples
from Utils import SaveResult, Preprocessing, ExperienceReplay

class variables():

    def __init__(self):

        tf.enable_eager_execution()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"

        self.seeds = [1]
        self.PROBLEM = 'GE_MazeKeyDoor-v10'
        self.ACTION_SPACE = [0, 1, 2, 3, 4]
        self.RESULTS_FOLDER = (os.path.basename(os.path.dirname(os.path.dirname(__file__))) + " - " + (os.path.basename(os.path.dirname(__file__))) + ' - 5/')
        self.SAVE_RESULT = SaveResult(self.RESULTS_FOLDER)
        self.FILE_NAME = 'Key_Door_A2C'

        self.distance_cluster= 5

        self.NUMBER_OF_EPOCHS = 100000

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

        self.buffer = ExperienceReplay(5000)

        learning_rate = 0.0001
        observation = SharedConvLayers()

        self.nn = EffortExplorationNN(len(self.ACTION_SPACE), learning_rate, observation, "./TF_models_weights/EffortExploration_weights")
        self.nn.load_weights()
        self.agent = ExplorationEffortRepresentativeStates(self.ACTION_SPACE, self.nn, self.distance_cluster, "KeyDoor")
