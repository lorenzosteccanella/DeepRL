from Models.A2CnetworksEager import *
import tensorflow as tf
import os
from Environment import Environment
from Agents import ExplorationEffortAgent_tabular
import gym
import gridenvs.examples
from Utils import SaveResult, Preprocessing, ExperienceReplay
from Wrappers_Env import Gridenv_position, Origin_wrapper

class variables():

    def __init__(self):

        tf.enable_eager_execution()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        self.seeds = [1]
        self.PROBLEM = 'GE_MazeKeyDoor-v10'
        self.ACTION_SPACE = [0, 1, 2, 3, 4]
        self.RESULTS_FOLDER = 'ExplorationEffort/'
        self.SAVE_RESULT = SaveResult(self.RESULTS_FOLDER)
        self.FILE_NAME = 'Key_Door_A2C'

        self.NUMBER_OF_EPOCHS = 1000000

        self.preprocess = None

        environment = gym.make(self.PROBLEM)

        self.number_of_stacked_frames = 1

        self.wrapper_params = {
            "width": 10,
            "height": 10,
        }

        self.wrapper = Gridenv_position(environment, self.wrapper_params)

        display_env = False

        if display_env:
            from Utils import ShowRenderHRL
            rendering = ShowRenderHRL
        else:
            rendering = False

        self.env = Environment(self.wrapper, preprocessing=False, rendering_custom_class=rendering)

    def reset(self):
        self.env.close()

        # Just to be sure that we don't have some others graph loaded
        tf.reset_default_graph()

        self.buffer = ExperienceReplay(10000)
        self.agent = ExplorationEffortAgent_tabular(self.ACTION_SPACE, self.buffer)