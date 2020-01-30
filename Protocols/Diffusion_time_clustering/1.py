# Position observation in gridworld


import os
from Environment import EnvironmentDiffusionTimeReset
from Agents import DiffusionTimeAgent
import gym
import gridenvs.examples
from Utils import SaveResult, Preprocessing
from Wrappers_Env import Gridenv_position, Origin_wrapper

class variables():

    def __init__(self):

        self.seeds = [1]
        self.PROBLEM = 'GE_MazeKeyDoorXL-v1'
        self.ACTION_SPACE = [1, 2, 3, 4]      # Warning removed 0 action, rest action
        self.RESULTS_FOLDER = 'DiffusionTimeClustering/'
        self.SAVE_RESULT = SaveResult(self.RESULTS_FOLDER)
        self.FILE_NAME = 'Clustering'

        self.NUMBER_OF_EPOCHS = 1000000

        self.preprocess = None

        environment = gym.make(self.PROBLEM)

        self.number_of_stacked_frames = 1

        self.wrapper_params = {
            "width": 30,
            "height": 30,
        }

        self.wrapper = Gridenv_position(environment, self.wrapper_params)

        display_env = False

        if display_env:
            from Utils import ShowRenderHRL
            rendering = ShowRenderHRL
        else:
            rendering = False

        self.env = EnvironmentDiffusionTimeReset(self.wrapper, preprocessing=False, rendering_custom_class=rendering)

    def reset(self):
        self.env.close()

        self.agent = DiffusionTimeAgent(self.ACTION_SPACE, self.env)