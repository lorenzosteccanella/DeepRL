from Agents import HrlAgent
import gym
import tensorflow as tf
import os
from Environment import Environment
from Wrappers_Env import WrapperObsHRL
from Utils import ShowRenderHRL
import gridenvs.examples


class variables():

    def __init__(self):

        tf.enable_eager_execution()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        self.seeds = [1]
        self.RESULTS_FOLDER = './results/'
        self.FILE_NAME = 'HRL.pkl'
        self.NUMBER_OF_EPOCHS = 200

        self.PROBLEM = 'GE_MazeKeyDoor-v0'
        environment = gym.make(self.PROBLEM)

        wrapper_params={
            "stack_images_length": 4,

            "NUMBER_ZONES_GRIDWORLD_X": 84,
            "NUMBER_ZONES_GRIDWORLD_Y": 84,
            "NUMBER_ZONES_MANAGER_X": 4,
            "NUMBER_ZONES_MANAGER_Y": 4,
            "GRAY_SCALE": False,
            "SSIM_PRECISION_FACTOR": 2
        }

        wrapper_params.update({"ZONE_SIZE_MANAGER_X": wrapper_params["NUMBER_ZONES_GRIDWORLD_X"] // wrapper_params["NUMBER_ZONES_MANAGER_X"],
                     "ZONE_SIZE_MANAGER_Y": wrapper_params["NUMBER_ZONES_GRIDWORLD_Y"] // wrapper_params["NUMBER_ZONES_MANAGER_Y"]})

        wrapper = WrapperObsHRL(environment, wrapper_params)

        self.env = Environment(wrapper, preprocessing=False, rendering_custom_class=ShowRenderHRL)

        self.ACTION_SPACE = [1, 2, 3, 4]

        self.agent = HrlAgent(self.ACTION_SPACE)








