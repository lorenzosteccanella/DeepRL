from Agents import HrlAgent
import gym
import tensorflow as tf
import os
from Environment import Environment
from Wrappers_Env import PositionGridenv_GE_MazeKeyDoor_v0
from Utils import ShowRenderHRL
from Models.A2CnetworksEager import *
import gridenvs.examples


class variables():

    def __init__(self):

        tf.enable_eager_execution()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        self.seeds = [1]
        self.RESULTS_FOLDER = './results/'
        self.FILE_NAME = 'HRL.pkl'
        self.NUMBER_OF_EPOCHS = 5000
        self.LAMBDA = 0.01
        self.MIN_EPSILON = 0.01

        self.PROBLEM = 'GE_MazeKeyDoor-v0'
        environment = gym.make(self.PROBLEM)

        self.ACTION_SPACE = [1, 2, 3, 4]

        wrapper_params = {
            "stack_images_length": 1,
        }

        wrapper = PositionGridenv_GE_MazeKeyDoor_v0(environment, wrapper_params)

        self.env = Environment(wrapper, preprocessing=False, rendering_custom_class=ShowRenderHRL)

        shared_conv_layers = SharedConvLayers()

        self.option_params = {
            "h_size": 30,
            "action_space": self.ACTION_SPACE,
            "critic_network": CriticNetwork,
            "actor_network": ActorNetwork,
            "shared_representation": shared_conv_layers,
            "weight_mse": 0.5,
            "weight_ce_exploration": 0.01,
            "gamma": 0.99,
            "batch_size": 1
        }

        self.agent = HrlAgent(self.option_params, None, self.LAMBDA, self.MIN_EPSILON)








