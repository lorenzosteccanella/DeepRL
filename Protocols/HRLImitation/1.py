from Agents import HrlImitationAgent, RandomAgentOption, A2COption
import gym
import tensorflow as tf
import os
from Environment import Environment
from Wrappers_Env import PositionGridenv_GE_MazeKeyDoor_v0
from Utils import ShowRenderHRL, ToolEpsilonDecayExploration
from Models.A2CnetworksEager import *
from Utils import SaveResult
import gridenvs.examples


class variables():

    def __init__(self):

        tf.enable_eager_execution()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        self.seeds = range(3)
        self.RESULTS_FOLDER = 'HRLImitation/'
        self.SAVE_RESULT = SaveResult(self.RESULTS_FOLDER)
        self.FILE_NAME = 'Key_Door_HRLImitation'
        self.NUMBER_OF_EPOCHS = 1500

        self.PROBLEM = 'GE_MazeKeyDoor-v0'
        environment = gym.make(self.PROBLEM)

        self.ACTION_SPACE = [0, 1, 2, 3, 4]

        wrapper_params = {
            "stack_images_length": 1,
        }

        self.wrapper = PositionGridenv_GE_MazeKeyDoor_v0(environment, wrapper_params)

        self.env = Environment(self.wrapper, preprocessing=False, rendering_custom_class=ShowRenderHRL)

        shared_conv_layers = SharedConvLayers()
        critic_network = CriticNetwork(30)
        actor_network = ActorNetwork(30, len(self.ACTION_SPACE))

        self.option_params = {
            "option": A2COption,
            "h_size": 30,
            "action_space": self.ACTION_SPACE,
            "critic_network": critic_network,
            "actor_network": actor_network,
            "shared_representation": shared_conv_layers,
            "weight_mse": 0.5,
            "weight_ce_exploration": 0.01,
            "learning_rate": 0.0001,
            "gamma": 0.99,
            "batch_size": 6
        }

        self.random_agent = RandomAgentOption(self.ACTION_SPACE)
        self.LAMBDA = 0.05
        self.MIN_EPSILON = 0.01

        # to know in how many episodes the epsilon will decay
        ToolEpsilonDecayExploration.epsilon_decay_end_steps(self.MIN_EPSILON, self.LAMBDA)

        self.agent = HrlImitationAgent(self.option_params, self.random_agent, self.LAMBDA, self.MIN_EPSILON)

    def reset(self):
        self.env.close()

        # Just to be sure that we don't have some others graph loaded
        tf.reset_default_graph()

        shared_conv_layers = SharedConvLayers()
        critic_network = CriticNetwork(30)
        actor_network = ActorNetwork(30, len(self.ACTION_SPACE))

        self.option_params = {
            "option": A2COption,
            "h_size": 30,
            "action_space": self.ACTION_SPACE,
            "critic_network": critic_network,
            "actor_network": actor_network,
            "shared_representation": shared_conv_layers,
            "weight_mse": 0.5,
            "weight_ce_exploration": 0.01,
            "learning_rate": 0.0001,
            "gamma": 0.99,
            "batch_size": 6
        }

        self.random_agent = RandomAgentOption(self.ACTION_SPACE)
        self.LAMBDA = 0.05
        self.MIN_EPSILON = 0.01

        # to know in how many episodes the epsilon will decay
        ToolEpsilonDecayExploration.epsilon_decay_end_steps(self.MIN_EPSILON, self.LAMBDA)

        self.agent = HrlImitationAgent(self.option_params, self.random_agent, self.LAMBDA, self.MIN_EPSILON)












