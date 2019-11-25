from Agents import HrlAgent, RandomAgentOption, A2CSILOption
import gym
import tensorflow as tf
import os
from Environment import Environment
from Wrappers_Env import PositionGridenv_GE_MazeKeyDoor_v0
from Utils import ToolEpsilonDecayExploration, Preprocessing
from Models.A2CnetworksEager import *
from Utils import SaveResult
import gridenvs.examples


class variables():

    def __init__(self):

        tf.enable_eager_execution()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        self.seeds = range(2)
        self.RESULTS_FOLDER = 'TEST_HRL_SIL_PSEUDO_COUNT_6/'
        self.SAVE_RESULT = SaveResult(self.RESULTS_FOLDER)
        self.FILE_NAME = 'Key_Door_HRL_SIL_PSEUDO_COUNT'
        self.NUMBER_OF_EPOCHS = 4000

        self.PROBLEM = 'GE_MazeKeyDoor-v20'
        environment = gym.make(self.PROBLEM)

        self.ACTION_SPACE = [0, 1, 2, 3, 4]

        self.wrapper_params = {
            "stack_images_length": 1,
            "width": 10,
            "height": 10,
            "n_zones": 9
        }

        self.wrapper = PositionGridenv_GE_MazeKeyDoor_v0(environment, self.wrapper_params)

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

        self.shared_conv_layers = SharedConvLayers(0.05)

        self.number_of_stacked_frames = 1

        preprocessing = Preprocessing(84, 84, 3, self.number_of_stacked_frames, False)

        self.option_params = {
            "option": A2CSILOption,
            "h_size": 30,
            "action_space": self.ACTION_SPACE,
            "critic_network": CriticNetwork,
            "actor_network": ActorNetwork,
            "shared_representation": self.shared_conv_layers,
            "weight_mse": 0.5,
            "sil_weight_mse": 0.01,
            "weight_ce_exploration": 0.01,
            "learning_rate": 0.0001,
            "gamma": 0.99,
            "batch_size": 6,
            "sil_batch_size": 64,
            "imitation_buffer_size":64*100,
            "imitation_learning_steps": 4,

            "preprocessing": preprocessing
        }

        self.random_agent = RandomAgentOption(self.ACTION_SPACE)
        self.LAMBDA = 0.05
        self.MIN_EPSILON = 0
        self.PSEUDO_COUNT = 0.1

        # to know in how many episodes the epsilon will decay
        ToolEpsilonDecayExploration.epsilon_decay_end_steps(0, self.PSEUDO_COUNT)

        self.agent = HrlAgent(self.option_params, self.random_agent, self.PSEUDO_COUNT, self.LAMBDA, self.MIN_EPSILON, 0.6, -0.6, self.SAVE_RESULT)












