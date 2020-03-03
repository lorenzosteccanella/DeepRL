from Agents import HrlAgent, HrlAgent_nextV_PR, RandomAgentOption, A2COption
import Models.ExplorationEffortnetworksEager as ExplorationEffortnetworksEager
import Models.A2CnetworksEager as A2CnetworksEager
import gym
import tensorflow as tf
import os
from Environment import Environment
from Wrappers_Env import DS_wrapper
from Utils import ToolEpsilonDecayExploration, Preprocessing
from Utils import SaveResult
from Utils.HrlExplorationStrategies import get_best_action, get_epsilon_best_action, get_epsilon_exploration, get_epsilon_count_exploration
import gridenvs.examples

class variables():

    def __init__(self):

        tf.enable_eager_execution()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

        self.seeds = range(2)
        self.RESULTS_FOLDER = (os.path.basename(os.path.dirname(os.path.dirname(__file__))) + '  -  TEST_HRL_E_GREEDY_2/')
        self.SAVE_RESULT = SaveResult(self.RESULTS_FOLDER)
        self.FILE_NAME = 'MontezumaRevenge-v0'
        self.NUMBER_OF_EPOCHS = 1000

        self.PROBLEM = 'MontezumaRevenge-v0'
        environment = gym.make(self.PROBLEM)

        self.ACTION_SPACE = list(range(0, environment.action_space.n))

        learning_rate = 0.0001
        observation = ExplorationEffortnetworksEager.SharedConvLayers()

        self.distance_cluster= 0

        self.wrapper_params = {
            "stack_images_length": 1,
            "NUMBER_ZONES_MANAGER_X": 6*3,
            "NUMBER_ZONES_MANAGER_Y": 4*5,
            "MAX_PIXEL_VALUE": 8,
            "distance_cluster": self.distance_cluster
        }

        self.wrapper = DS_wrapper(environment, self.wrapper_params)

        display_env = True

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

        self.shared_conv_layers = A2CnetworksEager.SharedConvLayers(0.05)

        self.number_of_stacked_frames = 1

        preprocessing = Preprocessing(84, 84, 3, self.number_of_stacked_frames, False)

        self.option_params = {
            "option": A2COption,
            "h_size": 30,
            "action_space": self.ACTION_SPACE,
            "critic_network": A2CnetworksEager.CriticNetwork,
            "actor_network": A2CnetworksEager.ActorNetwork,
            "shared_representation": self.shared_conv_layers,
            "weight_mse": 0.5,
            "weight_ce_exploration": 0.01,
            "learning_rate": 0.0001,
            "gamma": 0.99,
            "batch_size": 6,
            "preprocessing": preprocessing
        }

        self.random_agent = RandomAgentOption(self.ACTION_SPACE)
        self.LAMBDA = 0.005
        self.MIN_EPSILON = 0
        self.PSEUDO_COUNT = 1000

        self.exploration_fn = get_epsilon_best_action

        # to know in how many episodes the epsilon will decay
        ToolEpsilonDecayExploration.epsilon_decay_end_steps(self.MIN_EPSILON, self.LAMBDA)

        self.agent = HrlAgent_nextV_PR(self.option_params, self.random_agent, self.exploration_fn, self.PSEUDO_COUNT, self.LAMBDA, self.MIN_EPSILON, 1.1, 0, self.SAVE_RESULT)












