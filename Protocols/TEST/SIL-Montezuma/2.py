from Agents import HrlAgent, HrlAgent_heuristic_count_PR, HrlAgent_nextV_PR, RandomAgentOption, A2CSILOption, WayPointsAgent_nextV_PR
import gym
import tensorflow as tf
import os
from Environment import Environment
from Wrappers_Env import Montezuma_Pixel_position_wrapper_only_1key
from Utils import ToolEpsilonDecayExploration, Preprocessing
from Models.A2CnetworksEager import *
from Utils import SaveResult
from Utils.HrlExplorationStrategies import get_best_action, get_epsilon_best_action, get_epsilon_exploration, get_epsilon_count_exploration
import gridenvs.examples

class variables():

    def __init__(self):

        tf.enable_eager_execution()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        self.seeds = range(1)
        self.RESULTS_FOLDER = (os.path.basename(os.path.dirname(os.path.dirname(__file__))) + '  -  heuristic_count_TEST_SIL_MONTEZUMA_2/')
        self.SAVE_RESULT = SaveResult(self.RESULTS_FOLDER)
        self.FILE_NAME = 'Montezuma_SIL_HRL_E_GREEDY'
        self.NUMBER_OF_EPOCHS = 1000

        self.multi_processing = False

        self.PROBLEM = 'MontezumaRevenge-ram-v0'
        environment = gym.make(self.PROBLEM)

        self.ACTION_SPACE = list(range(0, environment.action_space.n))

        self.wrapper_params = {
            "stack_images_length": 4,
            "n_zones": 40
        }

        self.wrapper = Montezuma_Pixel_position_wrapper_only_1key(environment, self.wrapper_params)

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

        # self.shared_conv_layers = SharedConvLayers(0.05)
        # self.critic = CriticNetwork(30)
        # self.actor = ActorNetwork(30, len(self.ACTION_SPACE))

        preprocessing = None

        self.option_params = {
            "option": A2CSILOption,
            "h_size": 32,
            "action_space": self.ACTION_SPACE,
            "critic_network": CriticNetwork,
            "actor_network": ActorNetwork,
            "shared_representation": SharedConvLayers, #self.shared_conv_layers,
            "weight_mse": 0.5,
            "sil_weight_mse": 0.05, #0.01,
            "weight_ce_exploration": 0.01,
            "learning_rate": 0.0001,
            "learning_rate_reduction_obs": 0.05,  # WARNING
            "gamma": 0.99,
            "batch_size": 6,
            "sil_batch_size": 64,
            "imitation_buffer_size": 64*100,
            "imitation_learning_steps": 4,
            "preprocessing": preprocessing
        }

        self.random_agent = RandomAgentOption(self.ACTION_SPACE)
        self.LAMBDA = 0.05
        self.MIN_EPSILON = 0.1
        self.PSEUDO_COUNT = 1000
        self.exploration_fn = get_epsilon_count_exploration

        # to know in how many episodes the epsilon will decay
        ToolEpsilonDecayExploration.epsilon_decay_end_steps(self.MIN_EPSILON, self.LAMBDA)

        self.agent = HrlAgent_heuristic_count_PR(self.option_params, self.random_agent, self.exploration_fn, self.PSEUDO_COUNT, self.LAMBDA, self.MIN_EPSILON, 1.1, -1.1, self.SAVE_RESULT)
        #self.agent.load("/home/lorenzo/Documenti/UPF/DeepRL/results/TEST  -  TEST_HRL_E_GREEDY_1/Tue_Mar_17_15:34:02_2020/seed_0/full_model.pkl")











