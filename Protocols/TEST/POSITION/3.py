from Agents import HrlAgent, HrlAgent_heuristic_count_PR, RandomAgentOption, A2COption
import gym
import tensorflow as tf
import os
from Environment import Environment
from Wrappers_Env import Montezuma_position_wrapper_only_1key
from Models.A2CnetworksEager import *
from Utils import SaveResult
from Utils.HrlExplorationStrategies import get_epsilon_count_exploration
import gridenvs.examples

class variables():

    def __init__(self):

        tf.enable_eager_execution()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        self.seeds = range(1)
        self.RESULTS_FOLDER = (os.path.basename(os.path.dirname(os.path.dirname(__file__))) + '  -  Montezuma_POSITION_1/')
        self.SAVE_RESULT = SaveResult(self.RESULTS_FOLDER)
        self.FILE_NAME = 'Position_Montezuma'
        self.NUMBER_OF_EPOCHS = 3000

        self.multi_processing = False

        self.PROBLEM = 'MontezumaRevenge-ram-v0'
        environment = gym.make(self.PROBLEM)

        self.ACTION_SPACE = list(range(0, environment.action_space.n))

        self.wrapper_params = {
            "stack_images_length": 1,
            "n_zones": 10
        }

        self.wrapper = Montezuma_position_wrapper_only_1key(environment, self.wrapper_params)

        display_env = True

        if display_env:
            from Utils import ShowRenderHRL
            rendering = ShowRenderHRL
        else:
            rendering = False

        self.env = Environment(self.wrapper, preprocessing=False, rendering_custom_class=rendering, display_env=display_env)

    def reset(self):
        self.env.close()

        # Just to be sure that we don't have some others graph loaded
        tf.reset_default_graph()

        preprocessing = None

        self.option_params = {
            "option": A2COption,
            "h_size": 64,
            "action_space": self.ACTION_SPACE,
            "critic_network": CriticNetwork,
            "actor_network": ActorNetwork,
            "shared_representation": None,
            "weight_mse": 0.5,
            "weight_ce_exploration": 0.01,#.01,#.01,
            "learning_rate": 0.0005,
            "learning_rate_reduction_obs": 1,  # WARNING
            "gamma": 0.95,
            "batch_size": 6,
            "steps_of_training": 1,
            "preprocessing": preprocessing,
        }

        self.random_agent = RandomAgentOption(self.ACTION_SPACE)
        self.LAMBDA = 0.05
        self.MIN_EPSILON = 0
        self.exploration_fn = get_epsilon_count_exploration

        self.agent = HrlAgent_heuristic_count_PR(self.option_params, self.random_agent, self.exploration_fn,
                                                 self.LAMBDA, self.MIN_EPSILON, 1.1, -0.1, self.SAVE_RESULT)










