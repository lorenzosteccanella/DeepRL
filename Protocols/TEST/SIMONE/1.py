from Agents import HrlAgent, HrlAgent_heuristic_count_PR, RandomAgentOption, A2CSILOption, \
    HrlAgent_SubGoal_Plan_heuristic_count_PR, HrlAgent_heuristic_count_PR_v2, HrlAgent_SubGoal_Plan_heuristic_count_PR_v2
import gym
import tensorflow as tf
import os
from Environment import Environment
from stable_baselines.common.atari_wrappers import *
from Wrappers_Env import Montezuma_position_wrapper_only_1key
from Models.PPOnetworksEager import *
from Utils import SaveResult
from Utils.HrlExplorationStrategies import get_epsilon_count_exploration
import gridenvs.examples

class variables():

    def __init__(self):

        tf.enable_eager_execution()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        self.seeds = range(1)
        self.RESULTS_FOLDER = (os.path.basename(os.path.dirname(os.path.dirname(__file__))) + '  -  Montezuma_SIL_POSITION_1/')
        self.SAVE_RESULT = SaveResult(self.RESULTS_FOLDER)
        self.FILE_NAME = 'Position_SIL_Montezuma'
        self.NUMBER_OF_EPOCHS = 4000

        self.multi_processing = False

        self.PROBLEM = 'MontezumaRevenge-ram-v0'
        environment = gym.make(self.PROBLEM)
        environment = NoopResetEnv(environment)
        environment = FireResetEnv(environment)

        self.ACTION_SPACE = [0, 1, 2, 3, 4, 5, 14, 15]

        self.wrapper_params = {
            "stack_images_length": 1,
            "n_zones": 40
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
            "option": A2CSILOption,
            "h_size": 128,
            "action_space": self.ACTION_SPACE,
            "critic_network": CriticNetwork,
            "actor_network": ActorNetwork,
            "shared_representation": None,
            "weight_mse": 0.5,
            "sil_weight_mse": 0.01,
            "weight_ce_exploration": 0.01,
            "learning_rate": 0.0005,
            "gamma": 0.95,
            "batch_size": 6,
            "sil_batch_size": 64,
            "imitation_buffer_size": 1000,
            "imitation_learning_steps": 8,
            "preprocessing": preprocessing,
        }

        self.random_agent = RandomAgentOption(self.ACTION_SPACE)
        self.LAMBDA = 0.05
        self.MIN_EPSILON = 0
        self.exploration_fn = get_epsilon_count_exploration

        self.agent = HrlAgent_heuristic_count_PR_v2(self.option_params, self.random_agent, self.exploration_fn,
                                                              self.LAMBDA, self.MIN_EPSILON, 0.8, -0.1, self.SAVE_RESULT)

        self.agent.load("./Models_saved/Wed_May_13_18:15:16_2020/seed_0/model")










