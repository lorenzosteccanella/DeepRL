from Agents import HrlAgent, HrlAgent_heuristic_count_PR, RandomAgentOption, A2CSILOption, \
    HrlAgent_SubGoal_Plan_heuristic_count_PR, HrlAgent_heuristic_count_PR_v2, HrlAgent_SubGoal_Plan_heuristic_count_PR_v2
import gym
import tensorflow as tf
import os
from Environment import Environment
from Wrappers_Env import Position_observation_wrapper
from Models.PPOnetworksEager import *
from Utils import SaveResult
from Utils.HrlExplorationStrategies import get_epsilon_count_exploration
import gridenvs.examples

class variables():

    def __init__(self):

        tf.enable_eager_execution()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # to train on CPU

        self.seeds = range(1)
        self.RESULTS_FOLDER = (os.path.basename(os.path.dirname(os.path.dirname(__file__))) + '  -  heuristic_count_TEST_SIL_POSITION_1/')
        self.SAVE_RESULT = SaveResult(self.RESULTS_FOLDER)
        self.FILE_NAME = 'Position_SIL_HRL_E_GREEDY'
        self.NUMBER_OF_EPOCHS = 10000

        self.multi_processing = False

        self.PROBLEM = 'GE_MazeKeyDoor-v10'
        environment = gym.make(self.PROBLEM)

        self.ACTION_SPACE = [0, 1, 2, 3, 4]

        self.wrapper_params = {
            "width": 10,
            "height": 10,
            "n_zones": 2
        }

        self.wrapper = Position_observation_wrapper(environment, self.wrapper_params)

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
            "learning_rate": 0.001,
            "gamma": 0.95,
            "batch_size": 6,
            "sil_batch_size": 64,
            "imitation_buffer_size": 1000,
            "imitation_learning_steps": 8,
            "preprocessing": preprocessing,
        }

        self.random_agent = RandomAgentOption(self.ACTION_SPACE)
        self.LAMBDA = 0.001
        self.MIN_EPSILON = 0
        self.exploration_fn = get_epsilon_count_exploration

        self.agent = HrlAgent_heuristic_count_PR_v2(self.option_params, self.random_agent, self.exploration_fn,
                                                              self.LAMBDA, self.MIN_EPSILON, 0.8, - 1, self.SAVE_RESULT)

        #self.agent.load("/home/lorenzo/Documenti/UPF/DeepRL/results/TEST  -  heuristic_count_TEST_SIL_POSITION_1/Wed_May_13_17:56:31_2020/seed_0/model")










