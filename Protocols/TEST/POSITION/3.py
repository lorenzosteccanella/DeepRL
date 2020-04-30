from Agents import GoalHrlAgent, HrlAgent_heuristic_count_PR, RandomAgentOption, A2COption, GoalHrlAgentSinglePlan, HrlAgent_heuristic_count_PR_SinglePlan
import gym
import tensorflow as tf
import os
from Environment import Environment
from Wrappers_Env import XL_Position_observation_wrapper
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
        self.RESULTS_FOLDER = (os.path.basename(os.path.dirname(os.path.dirname(__file__))) + '  -  heuristic_count_TEST_A2C_POSITION_XL/')
        self.SAVE_RESULT = SaveResult(self.RESULTS_FOLDER)
        self.FILE_NAME = 'Position_XL_A2C_HRL_E_GREEDY'
        self.NUMBER_OF_EPOCHS = 10000

        self.multi_processing = False

        self.PROBLEM = 'GE_MazeKeyDoorXL-v0'
        environment = gym.make(self.PROBLEM)

        self.ACTION_SPACE = [1, 2, 3, 4]

        self.wrapper_params = {
            "width": 30,
            "height": 30,
            "n_zones": 4
        }

        self.wrapper = XL_Position_observation_wrapper(environment, self.wrapper_params)

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

        # self.shared_conv_layers = SharedDenseLayers(0.05)
        # self.goal_net_start = False #SharedGoalModel(32, 1)
        # self.goal_net_goal = self.shared_conv_layers #SharedGoalModel(32, 1)
        # self.critic = CriticNetwork(32)
        # self.actor = ActorNetwork(32, len(self.ACTION_SPACE))

        preprocessing = None #Preprocessing(84, 84, 3, self.number_of_stacked_frames, False)

        self.option_params = {
            "option": A2COption,
            "h_size": 64,
            "action_space": self.ACTION_SPACE,
            "critic_network": CriticNetwork,
            "actor_network": ActorNetwork,
            "shared_representation": None,
            "weight_mse": 0.5,
            "weight_ce_exploration": 0.01,#.01,#.01,
            "learning_rate": 0.001,
            "learning_rate_reduction_obs": 1,  # WARNING
            "gamma": 0.95,
            "batch_size": 6,
            "steps_of_training": 1,
            "preprocessing": preprocessing,
            "sil_weight_mse": 0.01,
            "sil_batch_size": 64,
            "imitation_buffer_size": 1000,
            "imitation_learning_steps": 8
        }

        self.random_agent = RandomAgentOption(self.ACTION_SPACE)
        self.LAMBDA = 0.05
        self.MIN_EPSILON = 0
        self.PSEUDO_COUNT = 1000
        self.exploration_fn = get_epsilon_count_exploration

        # to know in how many episodes the epsilon will decay
        ToolEpsilonDecayExploration.epsilon_decay_end_steps(self.MIN_EPSILON, self.LAMBDA)

        self.single_option = self.option_params["option"](self.option_params)

        self.agent = HrlAgent_heuristic_count_PR_SinglePlan(self.option_params, self.random_agent, self.exploration_fn,
                                                     self.PSEUDO_COUNT, self.LAMBDA, self.MIN_EPSILON, 1.1, -1.1,
                                                     self.SAVE_RESULT, False, False, False) #self.single_option)









