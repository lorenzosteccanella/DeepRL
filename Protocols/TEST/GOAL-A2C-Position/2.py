from Agents import PPOAgent, A2CAgent, A2CSILAgent, HrlAgent, GoalHrlAgent_heuristic_count_PR, GoalHrlAgent_nextV_PR, RandomAgentOption, A2COption
import gym
import tensorflow as tf
import os
from Environment import Environment
from Wrappers_Env import Gridenv_position
from Utils import ToolEpsilonDecayExploration, Preprocessing
from Models.A2CnetworksEager import *
from Utils import SaveResult
import gridenvs.examples
from Models.CommonA2C import *

class variables():

    def __init__(self):

        tf.enable_eager_execution()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        self.seeds = range(1)
        self.RESULTS_FOLDER = (os.path.basename(os.path.dirname(os.path.dirname(__file__))) + '  -  heuristic_count_TEST_GOAL_A2C_POSITION_1/')
        self.SAVE_RESULT = SaveResult(self.RESULTS_FOLDER)
        self.FILE_NAME = 'Position_GOAL_A2C_HRL_E_GREEDY'
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

        self.wrapper = Gridenv_position(environment, self.wrapper_params)

        display_env = False

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

        # self.shared_conv_layers = SharedDenseLayers(1)
        # self.goal_net_start = False #SharedGoalModel(32, 1)
        # self.goal_net_goal = self.shared_conv_layers #SharedGoalModel(32, 1)
        # self.critic = CriticNetwork(32)
        # self.actor = ActorNetwork(32, len(self.ACTION_SPACE))

        preprocessing = None #Preprocessing(84, 84, 3, self.number_of_stacked_frames, False)

        parameters = {
            "h_size": 128,
            "action_space": self.ACTION_SPACE,
            "critic_network": CriticNetwork,
            "actor_network": ActorNetwork,
            "target_actor_network": ActorNetwork,
            "shared_representation": None,
            "weight_mse": 0.5,
            "weight_ce_exploration": 0.01,
            "learning_rate": 0.0005,
            "learning_rate_reduction_obs": 1,  # WARNING
            "tau": 0.90,
            "e_clip": 0.2,
            "gamma": 0.99,
            "batch_size": 6,
            "steps_of_training": 1,
            "preprocessing": preprocessing
        }

        self.a2cDNN = A2CEagerSync(parameters["h_size"], len(parameters["action_space"]), parameters["critic_network"],
                                   parameters["actor_network"], parameters["learning_rate"], parameters["weight_mse"],
                                   parameters["weight_ce_exploration"], parameters["shared_representation"], parameters["learning_rate_reduction_obs"])

        self.agent = A2CAgent(parameters["action_space"], self.a2cDNN, parameters["gamma"], parameters["batch_size"])











