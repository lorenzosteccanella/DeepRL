from Agents import GoalHrlAgent, GoalHrlAgent_heuristic_count_PR, GoalHrlAgent_nextV_PR, RandomAgentOption, GoalA2COption
import gym
import tensorflow as tf
import os
from Environment import ParallelEnvironment
from Wrappers_Env import Tot_reward_positionGridenv_GE_MazeKeyDoor_v0
from Utils import ToolEpsilonDecayExploration, Preprocessing
from Models.GoalA2CnetworksEager import *
from Utils import SaveResult, Graph
from Utils.HrlExplorationStrategies import get_best_action, get_epsilon_best_action, get_epsilon_exploration, get_epsilon_count_exploration
import gridenvs.examples

import multiprocessing as mp

class variables():

    def __init__(self):

        tf.enable_eager_execution()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        self.seeds = range(1)
        self.RESULTS_FOLDER = (os.path.basename(os.path.dirname(os.path.dirname(__file__))) + '  -  heuristic_count_TEST_PARALLEL_GOAL_A2C_HRL_E_GREEDY_1/')
        self.SAVE_RESULT = SaveResult(self.RESULTS_FOLDER)
        self.FILE_NAME = 'Key_Door_PARALLEL_GOAL_A2C_HRL_E_GREEDY'
        self.NUMBER_OF_EPOCHS = 3000

        self.multi_processing = True
        self.num_workers = mp.cpu_count()

        self.PROBLEM = 'GE_MazeKeyDoor-v10'
        environment = gym.make(self.PROBLEM)

        self.ACTION_SPACE = [0, 1, 2, 3, 4]

        self.wrapper_params = {
            "stack_images_length": 1,
            "width": 10,
            "height": 10,
            "n_zones": 2
        }

        self.wrapper = Tot_reward_positionGridenv_GE_MazeKeyDoor_v0

        display_env = False

        if display_env:
            from Utils import ShowRenderHRL
            rendering = ShowRenderHRL
        else:
            rendering = False

        self.env = ParallelEnvironment(preprocessing=False, rendering_custom_class=rendering)

    def reset(self):
        self.env.close()

        # Just to be sure that we don't have some others graph loaded
        tf.reset_default_graph()

        self.shared_conv_layers = SharedConvLayers(1)
        self.goal_net_start = False #SharedGoalModel(32, 1)
        self.goal_net_goal = self.shared_conv_layers #SharedGoalModel(32, 1)
        self.critic = CriticNetwork(32)
        self.actor = ActorNetwork(32, len(self.ACTION_SPACE))

        preprocessing = None #Preprocessing(84, 84, 3, self.number_of_stacked_frames, False)

        self.option_params = {
            "option": GoalA2COption,
            "h_size": 32,
            "action_space": self.ACTION_SPACE,
            "critic_network": self.critic,
            "actor_network": self.actor,
            "shared_representation": self.shared_conv_layers,
            "shared_goal_representation_start": self.goal_net_start,
            "shared_goal_representation_goal": self.goal_net_goal,
            "weight_mse": 0.5,
            "weight_ce_exploration": 0.01,
            "learning_rate": 0.0001,
            "learning_rate_reduction_obs": 0.05,  # WARNING
            "gamma": 0.99,
            "batch_size": 6,
            "preprocessing": preprocessing
        }

        self.random_agent = RandomAgentOption(self.ACTION_SPACE)
        self.LAMBDA = 0.05
        self.MIN_EPSILON = 0.1
        self.PSEUDO_COUNT = 1000
        self.exploration_fn = get_epsilon_count_exploration

        # to know in how many episodes the epsilon will decay
        ToolEpsilonDecayExploration.epsilon_decay_end_steps(self.MIN_EPSILON, self.LAMBDA)

        self.single_option = self.option_params["option"](self.option_params)

        graphs = []
        graphs_param = (list(), list(), {}, {}, {}, {}, {})
        for i in range(self.num_workers):
            if i == 0:
                graphs.append(Graph(*graphs_param, self.SAVE_RESULT))
            else:
                graphs.append(Graph(*graphs_param, False))

        self.option_list = list()
        self.as_m2s_m = {}

        self.agent = []
        for i in range(self.num_workers):
            if i == 0:
                self.agent.append(GoalHrlAgent_heuristic_count_PR(self.option_params, self.random_agent, self.exploration_fn,
                                                              self.PSEUDO_COUNT, self.LAMBDA, self.MIN_EPSILON, 1.1,
                                                              -1.1, self.SAVE_RESULT, graphs[i], self.option_list,
                                                              self.single_option, self.as_m2s_m))
                #self.agent.load("/home/lorenzo/Documenti/UPF/DeepRL/results/TEST  -  TEST_HRL_E_GREEDY_1/Tue_Mar_17_15:34:02_2020/seed_0/full_model.pkl")

            else:
                self.agent.append(GoalHrlAgent_heuristic_count_PR(self.option_params, self.random_agent, self.exploration_fn,
                                                              self.PSEUDO_COUNT, self.LAMBDA, self.MIN_EPSILON, 1.1,
                                                              -1.1, False, graphs[i], self.option_list,
                                                              self.single_option, self.as_m2s_m))










