from Agents import HrlAgent, HrlAgent_heuristic_count_PR, HrlAgent_nextV_PR, RandomAgentOption, DoubleDQNOption
import gym
import tensorflow as tf
import os
from Environment import Environment
from Wrappers_Env import Tot_reward_positionGridenv_GE_MazeKeyDoor_v0
from Utils import ToolEpsilonDecayExploration, Preprocessing
from Models.QnetworksEager import *
from Utils import SaveResult, ExperienceReplay, PrioritizedExperienceReplay, SoftUpdateWeightsEager
from Utils.HrlExplorationStrategies import get_best_action, get_epsilon_best_action, get_epsilon_exploration, get_epsilon_count_exploration
import gridenvs.examples

class variables():

    def __init__(self):

        tf.enable_eager_execution()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        self.seeds = range(1)
        self.RESULTS_FOLDER = (os.path.basename(os.path.dirname(os.path.dirname(__file__))) + '  -  TEST_HRL_E_GREEDY_1/')
        self.SAVE_RESULT = SaveResult(self.RESULTS_FOLDER)
        self.FILE_NAME = 'Key_Door_HRL_E_GREEDY'
        self.NUMBER_OF_EPOCHS = 1000

        self.PROBLEM = 'GE_MazeKeyDoor-v10'
        environment = gym.make(self.PROBLEM)

        self.ACTION_SPACE = [0, 1, 2, 3, 4]

        self.wrapper_params = {
            "stack_images_length": 1,
            "width": 10,
            "height": 10,
            "n_zones": 2
        }

        self.wrapper = Tot_reward_positionGridenv_GE_MazeKeyDoor_v0(environment, self.wrapper_params)

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

        self.main_shared_conv_layers = SharedConvLayers(0.05)
        self.target_shared_conv_layers = SharedConvLayers(0.05)

        self.number_of_stacked_frames = 1

        preprocessing = None #Preprocessing(84, 84, 3, self.number_of_stacked_frames, False)

        self.option_params = {
            "option": DoubleDQNOption,
            "h_size": 30,
            "action_space": self.ACTION_SPACE,
            "head_model": Dueling_head,
            "main_shared_representation": SharedConvLayers,  #self.main_shared_conv_layers,
            "target_shared_representation": SharedConvLayers,  #self.target_shared_conv_layers,
            "learning_rate": 0.001,
            "learning_rate_reduction_obs": 0.05,  # WARNING
            "buffer_type": PrioritizedExperienceReplay,
            "batch_size": 32,
            "buffer_size": 32*100,
            "update_target_freq": 1,
            "tau": 0.08,
            "update_fn": SoftUpdateWeightsEager,
            "gamma": 0.99,
            "lambda": 0.0005,
            "min_epsilon": 0,
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










