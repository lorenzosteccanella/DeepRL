from Agents import HrlAgent, RandomAgentOption, A2COption
import gym
import tensorflow as tf
import os
from Wrappers_Env import PositionGridenv_GE_MazeKeyDoor_v0
from Utils import ToolEpsilonDecayExploration, Preprocessing
from Models.A2CnetworksEager import *
from Utils import SaveResult, LoadEnvironment
from Utils.HrlExplorationStrategies import get_best_action, get_epsilon_best_action, get_epsilon_exploration, get_epsilon_count_exploration
import gridenvs.examples

class variables():

    def __init__(self):

        self.index_execution = 0

        tf.enable_eager_execution()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        self.seeds = range(2)
        self.RESULTS_FOLDER = (os.path.basename(os.path.dirname(os.path.dirname(__file__))) + " - " + (os.path.basename(os.path.dirname(__file__))) + ' - 1/')
        self.SAVE_RESULT = SaveResult(self.RESULTS_FOLDER)
        self.FILE_NAME = 'Key_Door_HRL_E_GREEDY'
        self.NUMBER_OF_EPOCHS = 2000

        self.PROBLEM = 'GE_MazeKeyDoor18keyDoor1-v0'
        self.TEST_TRANSFER_PROBLEM = ['GE_MazeKeyDoor18keyDoor2-v0', 'GE_MazeKeyDoor18keyDoor3-v0']
        environment = gym.make(self.PROBLEM)

        self.ACTION_SPACE = [0, 1, 2, 3, 4]

        self.wrapper_params = {
            "stack_images_length": 1,
            "width": 10,
            "height": 10,
            "n_zones": 2
        }

        self.load_environment = LoadEnvironment()
        self.env = None

    def reset(self):
        self.index_execution = 0

        if self.env is not None:
            self.env.close()

        self.number_of_stacked_frames = 1
        environment = gym.make(self.PROBLEM)
        wrapper = PositionGridenv_GE_MazeKeyDoor_v0(environment, self.wrapper_params)
        self.env = self.load_environment.Load(wrapper, False, False)

        # Just to be sure that we don't have some others graph loaded
        tf.reset_default_graph()

        self.shared_conv_layers = SharedConvLayers(0.05)

        self.number_of_stacked_frames = 1

        preprocessing = Preprocessing(84, 84, 3, self.number_of_stacked_frames, False)

        self.option_params = {
            "option": A2COption,
            "h_size": 30,
            "action_space": self.ACTION_SPACE,
            "critic_network": CriticNetwork,
            "actor_network": ActorNetwork,
            "shared_representation": self.shared_conv_layers,
            "weight_mse": 0.5,
            "weight_ce_exploration": 0.01,
            "learning_rate": 0.0001,
            "gamma": 0.99,
            "batch_size": 6,
            "preprocessing": preprocessing
        }

        self.random_agent = RandomAgentOption(self.ACTION_SPACE)
        self.LAMBDA = 0.05
        self.MIN_EPSILON = 0
        self.PSEUDO_COUNT = 1000

        self.exploration_fn = get_epsilon_best_action

        # to know in how many episodes the epsilon will decay
        ToolEpsilonDecayExploration.epsilon_decay_end_steps(self.MIN_EPSILON, self.LAMBDA)

        self.agent = HrlAgent(self.option_params, self.random_agent, self.exploration_fn, self.PSEUDO_COUNT, self.LAMBDA, self.MIN_EPSILON, 1.1, -1.1, self.SAVE_RESULT)

        self.agent.set_RESET_EXPLORATION_WHEN_NEW_NODE(False)


    def transfer_learning_test(self):

        if self.env is not None:
            self.env.close()

        self.number_of_stacked_frames = 1
        environment = gym.make(self.TEST_TRANSFER_PROBLEM[self.index_execution])
        wrapper = PositionGridenv_GE_MazeKeyDoor_v0(environment, self.wrapper_params)
        self.env = self.load_environment.Load(wrapper, False, False)

        self.TRANSFER_FILE_NAME = self.FILE_NAME + " - " + self.TEST_TRANSFER_PROBLEM[self.index_execution]

        self.agent.set_name_file_2_save(self.TRANSFER_FILE_NAME)

        self.agent.graph.node_list.clear()

        self.agent.graph.edge_list.clear()

        self.agent.reset_exploration()
        self.agent.reset_pseudo_count_exploration()

        self.agent.reset_statistics()


        self.index_execution += 1














