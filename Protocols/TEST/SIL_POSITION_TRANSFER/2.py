from Agents import HrlAgent, HrlAgent_heuristic_count_PR, RandomAgentOption, A2CSILOption, \
    HrlAgent_SubGoal_Plan_heuristic_count_PR, HrlAgent_heuristic_count_PR_v2, \
    HrlAgent_SubGoal_Plan_heuristic_count_PR_v2, A2CSILAgent
import gym
import tensorflow as tf
import os
from Environment import Environment
from Wrappers_Env import Flat_Position_observation_wrapper_key_door
from Models.A2CSILnetworksEager import *
from Utils import SaveResult
from Utils.HrlExplorationStrategies import get_epsilon_count_exploration
import gridenvs.examples

class variables():

    def __init__(self):

        tf.enable_eager_execution()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # to train on CPU

        self.seeds = range(5)
        self.RESULTS_FOLDER = (os.path.basename(os.path.dirname(os.path.dirname(__file__))) + '  -  TEST_SIL_POSITION_TRANSFER_1/')
        self.SAVE_RESULT = SaveResult(self.RESULTS_FOLDER)
        self.FILE_NAME = 'Position_Transfer_SIL'
        #self.NUMBER_OF_EPOCHS = 1000
        self.NUMBER_OF_STEPS = 200000

        self.multi_processing = False

        self.PROBLEM = 'GE_MazeTreasure16keyDoor2-v0'
        self.TEST_TRANSFER_PROBLEM = []

        environment = gym.make(self.PROBLEM)

        self.ACTION_SPACE = [0, 1, 2, 3, 4]

        self.wrapper_params = {
            "width": 16,
            "height": 16,
        }

        self.wrapper = Flat_Position_observation_wrapper_key_door(environment, self.wrapper_params)

        self.display_env = False

        if self.display_env:
            from Utils import ShowRenderHRL
            rendering = ShowRenderHRL
        else:
            rendering = False

        self.env = Environment(self.wrapper, preprocessing=False, rendering_custom_class=rendering, display_env=self.display_env)

    def reset(self):

        self.index_execution = 0

        self.env.close()

        # Just to be sure that we don't have some others graph loaded
        tf.reset_default_graph()

        preprocessing = None

        self.parameters = {
            "h_size": 64,
            "action_space": self.ACTION_SPACE,
            "critic_network": CriticNetwork,
            "actor_network": ActorNetwork,
            "shared_representation": None,
            "weight_mse": 0.5,
            "sil_weight_mse": 0.01,
            "weight_ce_exploration": 0.01,
            "learning_rate": 0.0007,
            "gamma": 0.95,
            "batch_size": 6,
            "sil_batch_size": 512,
            "imitation_buffer_size": 100000,
            "imitation_learning_steps": 4,
            "preprocessing": preprocessing,
        }

        self.a2cDNN_SIL = A2CSILEagerSeparate(self.parameters["h_size"], len(self.parameters["action_space"]), self.parameters["critic_network"],
                                          self.parameters["actor_network"], self.parameters["learning_rate"], self.parameters["weight_mse"],
                                          self.parameters["sil_weight_mse"], self.parameters["weight_ce_exploration"])

        self.agent = A2CSILAgent(self.parameters["action_space"], self.a2cDNN_SIL, self.parameters["gamma"], self.parameters["batch_size"],
                                 self.parameters["sil_batch_size"], self.parameters["imitation_buffer_size"], self.parameters["imitation_learning_steps"] )

        #self.agent.load("/home/lorenzo/Documenti/UPF/DeepRL/results/TEST  -  heuristic_count_TEST_SIL_POSITION_1/Wed_May_13_17:56:31_2020/seed_0/model")

    # def transfer_learning_test(self):
    #
    #     if self.env is not None:
    #         self.env.close()
    #
    #     self.number_of_stacked_frames = 1
    #     environment = gym.make(self.TEST_TRANSFER_PROBLEM[self.index_execution])
    #     self.wrapper = Flat_Position_observation_wrapper_key_door(environment, self.wrapper_params)
    #
    #     if self.display_env:
    #         from Utils import ShowRenderHRL
    #         rendering = ShowRenderHRL
    #     else:
    #         rendering = False
    #
    #     self.env = Environment(self.wrapper, preprocessing=False, rendering_custom_class=rendering, display_env=self.display_env)
    #
    #     self.TRANSFER_FILE_NAME = self.FILE_NAME + " - " + self.TEST_TRANSFER_PROBLEM[self.index_execution]
    #
    #     self.agent.set_name_file_2_save(self.TRANSFER_FILE_NAME)
    #     self.agent.reset()
    #
    #     self.index_execution += 1







