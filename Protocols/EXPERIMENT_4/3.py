from Agents import HrlAgent, HrlAgent_heuristic_count_PR, RandomAgentOption, A2CSILOption, \
    HrlAgent_SubGoal_Plan_heuristic_count_PR, HrlAgent_heuristic_count_PR_v2, \
    HrlAgent_SubGoal_Plan_heuristic_count_PR_v2, A2CSILAgent, A2CSILwPHCAgent
import gym
import tensorflow as tf
import os
from Environment import Environment
from Wrappers_Env import Flat_Montezuma_position_wrapper_only_1key
from Models.A2CSILnetworksEager import *
from Utils import SaveResult
from Utils.HrlExplorationStrategies import get_epsilon_count_exploration
import gridenvs.examples

class variables():

    def __init__(self):

        #tf.enable_eager_execution()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # to train on CPU

        tf.config.optimizer.set_jit(True)

        self.seeds = [0]
        self.MAX_R = 100
        self.RESULTS_FOLDER = (os.path.basename(os.path.dirname(__file__)) + '  -  SIL-HC-totr' + str(self.MAX_R) + '- sf4 - v1/')
        self.SAVE_RESULT = SaveResult(self.RESULTS_FOLDER)
        self.FILE_NAME = 'SIL-HC-totr' + str(self.MAX_R)
        #self.NUMBER_OF_EPOCHS = 1000
        self.NUMBER_OF_STEPS = 400000

        self.multi_processing = False

        self.PROBLEM = 'MontezumaRevenge-ramNoFrameskip-v4'
        self.TEST_TRANSFER_PROBLEM = []

        environment = gym.make(self.PROBLEM)

        self.ACTION_SPACE = [0, 1, 2, 3, 4, 5, 14, 15]

        self.wrapper_params = {
            "stack_images_length": 4
        }

        self.wrapper = Flat_Montezuma_position_wrapper_only_1key(environment, self.wrapper_params)

        self.display_env = False

        rendering = False

        self.env = Environment(self.wrapper, preprocessing=False, rendering_custom_class=rendering, display_env=self.display_env)

    def reset(self):

        self.index_execution = 0

        self.env.close()

        # Just to be sure that we don't have some others graph loaded
        #tf.reset_default_graph()

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

        self.agent = A2CSILwPHCAgent(self.parameters["action_space"], self.a2cDNN_SIL, self.parameters["gamma"], self.parameters["batch_size"],
                                 self.parameters["sil_batch_size"], self.parameters["imitation_buffer_size"], self.parameters["imitation_learning_steps"] )

        #self.agent.load("/home/lorenzo/Documenti/UPF/DeepRL/results/TEST  -  heuristic_count_TEST_SIL_POSITION_1/Wed_May_13_17:56:31_2020/seed_0/model")








