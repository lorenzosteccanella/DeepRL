import tensorflow as tf
from tensorflow import keras
import numpy as np
from Losses.Losses import Losses
#import tensorflow.contrib.slim as slim
import inspect
from .CommonA2C import *

class SiameseActorCriticNetwork(keras.Model):

    def __init__(self, critic_model, actor_model, shared_observation_model=None):
        super(SiameseActorCriticNetwork, self).__init__(name="SiameseActorCriticNetwork")
        self.shared_observation_model = shared_observation_model
        self.critic_model = critic_model
        self.actor_model = actor_model
        self.L1_layer = keras.layers.Lambda(lambda tensors: keras.backend.abs(tensors[0] - tensors[1]))
        self.L2_layer = keras.layers.Lambda(lambda tensors: keras.backend.pow(tensors[0] - tensors[1], 2))

    def call(self, x1, x2, x3):

        if self.shared_observation_model is not None:
            # I'm using only the goal state for now!!
            obs1 = self.shared_observation_model(x1)[0] # Just the dense output
            obs3 = self.shared_observation_model(x3)[0]

            obs = self.L1_layer([obs1, obs3])

            actor = self.actor_model(obs)

            critic = self.critic_model(obs)

        else:

            actor = self.actor_model(x1, x2, x3)

            critic = self.actor_model(x1, x2, x3)

        return actor, critic


class GoalCriticNetwork(keras.Model):
    def __init__(self, h_size):
        super(GoalCriticNetwork, self).__init__(name="GoalCriticNetwork")
        self.dense1 = keras.layers.Dense(h_size, activation='relu', kernel_initializer='glorot_uniform')
        self.dense2 = keras.layers.Dense(h_size, activation='relu', kernel_initializer='glorot_uniform')
        self.out = keras.layers.Dense(1, activation='linear', kernel_initializer='glorot_uniform')
        self.L1_layer = keras.layers.Lambda(lambda tensors: keras.backend.abs(tensors[0] - tensors[1]))
        self.L2_layer = keras.layers.Lambda(lambda tensors: keras.backend.pow(tensors[0] - tensors[1], 2))

    def call(self, x1, x2, x3):
        x1 = self.dense1(x1)
        x3 = self.dense1(x3) # Goal observation
        x = tf.concat([x1, x3], axis=1)
        x = self.dense2(x)
        x = self.out(x)
        return x

class GoalActorNetwork(keras.Model):
    def __init__(self, h_size, n_actions):
        super(GoalActorNetwork, self).__init__(name="GoalActorNetwork")
        self.n_actions = n_actions
        self.dense1 = keras.layers.Dense(h_size, activation='relu', kernel_initializer='glorot_uniform')
        self.dense2 = keras.layers.Dense(h_size, activation='relu', kernel_initializer='glorot_uniform')
        self.dense3 = keras.layers.Dense(h_size, activation='relu', kernel_initializer='glorot_uniform')
        self.out = keras.layers.Dense(n_actions, activation=None, kernel_initializer='glorot_uniform')
        self.L1_layer = keras.layers.Lambda(lambda tensors: keras.backend.abs(tensors[0] - tensors[1]))
        self.L2_layer = keras.layers.Lambda(lambda tensors: keras.backend.pow(tensors[0] - tensors[1], 2))
        self.goal_encoding = keras.layers.Dense(h_size, activation="linear", use_bias = False, kernel_initializer='random_uniform')
        self.single_action_encoding = []
        for i in range(self.n_actions):
            self.single_action_encoding.append(keras.layers.Dense(h_size, activation="relu", kernel_initializer='random_uniform'))


    def call(self, x1, x2, x3):
        x1 = self.dense1(x1)
        x3 = self.dense1(x3) # Goal observation
        x = tf.concat([x1, x3], axis=1)#self.L1_layer([x1, x3])
        # action_encoding = []
        # for i in range(self.n_actions):
        #     action_encoding.append(tf.expand_dims(self.single_action_encoding[i](x), 1))
        #
        # action_encoding = tf.concat(action_encoding, axis=1)
        # goal_encoding = tf.expand_dims(self.goal_encoding(tf.stop_gradient(x3)), 1)
        # # dot product
        # mul_x = tf.multiply( action_encoding, goal_encoding)
        # x = tf.reduce_sum(mul_x, axis = 2)
        x = self.dense2(x)
        x = self.out(x)
        return x