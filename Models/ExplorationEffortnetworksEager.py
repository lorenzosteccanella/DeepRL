import tensorflow as tf
from tensorflow import keras
import numpy as np
from Losses.Losses import Losses
import tensorflow.contrib.slim as slim
import inspect

class SharedConvLayers(keras.Model):
    def __init__(self, learning_rate_observation_adjust=1):
        super(SharedConvLayers, self).__init__(name="SharedConvLayers")
        self.conv1 = keras.layers.Conv2D(16, 8, (4, 4), padding='VALID', activation='elu', kernel_initializer='he_normal', )
        self.conv2 = keras.layers.Conv2D(16, 4, (2, 2), padding='VALID', activation='elu', kernel_initializer='he_normal')
        self.conv3 = keras.layers.Conv2D(16, 3, (1, 1), padding='VALID', activation='elu', kernel_initializer='he_normal')
        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(256, activation='elu', kernel_initializer='he_normal')
        self.normalization_layer = keras.layers.LayerNormalization()
        self.learning_rate_adjust = learning_rate_observation_adjust

    def call(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        denseOut = self.normalization_layer(x)
        x = self.learning_rate_adjust * denseOut + (1-self.learning_rate_adjust) * tf.stop_gradient(denseOut)  # U have to test this!!!

        return [x, denseOut]


class SharedDenseLayers(keras.Model):
    def __init__(self, h_size=256):
        super(SharedDenseLayers, self).__init__(name="SharedDenseLayers")
        self.dense1 = keras.layers.Dense(h_size, activation='elu', kernel_initializer='he_normal')
        self.dense2 = keras.layers.Dense(h_size, activation='elu', kernel_initializer='he_normal')
        self.normalization_layer = keras.layers.LayerNormalization()

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.normalization_layer(x)

        return [x, x]


class SiameseNetwork(keras.Model):

    def __init__(self, obs_model, n_actions):
        super(SiameseNetwork, self).__init__(name="SiameseNetwork")
        self.shared_observation_model = obs_model
        self.dense1 = keras.layers.Dense(256, activation='elu', kernel_initializer='he_normal')
        self.dense2 = keras.layers.Dense(n_actions, activation='linear')

    def call(self, x1, x2):

        obs1 = self.shared_observation_model(x1)[0] # Just the dense output
        obs2 = self.shared_observation_model(x2)[0] # Just the dense output

        x = keras.layers.concatenate([obs1, obs2], axis=-1)
        x = self.dense1(x)
        x = self.dense2(x)


        return x


class EffortExplorationNN:

    def __init__(self, n_actions, learning_rate, shared_observation_model):

        self.shared_observation_model = shared_observation_model

        self.model_exploration_effort = SiameseNetwork(self.shared_observation_model, n_actions)

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        self.global_step = tf.Variable(0)

    def prediction_distance(self, s1, s2):

        s1 = np.array(s1, dtype=np.float32)

        s1 = tf.convert_to_tensor(s1)

        s2 = np.array(s2, dtype=np.float32)

        s2 = tf.convert_to_tensor(s2)

        return self.model_exploration_effort(s1, s2).numpy()

    def grad(self, model_exploration_effort, x1, x2, y):

        with tf.GradientTape() as tape:
            out = model_exploration_effort(x1, x2)
            loss = Losses.mse_loss(out, y)
            loss_value = loss

        return loss_value, tape.gradient(loss_value, self.model_exploration_effort.trainable_variables)


    def train(self, s1, s2, y, max_grad_norm=5):

        s1 = np.array(s1, dtype=np.float32)

        s1 = tf.convert_to_tensor(s1)

        s2 = np.array(s2, dtype=np.float32)

        s2 = tf.convert_to_tensor(s2)

        loss_value, grads = self.grad(self.model_exploration_effort, s1, s2, y)

        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        self.optimizer.apply_gradients(zip(grads, self.model_exploration_effort.trainable_variables), self.global_step)

        return [None, None]


    def save_weights(self):
        self.model_actor_critic.save_weights("/home/lorenzo/Documenti/UPF/DeepRL/TF_models_weights/EffortExploration_weights")

    def load_weights(self):
        self.model_actor_critic.load_weights("/home/lorenzo/Documenti/UPF/DeepRL/TF_models_weights/EffortExploration_weights")