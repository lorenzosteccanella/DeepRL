import tensorflow as tf
from tensorflow import keras
import numpy as np
from Losses.Losses import Losses
# import tensorflow.contrib.slim as slim
import inspect


class SharedConvLayers(keras.Model):
    def __init__(self, learning_rate_observation_adjust=1):
        super(SharedConvLayers, self).__init__(name="SharedConvLayers")
        self.conv1 = keras.layers.Conv2D(32, 8, (4, 4), padding='VALID', activation='relu',
                                         kernel_initializer='he_uniform')
        self.conv2 = keras.layers.Conv2D(64, 4, (2, 2), padding='VALID', activation='relu',
                                         kernel_initializer='he_uniform')
        self.conv3 = keras.layers.Conv2D(64, 3, (1, 1), padding='VALID', activation='relu',
                                         kernel_initializer='he_uniform')
        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(256, activation='elu', kernel_initializer='he_uniform')
        self.normalization_layer = keras.layers.LayerNormalization()
        self.learning_rate_adjust = learning_rate_observation_adjust

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        denseOut = self.dense(x)
        x = self.learning_rate_adjust * denseOut + (1 - self.learning_rate_adjust) * tf.stop_gradient(denseOut)

        return [x, denseOut]

    def prediction_h(self, s):
        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        return self.call(s)[1].numpy()


class SharedDenseLayers(keras.Model):
    def __init__(self, h_size=256, learning_rate_observation_adjust=1):
        super(SharedDenseLayers, self).__init__(name="SharedDenseLayers")
        self.dense1 = keras.layers.Dense(h_size, activation='relu', kernel_initializer='glorot_uniform')
        self.dense2 = keras.layers.Dense(h_size, activation='relu', kernel_initializer='glorot_uniform')
        self.learning_rate_adjust = learning_rate_observation_adjust

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.learning_rate_adjust * x + (1 - self.learning_rate_adjust) * tf.stop_gradient(
            x)  # U have to test this!!!

        return [x, x]


class CriticNetwork(keras.Model):
    def __init__(self, h_size):
        super(CriticNetwork, self).__init__(name="CriticNetwork")
        self.dense1 = keras.layers.Dense(h_size, activation='relu', kernel_initializer='glorot_uniform')
        self.dense2 = keras.layers.Dense(h_size, activation='relu', kernel_initializer='glorot_uniform')
        self.out = keras.layers.Dense(1, activation='linear', kernel_initializer='glorot_uniform')

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.out(x)
        return x


class ActorNetwork(keras.Model):
    def __init__(self, h_size, n_actions):
        super(ActorNetwork, self).__init__(name="ActorNetwork")
        self.dense1 = keras.layers.Dense(h_size, activation='relu', kernel_initializer='glorot_uniform')
        self.dense2 = keras.layers.Dense(h_size, activation='relu', kernel_initializer='glorot_uniform')
        self.out = keras.layers.Dense(n_actions, activation=None, kernel_initializer='random_uniform')

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.out(x)

        return x


class ActorCriticNetwork(keras.Model):

    def __init__(self, critic_model, actor_model, shared_observation_model=None):
        super(ActorCriticNetwork, self).__init__(name="ActorCriticNetwork")
        self.shared_observation_model = shared_observation_model
        self.critic_model = critic_model
        self.actor_model = actor_model

    def call(self, x):

        if self.shared_observation_model is not None:

            obs = self.shared_observation_model(x)[0]  # Just the dense output
        else:
            obs = x

        actor = self.actor_model(obs)

        critic = self.critic_model(obs)

        return actor, critic, None


class NoisyActorNetwork(keras.Model):
    def __init__(self, h_size, n_actions):
        super(NoisyActorNetwork, self).__init__(name="NoisyActorNetwork")
        self.dense1 = keras.layers.Dense(h_size, activation='relu', kernel_initializer='glorot_uniform')
        self.dense2 = keras.layers.Dense(h_size, activation='relu', kernel_initializer='glorot_uniform')
        self.out = NoisyDense(n_actions, h_size)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.out(x)

        return x


# Factorized Gaussian Noise Layer
# Reference from https://github.com/Kaixhin/Rainbow/blob/master/model.py
class NoisyDense(keras.layers.Layer):
    def __init__(self, units, input_dim, std_init=0.5):
        super().__init__()
        self.units = units
        self.std_init = std_init
        self.reset_noise(input_dim)
        mu_range = 1 / np.sqrt(input_dim)
        mu_initializer = tf.random_uniform_initializer(-mu_range, mu_range)
        sigma_initializer = tf.constant_initializer(self.std_init / np.sqrt(self.units))

        self.weight_mu = tf.Variable(initial_value=mu_initializer(shape=(input_dim, units), dtype='float32'),
                                     trainable=True)

        self.weight_sigma = tf.Variable(initial_value=sigma_initializer(shape=(input_dim, units), dtype='float32'),
                                        trainable=True)

        self.bias_mu = tf.Variable(initial_value=mu_initializer(shape=(units,), dtype='float32'),
                                     trainable=True)

        self.bias_sigma = tf.Variable(initial_value=sigma_initializer(shape=(units,), dtype='float32'),
                                        trainable=True)

    def call(self, inputs):
        # output = tf.tensordot(inputs, self.kernel, 1)
        # tf.nn.bias_add(output, self.bias)
        # return output
        self.kernel = self.weight_mu + self.weight_sigma * self.weights_eps
        self.bias = self.bias_mu + self.bias_sigma * self.bias_eps
        return tf.matmul(inputs, self.kernel) + self.bias

    def _scale_noise(self, dim):
        noise = tf.random.normal([dim])
        return tf.sign(noise) * tf.sqrt(tf.abs(noise))

    def reset_noise(self, input_shape):
        eps_in = self._scale_noise(input_shape)
        eps_out = self._scale_noise(self.units)
        self.weights_eps = tf.multiply(tf.expand_dims(eps_in, 1), eps_out)
        self.bias_eps = eps_out
