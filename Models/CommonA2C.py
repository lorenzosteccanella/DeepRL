import tensorflow as tf
from tensorflow import keras
import numpy as np
from Losses.Losses import Losses
#import tensorflow.contrib.slim as slim
import inspect

# self.conv1 = keras.layers.Conv2D(32, 8, (4, 4), padding='VALID', activation='elu', kernel_initializer='he_normal', )
# self.conv2 = keras.layers.Conv2D(64, 4, (2, 2), padding='VALID', activation='elu', kernel_initializer='he_normal')
# self.conv3 = keras.layers.Conv2D(64, 3, (1, 1), padding='VALID', activation='elu', kernel_initializer='he_normal')
# self.flatten = keras.layers.Flatten()
# self.dense = keras.layers.Dense(256)

class SharedConvLayers(keras.Model):
    def __init__(self, learning_rate_observation_adjust=1):
        super(SharedConvLayers, self).__init__(name="SharedConvLayers")
        self.conv1 = keras.layers.Conv2D(32, 8, (4, 4), padding='VALID', activation='relu', kernel_initializer='he_uniform', )
        self.conv2 = keras.layers.Conv2D(64, 4, (2, 2), padding='VALID', activation='relu', kernel_initializer='he_uniform')
        self.conv3 = keras.layers.Conv2D(64, 3, (1, 1), padding='VALID', activation='relu', kernel_initializer='he_uniform')
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
        #denseOut = self.normalization_layer(denseOut)
        x = self.learning_rate_adjust * denseOut + (1-self.learning_rate_adjust) * tf.stop_gradient(denseOut)  # U have to test this!!!

        return [x, denseOut] # super importante ricordati che negli actor e critic modelli stai indicizzando a 0 ho bisogno di questo per la vae observation

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
        x = self.learning_rate_adjust * x + (1 - self.learning_rate_adjust) * tf.stop_gradient(x)  # U have to test this!!!

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

# class ProbabilityDistribution(tf.keras.Model):
#   def call(self, logits, **kwargs):
#     # Sample a random categorical action from the given logits.
#     return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class ActorNetwork(keras.Model):
    def __init__(self, h_size, n_actions):
        super(ActorNetwork, self).__init__(name="ActorNetwork")
        self.dense1 = keras.layers.Dense(h_size, activation='relu', kernel_initializer='random_uniform')
        self.dense2 = keras.layers.Dense(h_size, activation='relu', kernel_initializer='random_uniform')
        self.out = keras.layers.Dense(n_actions, activation=None, kernel_initializer='random_uniform')
        #self.dist = ProbabilityDistribution()

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.out(x)

        return x

    def get_action(self, x):
        #logits = self.call(x)
        #action = self.dist.predict(logits)
        #return np.squeeze(action, axis=-1)
        return None

class ActorCriticNetwork(keras.Model):

    def __init__(self, critic_model, actor_model, shared_observation_model=None):
        super(ActorCriticNetwork, self).__init__(name="ActorCriticNetwork")
        self.shared_observation_model = shared_observation_model
        self.critic_model = critic_model
        self.actor_model = actor_model
        #self.dist = ProbabilityDistribution()

    def call(self, x):

        if self.shared_observation_model is not None:

            obs = self.shared_observation_model(x)[0] # Just the dense output
        else:
            obs = x

        actor = self.actor_model(obs)

        critic = self.critic_model(obs)

        return actor, critic, None

    def get_action(self, x):
        # if self.shared_observation_model is not None:
        #
        #     obs = self.shared_observation_model(x)[0] # Just the dense output
        # else:
        #     obs = x
        #
        # logits = self.actor_model(obs)
        # action = self.dist.predict(logits)
        #
        # return np.squeeze(action, axis=-1)
        return None