import tensorflow as tf
from tensorflow import keras
import numpy as np
from Losses.Losses import Losses
import tensorflow.contrib.slim as slim
import inspect

class SharedConvLayers(keras.Model):
    def __init__(self, learning_rate_observation_adjust=1):
        super(SharedConvLayers, self).__init__(name="SharedConvLayers")
        self.conv1 = keras.layers.Conv2D(32, 8, (4, 4), padding='VALID', activation='elu', kernel_initializer='he_normal', )
        self.conv2 = keras.layers.Conv2D(64, 4, (2, 2), padding='VALID', activation='elu', kernel_initializer='he_normal')
        self.conv3 = keras.layers.Conv2D(64, 3, (1, 1), padding='VALID', activation='elu', kernel_initializer='he_normal')
        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(256)
        self.learning_rate_adjust = learning_rate_observation_adjust

    def call(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.learning_rate_adjust * x + (1-self.learning_rate_adjust) * tf.stop_gradient(x)  # U have to test this!!!

        return [x] # super importante ricordati che negli actor e critic modelli stai indicizzando a 0 ho bisogno di questo per la vae observation


class SharedDenseLayers(keras.Model):
    def __init__(self, h_size):
        super(SharedDenseLayers, self).__init__(name="SharedDenseLayers")
        self.dense1 = keras.layers.Dense(h_size, activation='elu', kernel_initializer='he_normal')
        self.dense2 = keras.layers.Dense(h_size, activation='elu', kernel_initializer='he_normal')

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)

        return [x]


class CriticNetwork(keras.Model):
    def __init__(self, h_size):
        super(CriticNetwork, self).__init__(name="CriticNetwork")
        self.dense1 = keras.layers.Dense(h_size, activation='elu', kernel_initializer='he_normal')
        self.dense2 = keras.layers.Dense(h_size, activation='elu', kernel_initializer='he_normal')
        self.out = keras.layers.Dense(1, activation='linear')

    def call(self, x):

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.out(x)
        return x


class ActorNetwork(keras.Model):
    def __init__(self, h_size, n_actions):
        super(ActorNetwork, self).__init__(name="ActorNetwork")
        self.dense1 = keras.layers.Dense(h_size, activation='elu', kernel_initializer='he_normal')
        self.dense2 = keras.layers.Dense(h_size, activation='elu', kernel_initializer='he_normal')
        self.out = keras.layers.Dense(n_actions, activation=keras.activations.softmax)

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

            x = self.shared_observation_model(x)[0] # Just the dense output

        actor = self.actor_model(x)

        critic = self.critic_model(x)

        return actor, critic

class A2CSILEagerSync:

    def __init__(self, h_size, n_actions, model_critic, model_actor, learning_rate_online, weight_mse, weight_sil_mse, weight_ce, shared_observation_model=None, train_observation=False):

        if inspect.isclass(shared_observation_model):
            self.shared_observation_model = shared_observation_model()
        else:
            self.shared_observation_model = shared_observation_model

        if inspect.isclass(model_critic):
            self.model_critic = model_critic(h_size)
        else:
            self.model_critic = model_critic

        if inspect.isclass(model_actor):
            self.model_actor = model_actor(h_size, n_actions)
        else:
            self.model_actor = model_actor

        self.model_actor_critic = ActorCriticNetwork(self.model_critic, self.model_actor, self.shared_observation_model)

        self.train_observation = train_observation
        self.weight_mse = weight_mse
        self.weight_sil_mse = weight_sil_mse
        self.weight_ce = weight_ce

        #print("\n ACTOR CRITIC MODEL \n")

        #slim.model_analyzer.analyze_vars(self.model_actor_critic.trainable_variables, print_info=True)

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate_online)
        #self.optimizer_imitation = tf.train.RMSPropOptimizer(learning_rate=learning_rate_imitation)
        if self.train_observation:
            self.optimizer_observation = tf.train.RMSPropOptimizer(learning_rate=1e-3)
        self.global_step = tf.Variable(0)

    def prediction(self, s):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        a = self.model_actor(s)

        return np.argmax(a, 1)

    def prediction_actor(self, s):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        return self.model_actor_critic(s)[0].numpy()

    def prediction_critic(self, s):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        return self.model_actor_critic(s)[1].numpy()

    def grad(self, model_actor_critic, inputs, targets, one_hot_a, advantage, weight_ce, weight_mse = 0.5):

        with tf.GradientTape() as tape:
            softmax_logits, value_critic = model_actor_critic(inputs)
            loss_pg = Losses.reinforce_loss(softmax_logits, one_hot_a, advantage)
            loss_ce = Losses.entropy_exploration_loss(softmax_logits)
            loss_critic = Losses.mse_loss(value_critic, targets)
            loss_value = (weight_mse * loss_critic) + (loss_pg - weight_ce * loss_ce)

        return loss_value, tape.gradient(loss_value, model_actor_critic.trainable_variables)

    def grad_imitation(self, model_actor_critic, inputs, targets, one_hot_a, advantage, imp_w, weight_mse=0.5, weight_sil_mse=0.01):

        with tf.GradientTape() as tape:
            softmax_logits, value_critic = model_actor_critic(inputs)
            loss_pg_imitation = Losses.reinforce_loss_imp_w(softmax_logits, one_hot_a, advantage, imp_w)
            loss_critic_imitation = (weight_mse * Losses.mse_loss_self_imitation_learning_imp_w(value_critic, targets, imp_w))
            loss_value = weight_sil_mse * loss_critic_imitation + loss_pg_imitation

        return loss_value, tape.gradient(loss_value, model_actor_critic.trainable_variables)

    def grad_observation(self, model, inputs):

        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss_value = Losses.vae_loss(inputs, outputs[3], outputs[1], outputs[2])

        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def train(self, s, y, one_hot_a, advantage, max_grad_norm=5):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        loss_value, grads = self.grad(self.model_actor_critic, s, y, one_hot_a, advantage, self.weight_ce, self.weight_mse)

        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        self.optimizer.apply_gradients(zip(grads, self.model_actor_critic.trainable_variables), self.global_step)

        return [None, None]

    def train_imitation(self, s, y, one_hot_a, advantage, imp_w, max_grad_norm=5):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        loss_value, grads = self.grad_imitation(self.model_actor_critic, s, y, one_hot_a, advantage, imp_w, self.weight_mse, self.weight_sil_mse)

        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        self.optimizer.apply_gradients(zip(grads, self.model_actor_critic.trainable_variables), self.global_step)   # separate optimizer why doesn't work?

    def train_obs(self, s):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        loss_value, grads = self.grad_observation(self.shared_observation_model, s)

        print("OBSERVATION", loss_value)

        # grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        self.optimizer_observation.apply_gradients(zip(grads, self.shared_observation_model.trainable_variables),
                                                   self.global_step)

        return [None, None]