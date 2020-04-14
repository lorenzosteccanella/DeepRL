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
        self.conv1 = keras.layers.Conv2D(16, 8, (4, 4), padding='VALID', activation='elu', kernel_initializer='he_normal')#, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.conv2 = keras.layers.Conv2D(16, 4, (2, 2), padding='VALID', activation='elu', kernel_initializer='he_normal')#, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.conv3 = keras.layers.Conv2D(16, 3, (1, 1), padding='VALID', activation='elu', kernel_initializer='he_normal')#, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(256, activation='elu', kernel_initializer='he_normal')#, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.normalization_layer = keras.layers.LayerNormalization()
        self.learning_rate_adjust = learning_rate_observation_adjust

    def call(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        denseOut = self.dense(x)
        x = self.learning_rate_adjust * denseOut + (1-self.learning_rate_adjust) * tf.stop_gradient(denseOut)  # U have to test this!!!
        #x = self.normalization_layer(x)
        return [x, denseOut]


class SharedDenseLayers(keras.Model):
    def __init__(self, h_size=256, learning_rate_observation_adjust=1):
        super(SharedDenseLayers, self).__init__(name="SharedDenseLayers")
        self.dense1 = keras.layers.Dense(h_size, activation='elu', kernel_initializer='he_normal')#, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.dense2 = keras.layers.Dense(h_size, activation='elu', kernel_initializer='he_normal')#, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.learning_rate_adjust = learning_rate_observation_adjust

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.learning_rate_adjust * x + (1 - self.learning_rate_adjust) * tf.stop_gradient(x)  # U have to test this!!!

        return [x, x]

class SharedGoalModel(keras.Model):
    def __init__(self, h_size=256, learning_rate_observation_adjust=1):
        super(SharedGoalModel, self).__init__(name="SharedGoalModel")
        self.dense1 = keras.layers.Dense(h_size, activation='elu', kernel_initializer='he_normal')#, kernel_regularizer=tf.keras.regularizers.l1(0.01))
        self.dense2 = keras.layers.Dense(h_size, activation='elu', kernel_initializer='he_normal')#, kernel_regularizer=tf.keras.regularizers.l1(0.01))
        self.normalization_layer = keras.layers.LayerNormalization()
        self.learning_rate_adjust = learning_rate_observation_adjust

    def call(self, x):
        x = self.dense1(x)
        denseOut = self.dense2(x)
        x = self.learning_rate_adjust * denseOut + (1-self.learning_rate_adjust) * tf.stop_gradient(denseOut)  # U have to test this!!!
        #x = self.normalization_layer(x)
        return [x, x]


class CriticNetwork(keras.Model):
    def __init__(self, h_size):
        super(CriticNetwork, self).__init__(name="CriticNetwork")
        self.dense1 = keras.layers.Dense(h_size, activation='elu', kernel_initializer='he_normal')#, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.dense2 = keras.layers.Dense(h_size, activation='elu', kernel_initializer='he_normal')#, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.out = keras.layers.Dense(1, activation='linear')#, kernel_regularizer=tf.keras.regularizers.l2(0.01))

    def call(self, x):

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.out(x)
        return x


class ActorNetwork(keras.Model):
    def __init__(self, h_size, n_actions):
        super(ActorNetwork, self).__init__(name="ActorNetwork")
        self.dense1 = keras.layers.Dense(h_size, activation='elu', kernel_initializer='he_normal')#, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.dense2 = keras.layers.Dense(h_size, activation='elu', kernel_initializer='he_normal')#, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.out = keras.layers.Dense(n_actions, activation=keras.activations.softmax)#, kernel_regularizer=tf.keras.regularizers.l2(0.01))

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.out(x)
        return x


class SiameseActorCriticNetwork(keras.Model):

    def __init__(self, critic_model, actor_model, shared_observation_model=None, shared_goal_model_start=None, shared_goal_model_goal=None):
        super(SiameseActorCriticNetwork, self).__init__(name="SiameseActorCriticNetwork")
        self.shared_observation_model = shared_observation_model
        self.shared_goal_model_start = shared_goal_model_start
        self.shared_goal_model_goal = shared_goal_model_goal
        self.critic_model = critic_model
        self.actor_model = actor_model
        self.L1_layer = keras.layers.Lambda(lambda tensors: keras.backend.abs(tensors[0] - tensors[1]))
        self.L2_layer = keras.layers.Lambda(lambda tensors: keras.backend.pow(tensors[0] - tensors[1], 2))

    def call(self, x1, x2, x3):

        obs1 = self.shared_observation_model(x1)[0] # Just the dense output
        if self.shared_goal_model_start is not False:
            obs2 = self.shared_goal_model_start(x2)[0] # Just the dense output
        obs3 = self.shared_goal_model_goal(x3)[0]

        if self.shared_goal_model_start is not False:
            obs = keras.layers.concatenate([obs1, obs2, obs3], axis=-1)
        else:
            #obs = keras.layers.concatenate([obs1, obs3], axis=-1)
            obs = self.L2_layer([obs1, obs3])

        actor = self.actor_model(obs)

        critic = self.critic_model(obs)

        return actor, critic


class GoalA2CEagerSync:

    def __init__(self, h_size, n_actions, model_critic, model_actor, learning_rate_online, weight_mse, weight_ce,
                 shared_observation_model=None, learning_rate_observation_adjust=1, shared_goal_model_start=None,
                 shared_goal_model_goal=None, train_observation=False):

        if inspect.isclass(shared_observation_model):
            self.shared_observation_model = shared_observation_model(learning_rate_observation_adjust)
        else:
            self.shared_observation_model = shared_observation_model

        if inspect.isclass(shared_goal_model_start):
            self.shared_goal_model_start = shared_goal_model_start(learning_rate_observation_adjust)
        else:
            self.shared_goal_model_start = shared_goal_model_start

        if inspect.isclass(shared_goal_model_goal):
            self.shared_goal_model_goal = shared_goal_model_goal(learning_rate_observation_adjust)
        else:
            self.shared_goal_model_goal = shared_goal_model_goal

        if inspect.isclass(model_critic):
            self.model_critic = model_critic(h_size)
        else:
            self.model_critic = model_critic

        if inspect.isclass(model_actor):
            self.model_actor = model_actor(h_size, n_actions)
        else:
            self.model_actor = model_actor

        self.model_actor_critic = SiameseActorCriticNetwork(self.model_critic, self.model_actor,
                                                            self.shared_observation_model, self.shared_goal_model_start,
                                                            self.shared_goal_model_goal)

        self.train_observation = train_observation
        self.weight_mse = weight_mse
        self.weight_ce = weight_ce

        #print("\n ACTOR CRITIC MODEL \n")

        #slim.model_analyzer.analyze_vars(self.model_actor_critic.trainable_variables, print_info=True)

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate_online)
        #self.optimizer_imitation = tf.train.RMSPropOptimizer(learning_rate=learning_rate_imitation)
        if self.train_observation:
            self.optimizer_observation = tf.train.RMSPropOptimizer(learning_rate=1e-3)
        self.global_step = tf.Variable(0)

    def prediction_actor(self, s1, s2, s3):

        s1 = np.array(s1, dtype=np.float32)

        s1 = tf.convert_to_tensor(s1)

        s2 = np.array(s2, dtype=np.float32)

        s2 = tf.convert_to_tensor(s2)

        s3 = np.array(s3, dtype=np.float32)

        s3 = tf.convert_to_tensor(s3)

        return self.model_actor_critic(s1, s2, s3)[0].numpy()

    def prediction_critic(self, s1, s2, s3):

        s1 = np.array(s1, dtype=np.float32)

        s1 = tf.convert_to_tensor(s1)

        s2 = np.array(s2, dtype=np.float32)

        s2 = tf.convert_to_tensor(s2)

        s3 = np.array(s3, dtype=np.float32)

        s3 = tf.convert_to_tensor(s3)

        return self.model_actor_critic(s1, s2, s3)[1].numpy()

    def grad(self, model_actor_critic, inputs, targets, one_hot_a, advantage, weight_ce, weight_mse = 0.5):

        with tf.GradientTape() as tape:
            softmax_logits, value_critic = model_actor_critic(inputs[0], inputs[1], inputs[2])
            loss_pg = Losses.reinforce_loss(softmax_logits, one_hot_a, advantage)
            loss_ce = Losses.entropy_exploration_loss(softmax_logits)
            loss_critic = Losses.mse_loss(value_critic, targets)
            loss_value = (weight_mse * loss_critic) + (loss_pg - weight_ce * loss_ce)

        return loss_value, tape.gradient(loss_value, model_actor_critic.trainable_variables), loss_ce

    def train(self, s1, s2, s3, y, one_hot_a, advantage, max_grad_norm=5):

        s1 = np.array(s1, dtype=np.float32)

        s1 = tf.convert_to_tensor(s1)

        s2 = np.array(s2, dtype=np.float32)

        s2 = tf.convert_to_tensor(s2)

        s3 = np.array(s3, dtype=np.float32)

        s3 = tf.convert_to_tensor(s3)

        loss_value, grads, loss_ce = self.grad(self.model_actor_critic, (s1, s2, s3), y, one_hot_a, advantage, self.weight_ce, self.weight_mse)

        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        self.optimizer.apply_gradients(zip(grads, self.model_actor_critic.trainable_variables), self.global_step)

        return [None, None, loss_ce.numpy()]

    def save_weights(self):
        self.model_actor_critic.save_weights("/home/lorenzo/Documenti/UPF/DeepRL/TF_models_weights/A2C_weights")

    def load_weights(self):
        self.model_actor_critic.load_weights("/home/lorenzo/Documenti/UPF/DeepRL/TF_models_weights/A2C_weights")