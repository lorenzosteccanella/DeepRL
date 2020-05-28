import tensorflow as tf
from tensorflow import keras
import numpy as np
from Losses.Losses import Losses
#import tensorflow.contrib.slim as slim
import inspect
from .CommonA2C import *

class A2CSILEagerSync:

    def __init__(self, h_size, n_actions, model_critic, model_actor, learning_rate_online, weight_mse, weight_sil_mse,
                 weight_ce, shared_observation_model=None, learning_rate_observation_adjust=1, train_observation=False):

        if inspect.isclass(shared_observation_model):
            self.shared_observation_model = shared_observation_model(learning_rate_observation_adjust)
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

        self.optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=learning_rate_online)
        #self.optimizer_imitation = tf.compat.v1.train.RMSPropOptimizer(learning_rate=learning_rate_imitation)
        self.global_step = tf.Variable(0)

    def get_action(self, s):
        return self.model_actor_critic.get_action(s)

    def prediction_actor(self, s):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        logits = self.model_actor_critic(s)
        prob = tf.nn.softmax(logits)
        return prob.numpy()

    def prediction_critic(self, s):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        return self.model_actor_critic(s)[1].numpy()

    def grad(self, model_actor_critic, inputs, targets, one_hot_a, advantage, weight_ce, weight_mse = 0.5):

        with tf.GradientTape() as tape:
            logits, value_critic = model_actor_critic(inputs)
            loss_pg = Losses.reinforce_loss(logits, one_hot_a, advantage)
            loss_ce = Losses.entropy_exploration_loss(logits)
            loss_critic = Losses.mse_loss(value_critic, targets)
            loss_value = (weight_mse * loss_critic) + (loss_pg - (weight_ce * loss_ce))

        return loss_value, tape.gradient(loss_value, model_actor_critic.trainable_variables), loss_ce

    def grad_imitation(self, model_actor_critic, inputs, targets, one_hot_a, advantage, imp_w, weight_mse=0.5, weight_sil_mse=0.01):

        with tf.GradientTape() as tape:
            logits, value_critic = model_actor_critic(inputs)
            loss_pg_imitation = Losses.reinforce_loss_imp_w(logits, one_hot_a, advantage, imp_w)
            loss_critic_imitation = (weight_mse * Losses.mse_loss_self_imitation_learning_imp_w(value_critic, targets, imp_w))
            loss_value = weight_sil_mse * loss_critic_imitation + loss_pg_imitation

        return loss_value, tape.gradient(loss_value, model_actor_critic.trainable_variables)

    def train(self, s, y, one_hot_a, advantage, max_grad_norm=5):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        loss_value, grads, loss_ce = self.grad(self.model_actor_critic, s, y, one_hot_a, advantage, self.weight_ce, self.weight_mse)

        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        self.optimizer.apply_gradients(zip(grads, self.model_actor_critic.trainable_variables), self.global_step)

        return [None, None, loss_ce.numpy()]

    def train_imitation(self, s, y, one_hot_a, advantage, imp_w, max_grad_norm=5):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        loss_value, grads = self.grad_imitation(self.model_actor_critic, s, y, one_hot_a, advantage, imp_w, self.weight_mse, self.weight_sil_mse)

        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        self.optimizer.apply_gradients(zip(grads, self.model_actor_critic.trainable_variables), self.global_step)   # separate optimizer why doesn't work?


class A2CSILEagerSeparate:

    def __init__(self, h_size, n_actions, model_critic, model_actor, learning_rate_online, weight_mse, weight_sil_mse, weight_ce):

        if inspect.isclass(model_critic):
            self.model_critic = model_critic(h_size)
        else:
            self.model_critic = model_critic

        if inspect.isclass(model_actor):
            self.model_actor = model_actor(h_size, n_actions)
        else:
            self.model_actor = model_actor

        self.weight_mse = weight_mse
        self.weight_sil_mse = weight_sil_mse
        self.weight_ce = weight_ce

        self.optimizer_critic = tf.compat.v1.train.RMSPropOptimizer(learning_rate=learning_rate_online)
        self.optimizer_actor = tf.compat.v1.train.RMSPropOptimizer(learning_rate=learning_rate_online)
        self.optimizer_critic_imitation = self.optimizer_critic
        self.optimizer_actor_imitation = self.optimizer_actor
        self.global_step = tf.Variable(0)

    def get_action(self, s):
        return self.model_actor.get_action(s)

    def prediction_actor(self, s):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        logits = self.model_actor(s)
        prob = tf.nn.softmax(logits)
        return prob.numpy()

    def prediction_critic(self, s):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        return self.model_critic(s).numpy()

    def grad_actor(self, model, inputs, one_hot_a, advantage, weight_ce):

        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss_pg = Losses.reinforce_loss(outputs, one_hot_a, advantage)
            loss_ce = Losses.entropy_exploration_loss(outputs)
            loss_value = loss_pg - (weight_ce * loss_ce)

        return loss_value, tape.gradient(loss_value, model.trainable_variables), loss_ce

    def grad_critic(self, model, inputs, targets, weight_mse=0.5):

        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss_value = weight_mse * Losses.mse_loss(outputs, targets)

        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def grad_imitation_actor(self, model, inputs, one_hot_a, advantage, imp_w):

        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss_pg_imitation = Losses.reinforce_loss_imp_w(outputs, one_hot_a, advantage, imp_w)
            loss_value = loss_pg_imitation

        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def grad_imitation_critic(self, model, inputs, targets, imp_w, weight_sil_mse=0.05):

        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss_critic_imitation = Losses.mse_loss_self_imitation_learning_imp_w(outputs, targets, imp_w)
            loss_value = weight_sil_mse * loss_critic_imitation

        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def train(self, s, y, one_hot_a, advantage, max_grad_norm=0.5):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        loss_value, grads = self.grad_critic(self.model_critic, s, y, self.weight_mse)

        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        self.optimizer_critic.apply_gradients(zip(grads, self.model_critic.trainable_variables), self.global_step)

        loss_value, grads, loss_ce = self.grad_actor(self.model_actor, s, one_hot_a, advantage, self.weight_ce)

        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        self.optimizer_actor.apply_gradients(zip(grads, self.model_actor.trainable_variables),
                                             self.global_step)

        return [None, None, loss_ce.numpy()]

    def train_imitation(self, s, y, one_hot_a, advantage, imp_w, max_grad_norm=0.5):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        loss_value, grads = self.grad_imitation_critic(self.model_critic, s, y, imp_w, self.weight_sil_mse)

        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        self.optimizer_critic_imitation.apply_gradients(zip(grads, self.model_critic.trainable_variables), self.global_step)

        loss_value, grads = self.grad_imitation_actor(self.model_actor, s, one_hot_a, advantage, imp_w)

        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        self.optimizer_actor_imitation.apply_gradients(zip(grads, self.model_actor.trainable_variables),
                                             self.global_step)

        return [None, None, None]
