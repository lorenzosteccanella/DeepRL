import tensorflow as tf
from tensorflow import keras
import numpy as np
from Losses.Losses import Losses
#import tensorflow.contrib.slim as slim
import inspect
from .CommonGoalA2C import *


class GoalA2CEagerSync:

    def __init__(self, h_size, n_actions, model_critic, model_actor, learning_rate, weight_mse, weight_ce, shared_observation_model=None, learning_rate_observation_adjust=1, train_observation=False):

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
        self.weight_ce = weight_ce

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        self.global_step = tf.Variable(0)

    def prediction_actor(self, s1, s2, s3):

        s1 = np.array(s1, dtype=np.float32)

        s1 = tf.convert_to_tensor(s1)

        s2 = np.array(s2, dtype=np.float32)

        s2 = tf.convert_to_tensor(s2)

        s3 = np.array(s3, dtype=np.float32)

        s3 = tf.convert_to_tensor(s3)

        logits = self.model_actor_critic(s1, s2, s3)[0]
        prob = tf.nn.softmax(logits)
        return prob.numpy()

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
            logits, value_critic, _ = model_actor_critic(*inputs)
            loss_pg = Losses.reinforce_loss(logits, one_hot_a, advantage)
            loss_ce = Losses.entropy_exploration_loss(logits)
            loss_critic = Losses.mse_loss(value_critic, targets)
            loss_value = (weight_mse * loss_critic) + (loss_pg - (weight_ce * loss_ce))

        return loss_value, tape.gradient(loss_value, model_actor_critic.trainable_variables), loss_ce

    def train(self, s1, s2, s3, y, one_hot_a, advantage, max_grad_norm=40):

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


class GoalA2CEagerSeparate:

    def __init__(self, h_size, n_actions, model_critic, model_actor, learning_rate, weight_mse, weight_ce, shared_observation_model=False, learning_rate_observation_adjust=False, train_observation=False):

        if inspect.isclass(model_critic):
            self.model_critic = model_critic(h_size)
        else:
            self.model_critic = model_critic

        if inspect.isclass(model_actor):
            self.model_actor = model_actor(h_size, n_actions)
        else:
            self.model_actor = model_actor

        self.weight_mse = weight_mse
        self.weight_ce = weight_ce
        self.n_actions = n_actions

        self.optimizer_critic = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        self.optimizer_actor = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        self.global_step = tf.Variable(0)

    def prediction_actor(self, s1, s2, s3):

        s1 = np.array(s1, dtype=np.float32)

        s1 = tf.convert_to_tensor(s1)

        s2 = np.array(s2, dtype=np.float32)

        s2 = tf.convert_to_tensor(s2)

        s3 = np.array(s3, dtype=np.float32)

        s3 = tf.convert_to_tensor(s3)

        logits = self.model_actor(s1, s2, s3)
        prob = tf.nn.softmax(logits)
        return prob.numpy()

    def prediction_critic(self, s1, s2, s3):

        s1 = np.array(s1, dtype=np.float32)

        s1 = tf.convert_to_tensor(s1)

        s2 = np.array(s2, dtype=np.float32)

        s2 = tf.convert_to_tensor(s2)

        s3 = np.array(s3, dtype=np.float32)

        s3 = tf.convert_to_tensor(s3)

        return self.model_critic(s1, s2, s3).numpy()

    def grad_actor(self, model, inputs, one_hot_a, advantage, weight_ce):

        with tf.GradientTape() as tape:
            outputs = model(*inputs)
            loss_pg = Losses.reinforce_loss(outputs, one_hot_a, advantage)
            loss_ce = Losses.entropy_exploration_loss(outputs)
            loss_value = loss_pg - (weight_ce * loss_ce)

        return loss_value, tape.gradient(loss_value, model.trainable_variables), loss_ce

    def grad_critic(self, model, inputs, targets, weight_mse=0.5):

        with tf.GradientTape() as tape:
            outputs = model(*inputs)
            loss_value = weight_mse * Losses.mse_loss(outputs, targets)

        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def train(self, s1, s2, s3, y, one_hot_a, advantage, max_grad_norm=40):

        s1 = np.array(s1, dtype=np.float32)

        s1 = tf.convert_to_tensor(s1)

        s2 = np.array(s2, dtype=np.float32)

        s2 = tf.convert_to_tensor(s2)

        s3 = np.array(s3, dtype=np.float32)

        s3 = tf.convert_to_tensor(s3)

        loss_value, grads = self.grad_critic(self.model_critic, (s1, s2, s3), y, self.weight_mse)

        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        self.optimizer_critic.apply_gradients(zip(grads, self.model_critic.trainable_variables), self.global_step)

        loss_value, grads, loss_ce = self.grad_actor(self.model_actor, (s1, s2, s3), one_hot_a, advantage, self.weight_ce)

        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        self.optimizer_actor.apply_gradients(zip(grads, self.model_actor.trainable_variables),
                                             self.global_step)

        return [None, None, loss_ce.numpy()]


    def save_weights(self):
        self.model_actor_critic.save_weights("/home/lorenzo/Documenti/UPF/DeepRL/TF_models_weights/A2C_weights")

    def load_weights(self):
        self.model_actor_critic.load_weights("/home/lorenzo/Documenti/UPF/DeepRL/TF_models_weights/A2C_weights")