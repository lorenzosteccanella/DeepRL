import math
import random
import numpy as np
from abc import ABC, abstractmethod
from Utils.Utils import ExperienceReplay, PrioritizedExperienceReplay
import matplotlib.pyplot as plt


class RandomAgent:

    random.seed(1)

    exp = 0

    def __init__(self, action_space, buffer):
        self.action_space = action_space
        self.buffer = buffer

    def act(self, s):
        return random.choice(self.action_space)

    def observe(self, sample):  # in (s, a, r, s_) format
        # error = abs(sample[2])  # reward
        self.buffer.add(sample)
        self.exp += 1

    def replay(self):
        pass


class DQNAgent:

    random.seed(1)

    exp = 0
    epsilon = 1

    def __init__(self, action_space, state_dimension, buffer, main_model_nn, target_model_nn, LAMBDA,
                 update_target_freq, update_fn, gamma, batch_size, tf_sess, MIN_EPSILON, analyze_memory = False):
        self.action_space = action_space
        self.state_dimension = state_dimension
        self.buffer = buffer
        self.main_model_nn = main_model_nn
        self.target_model_nn = target_model_nn
        self.LAMBDA = LAMBDA
        self.update_target_freq = update_target_freq
        self.update_fn = update_fn
        self.gamma = gamma
        self.tf_sess = tf_sess  # we need always the current session
        self.batch_size = batch_size
        self.MIN_EPSILON = MIN_EPSILON
        self.analyze_memory = analyze_memory

    def act(self, s):
        if random.random() < self.epsilon:
            return random.choice(self.action_space)
        else:
            index_action = self.main_model_nn.prediction([s])[0]

            return self.action_space[index_action]

    def _get_tderror(self, batch):

        no_state = np.zeros(self.state_dimension)

        states = np.array([o[1][0] for o in batch])
        states_ = np.array([(no_state if o[1][3] is None else o[1][3]) for o in batch])

        p = self.main_model_nn.Qprediction(states)
        p_target_ = self.target_model_nn.Qprediction(states_)

        x = np.zeros((len(batch),) + self.state_dimension)
        y = np.zeros((len(batch), len(self.action_space)))
        errors = np.zeros(len(batch))

        for i in range(len(batch)):
            o = batch[i][1]
            s = o[0]
            a = o[1]
            r = o[2]
            s_ = o[3]

            t = p[i]
            a_index = self.action_space.index(a)
            old_val = t[a_index]
            if s_ is None:
                t[a_index] = r
            else:
                t[a_index] = r + self.gamma * np.amax(p_target_[i]) # DQN

            x[i] = s
            y[i] = t
            errors[i] = abs(old_val - t[a_index])
        return x, y, errors

    def observe(self, sample):  # in (s, a, r, s_) format
        # x, y, errors = self._get_tderror([(0, sample)])
        self.buffer.add(sample)
        if self.exp % self.update_target_freq == 0:
            self.update_fn.update()

        # slowly decrease Epsilon based on our experience
        self.exp += 1
        self.epsilon = self.MIN_EPSILON + (1 - self.MIN_EPSILON) * math.exp(-self.LAMBDA * self.exp)
        #self.epsilon -= self.epsilon_step



    def replay(self):

        if self.buffer.buffer_len() >= self.batch_size:
            batch, imp_w = self.buffer.sample(self.batch_size)
            if self.analyze_memory:
                self.analyze_memory.add_batch(batch)
            x, y, errors = self._get_tderror(batch)

            # update errors if you use prioritized experience replay
            for i in range(len(batch)):
                idx = batch[i][0]
                self.buffer.update(idx, errors[i])

            _, loss = self.main_model_nn.train( x, y, imp_w)
            return loss



    def replay(self):

        if self.buffer.buffer_len() >= self.batch_size:
            batch, imp_w = self.buffer.sample(self.batch_size)
            if self.analyze_memory:
                self.analyze_memory.add_batch(batch)
            x, y, errors = self._get_tderror(batch)

            # update errors if you use prioritized experience replay
            for i in range(len(batch)):
                idx = batch[i][0]
                self.buffer.update(idx, errors[i])

            _, loss = self.main_model_nn.train( x, y, imp_w)
            return loss


class DDQNAgent:
    random.seed(1)

    exp = 0
    epsilon = 1

    def __init__(self, action_space, state_dimension, buffer, main_model_nn, target_model_nn, LAMBDA,
                 update_target_freq, update_fn, gamma, batch_size, tf_sess, MIN_EPSILON, analyze_memory=False):
        self.action_space = action_space
        self.state_dimension = state_dimension
        self.buffer = buffer
        self.main_model_nn = main_model_nn
        self.target_model_nn = target_model_nn
        self.LAMBDA = LAMBDA
        self.update_target_freq = update_target_freq
        self.update_fn = update_fn
        self.gamma = gamma
        self.tf_sess = tf_sess  # we need always the current session
        self.batch_size = batch_size
        self.MIN_EPSILON = MIN_EPSILON
        self.analyze_memory = analyze_memory

    def act(self, s):
        if random.random() < self.epsilon:
            return random.choice(self.action_space)
        else:
            index_action = self.main_model_nn.prediction([s])[0]

            return self.action_space[index_action]

    def _get_tderror(self, batch):

        no_state = np.zeros(self.state_dimension)

        states = np.array([o[1][0] for o in batch])
        states_ = np.array([(no_state if o[1][3] is None else o[1][3]) for o in batch])

        p = self.main_model_nn.Qprediction(states)
        p_ = self.main_model_nn.Qprediction(states_)
        p_target_ = self.target_model_nn.Qprediction(states_)

        x = np.zeros((len(batch),) + self.state_dimension)
        y = np.zeros((len(batch), len(self.action_space)))
        errors = np.zeros(len(batch))

        for i in range(len(batch)):
            o = batch[i][1]
            s = o[0]
            a = o[1]
            r = o[2]
            s_ = o[3]

            t = p[i]
            a_index = self.action_space.index(a)
            old_val = t[a_index]
            if s_ is None:
                t[a_index] = r
            else:

                t[a_index] = r + self.gamma * p_target_[i][np.argmax(p_[i])]  # double DQN

                #t[a_index] = r + self.gamma * np.amax(p_target_[i])  # Double DQN

            x[i] = s
            y[i] = t
            errors[i] = abs(old_val - t[a_index])
        return x, y, errors

    def observe(self, sample):  # in (s, a, r, s_) format
        # x, y, errors = self._get_tderror([(0, sample)])
        self.buffer.add(sample)
        if self.exp % self.update_target_freq == 0:
            self.update_fn.update()

        # slowly decrease Epsilon based on our experience
        self.exp += 1
        self.epsilon = self.MIN_EPSILON + (1 - self.MIN_EPSILON) * math.exp(-self.LAMBDA * self.exp)
        # self.epsilon -= self.epsilon_step

    def replay(self):

        if self.buffer.buffer_len() >= self.batch_size:
            batch, imp_w = self.buffer.sample(self.batch_size)
            if self.analyze_memory:
                self.analyze_memory.add_batch(batch)
            x, y, errors = self._get_tderror(batch)

            # update errors if you use prioritized experience replay
            for i in range(len(batch)):
                idx = batch[i][0]
                self.buffer.update(idx, errors[i])

            _, loss = self.main_model_nn.train(x, y, imp_w)
            return loss


class ReinforceAgent:

    np.random.seed(1)

    def __init__(self, action_space, state_dimension, buffer, main_model_nn, gamma, analyze_memory = False):
        self.buffer=buffer

        self.action_space = action_space
        self.state_dimension = state_dimension
        self.main_model_nn = main_model_nn
        self.analyze_memory = analyze_memory
        self.gamma = gamma

        # Placeholders for our observations, outputs and rewards
        self.batch = []

    def _discount_rewards(self, r, gamma=0.8):
        """Takes 1d float array of rewards and computes discounted reward
        e.g. f([1, 1, 1], 0.99) -> [2.9701, 1.99, 1]
        """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def act(self, s):

        predict = self.main_model_nn.PolicyPrediction([s])[0]

        return np.random.choice(self.action_space, p=predict)

    def observe(self, sample): # in (s, a, r, s_) format
        self.batch.append(sample)

        if sample[3] is None:
            states = []
            rewards = []
            a_one_hot = np.zeros((len(self.batch), len(self.action_space)))
            for i in range(len(self.batch)):
                o = self.batch[i][0]
                a = self.batch[i][1]
                r = self.batch[i][2]

                a_index = self.action_space.index(a)
                a_one_hot[i][a_index] = 1
                states.append(o)
                rewards.append(r)

            np_rewards = np.asarray(rewards)
            np_states = np.asarray(states)

            discounted_rewards = self._discount_rewards(np_rewards, self.gamma)

            _, loss = self.main_model_nn.train(np_states, a_one_hot, discounted_rewards)

            self.batch.clear()

    def replay(self):
        pass


class A2CAgent:

    np.random.seed(1)

    def __init__(self, action_space, main_model_nn, gamma, batch_size, analyze_memory = False):

        self.batch_size = batch_size
        self.buffer = ExperienceReplay(self.batch_size)
        self.action_space = action_space
        self.main_model_nn = main_model_nn
        self.analyze_memory = analyze_memory
        self.gamma = gamma

    def _get_actor_critic_error(self, batch):

        states_t = np.array([o[1][0] for o in batch])
        p = self.main_model_nn.prediction_critic(states_t)[:, 0]
        a_one_hot = np.zeros((len(batch), len(self.action_space)))
        dones = np.zeros((len(batch)))
        rewards = np.zeros((len(batch)))

        for i in range(len(batch)):
            o = batch[i][1]
            a = o[1]
            r = o[2]
            s_ = o[3]

            a_index = self.action_space.index(a)

            if s_ is None:
                dones[i] = 1
                p_ = [0]
            elif i == len(batch)-1:
                p_ = self.main_model_nn.prediction_critic([s_])[0]
            rewards[i] = r
            a_one_hot[i][a_index] = 1

        y_critic, adv_actor = self._returns_advantages(rewards, dones, p, p_)
        y_critic = np.expand_dims(y_critic, axis=-1)
        return states_t, adv_actor, a_one_hot, y_critic

    def _returns_advantages(self, rewards, dones, values, next_value):
        # next_value is the bootstrap value estimate of a future state (the critic)
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        # returns are calculated as discounted sum of future rewards
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]
        # advantages are returns - baseline, value estimates in our case
        advantages = returns - values
        return returns, advantages

    def act(self, s):

        predict = self.main_model_nn.prediction_actor([s])[0]

        return np.random.choice(self.action_space, p=predict)

    def observe(self, sample): # in (s, a, r, s_) format

        self.buffer.add(sample)

    def replay(self):

        if self.buffer.buffer_len() >= self.batch_size:

            batch, imp_w = self.buffer.sample(self.batch_size, False)

            x, adv_actor, a_one_hot, y_critic = self._get_actor_critic_error(batch)

            self.main_model_nn.train(x, y_critic, a_one_hot, adv_actor)
            #self.main_model_nn.train_critic(x, y_critic)
            #self.main_model_nn.train_obs(x)

            self.buffer.reset_buffer()


class A2C_SIL_Agent:

    np.random.seed(1)

    def __init__(self, action_space, main_model_nn, gamma, batch_size, sil_batch_size, imitation_buffer_size, imitation_learnig_steps, analyze_memory = False):

        self.batch_size = batch_size
        self.buffer_online = ExperienceReplay(self.batch_size)
        self.buffer_imitation = PrioritizedExperienceReplay(imitation_buffer_size)
        self.trajectory = []
        self.action_space = action_space
        self.main_model_nn = main_model_nn
        self.analyze_memory = analyze_memory
        self.gamma = gamma
        self.imitation_learning_steps = imitation_learnig_steps
        self.sil_batch_size = sil_batch_size

    def _get_actor_critic_error(self, batch):

        states_t = np.array([o[1][0] for o in batch])
        p = self.main_model_nn.prediction_critic(states_t)[0]
        a_one_hot = np.zeros((len(batch), len(self.action_space)))
        dones = np.zeros((len(batch)))
        rewards = np.zeros((len(batch)))

        for i in range(len(batch)):
            o = batch[i][1]
            a = o[1]
            r = o[2]
            s_ = o[3]

            a_index = self.action_space.index(a)

            if s_ is None:
                dones[i] = 1
                p_ = [0]
            elif i == len(batch)-1:
                p_ = self.main_model_nn.prediction_critic([s_])[0]

            rewards[i] = r
            a_one_hot[i][a_index] = 1

        y_critic, adv_actor = self._returns_advantages(rewards, dones, p, p_)
        y_critic = np.expand_dims(y_critic, axis=-1)

        return states_t, adv_actor, a_one_hot, y_critic

    def _get_imitation_error(self, batch):
        states_t = np.array([o[1][0] for o in batch])
        p = self.main_model_nn.prediction_critic(states_t)[0]
        a_one_hot = np.zeros((len(batch), len(self.action_space)))
        rewards = np.zeros((len(batch)))
        for i in range(len(batch)):
            o = batch[i][1]
            a = o[1]
            r = o[2]
            rewards[i] = r
            a_index = self.action_space.index(a)
            a_one_hot[i][a_index] = 1

        advantages = rewards - p
        clip_advantages = np.clip(advantages, a_min=0, a_max=np.inf)

        y_critic = np.expand_dims(rewards, axis=-1)

        return states_t, clip_advantages, a_one_hot, y_critic

    def _returns_advantages(self, rewards, dones, values, next_value):
        # next_value is the bootstrap value estimate of a future state (the critic)
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        # returns are calculated as discounted sum of future rewards
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]
        # advantages are returns - baseline, value estimates in our case
        advantages = returns - values
        return returns, advantages

    def _discount_rewards(self, r, gamma=0.8):
        """Takes 1d float array of rewards and computes discounted reward
        e.g. f([1, 1, 1], 0.99) -> [2.9701, 1.99, 1]
        """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def act(self, s):

        predict = self.main_model_nn.prediction_actor([s])[0]

        return np.random.choice(self.action_space, p=predict)

    def observe(self, sample):
        self.buffer_online.add(sample)
        self.add_multy_trajectory_memory(sample)

    def add_multy_trajectory_memory(self, sample):
        self.trajectory.append((sample[0], sample[1], sample[2]))
        if sample[3] is None:
            s = np.array([o[0] for o in self.trajectory])
            a = np.array([o[1] for o in self.trajectory])
            r = np.array([o[2] for o in self.trajectory])
            discounted_rewards = self._discount_rewards(r, self.gamma)
            for i in range(len(discounted_rewards)):
                self.buffer_imitation.add((s[i], a[i], discounted_rewards[i]))

            self.trajectory.clear()

    def add_single_trajectory_memory(self, sample): # in (s, a, r, s_) format
        self.trajectory.append([sample[0], sample[1], sample[2]])

        if sample[3] is None:
            s = np.array([o[1] for o in self.trajectory])
            a = np.array([o[1] for o in self.trajectory])
            r = np.array([o[2] for o in self.trajectory])
            discounted_rewards = self._discount_rewards(r, self.gamma)
            total_reward = np.sum(r) / len(r)
            if total_reward > self.buffer_imitation.max_reward :
                self.buffer_imitation.max_reward=total_reward
                self.buffer_imitation.reset_buffer() # only one trajectory per time

                for i in range(len(discounted_rewards)):
                    self.buffer_imitation.add((s[i], a[i], r[i]))

            self.trajectory.clear()

    def replay(self):
        if self.buffer_online.buffer_len() >= self.batch_size:

            batch, imp_w = self.buffer_online.sample(self.batch_size, False)
            x, adv_actor, a_one_hot, y_critic = self._get_actor_critic_error(batch)

            self.main_model_nn.train(x, y_critic, a_one_hot, adv_actor)
            #self.main_model_nn.train_actor(x, a_one_hot, adv_actor, self.weight_ce_exploration)
            #self.main_model_nn.train_critic(x, y_critic)
            #self.main_model_nn.train_obs(x)

            self.buffer_online.reset_buffer()

            if self.buffer_imitation.buffer_len() >= self.sil_batch_size:
                for i in range(self.imitation_learning_steps):
                    batch_imitation, imp_w = self.buffer_imitation.sample(self.sil_batch_size)
                    x, adv_actor, a_one_hot, y_critic = self._get_imitation_error(batch_imitation)
                    #self.main_model_nn.train_actor_imitation(x, a_one_hot, adv_actor, imp_w)
                    #self.main_model_nn.train_critic_imitation(x, y_critic, imp_w)

                    self.main_model_nn.train_imitation(x, y_critic, a_one_hot, adv_actor, imp_w)

                    # update errors
                    for k in range(len(batch_imitation)):
                        idx = batch_imitation[k][0]
                        self.buffer_imitation.update(idx, adv_actor[k])
