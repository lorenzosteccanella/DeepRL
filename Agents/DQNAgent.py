import random
import math
import numpy as np
from Agents.AbstractAgent import AbstractAgent


class DQNAgent(AbstractAgent):

    exp = 0
    epsilon = 1

    def __init__(self, action_space, state_dimension, buffer, main_model_nn, target_model_nn, LAMBDA,
                 update_target_freq, update_fn, gamma, batch_size, MIN_EPSILON, analyze_memory = False):
        self.action_space = action_space
        self.state_dimension = state_dimension
        self.buffer = buffer
        self.main_model_nn = main_model_nn
        self.target_model_nn = target_model_nn
        self.LAMBDA = LAMBDA
        self.update_target_freq = update_target_freq
        self.update_fn = update_fn
        self.gamma = gamma
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

        states = np.array([o[1][0] for o in batch])
        states_ = np.array([o[1][3] for o in batch])

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
            done = o[4]

            t = p[i]
            a_index = self.action_space.index(a)
            old_val = t[a_index]
            if done:
                t[a_index] = r
            else:
                t[a_index] = r + self.gamma * np.amax(p_target_[i]) # DQN

            x[i] = s
            y[i] = t
            errors[i] = abs(old_val - t[a_index])
        return x, y, errors

    def observe(self, sample):  # in (s, a, r, s_, done, info) format
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
