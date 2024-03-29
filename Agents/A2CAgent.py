import numpy as np
from Agents.AbstractAgent import AbstractAgent
from Utils import ExperienceReplay

class A2CAgent(AbstractAgent):

    def __init__(self, action_space, main_model_nn, gamma, batch_size):

        self.batch_size = batch_size
        self.buffer = ExperienceReplay(self.batch_size)
        self.action_space = action_space
        self.main_model_nn = main_model_nn
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
            done = o[4]

            a_index = self.action_space.index(a)

            if done:
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

    def observe(self, sample): # in (s, a, r, s_, done, info) format

        self.buffer.add(sample)

    def replay(self):

        if self.buffer.buffer_len() >= self.batch_size:

            batch, imp_w = self.buffer.sample(self.batch_size, False)

            x, adv_actor, a_one_hot, y_critic = self._get_actor_critic_error(batch)

            self.main_model_nn.train(x, y_critic, a_one_hot, adv_actor)

            self.buffer.reset_buffer()