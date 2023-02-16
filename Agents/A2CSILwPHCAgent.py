import numpy as np
from Agents.AbstractAgent import AbstractAgent
from Utils import ExperienceReplay, PrioritizedExperienceReplay
import collections
import math

class A2CSILwPHCAgent(AbstractAgent):
    id = 0

    def __init__(self, action_space, main_model_nn, gamma, batch_size, sil_batch_size, imitation_buffer_size, imitation_learnig_steps):

        self.batch_size = batch_size
        self.imitation_buffer_size = imitation_buffer_size
        self.buffer_online = ExperienceReplay(self.batch_size)
        self.buffer_imitation = PrioritizedExperienceReplay(self.imitation_buffer_size)
        self.trajectory = []
        self.action_space = action_space
        self.main_model_nn = main_model_nn
        self.gamma = gamma
        self.imitation_learning_steps = imitation_learnig_steps
        self.sil_batch_size = sil_batch_size
        self.ce_loss = None
        self.id = A2CSILwPHCAgent.id
        A2CSILwPHCAgent.id += 1
        self.S = {}
        self.beta = 0.2

        self.correct_termination = collections.deque(maxlen = 10)
        self.max_return_reward = 1000 # WARNING WARNING WARNING WARNING this depends on the correct termination reward u set!!!!

        self.n_steps = 0
        self.n_episodes = 0

    def update_correct_termination(self, reward, done):
        if done:
            self.correct_termination.append(reward)
            # if self.max_return_reward < reward:
            #     self.max_return_reward = reward

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

        if s not in self.S:
            self.S[s] = 1
        else:
            self.S[s] += 1

        predict = self.main_model_nn.prediction_actor([s])[0]
        a = np.random.choice(self.action_space, p=predict)

        if sum(self.correct_termination) == (10 * self.max_return_reward):
            print("Become Deterministic, option n ", self.id)
            i_a = np.argmax(predict)
            a = self.action_space[i_a]

        return a

    def observe(self, sample):

        self.update_correct_termination(sample[2], sample[4])

        self.n_steps += 1
        s = sample[0]
        a = sample[1]
        s_ = sample[3]

        if s not in self.S:
            self.S[s] = 1

        h_c_r = sample[2] + (self.beta/math.sqrt(self.S[s]))

        self.buffer_online.add((s, a, h_c_r, s_, sample[4], sample[5]))
        self.add_multy_trajectory_memory(sample)

        if sample[4]:
            self.n_episodes += 1

        return self.n_steps, self.n_episodes

    def observe_online(self, sample):

        self.n_steps += 1

        self.buffer_online.add(sample)

        if sample[4]:
            self.n_episodes += 1

        return self.n_steps, self.n_episodes

    def observe_imitation(self, sample):
        self.add_multy_trajectory_memory(sample)

    def add_multy_trajectory_memory(self, sample):
        self.trajectory.append((sample[0], sample[1], sample[2]))
        if sample[4]:
            s = np.array([o[0] for o in self.trajectory])
            a = np.array([o[1] for o in self.trajectory])
            r = np.array([o[2] for o in self.trajectory])
            discounted_rewards = self._discount_rewards(r, self.gamma)
            for i in range(len(discounted_rewards)):
                self.buffer_imitation.add((s[i], a[i], discounted_rewards[i]))

            self.trajectory.clear()

    def add_single_trajectory_memory(self, sample): # in (s, a, r, s_) format
        self.trajectory.append([sample[0], sample[1], sample[2]])

        if sample[4]:
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

    def train_imitation(self):
        if self.buffer_imitation.buffer_len() >= self.sil_batch_size:
            batch_imitation, imp_w = self.buffer_imitation.sample(self.sil_batch_size, True)
            x, adv_actor, a_one_hot, y_critic = self._get_imitation_error(batch_imitation)

            self.main_model_nn.train_imitation(x, y_critic, a_one_hot, adv_actor, imp_w)

            # update errors
            for k in range(len(batch_imitation)):
                idx = batch_imitation[k][0]
                self.buffer_imitation.update(idx, adv_actor[k])

    def train_single_trajectory_imitation(self):
        batch_imitation, imp_w = self.buffer_imitation.sample(self.buffer_imitation.buffer_len(), True)
        x, adv_actor, a_one_hot, y_critic = self._get_imitation_error(batch_imitation)

        self.main_model_nn.train_imitation(x, y_critic, a_one_hot, adv_actor, imp_w)

        # update errors
        for k in range(len(batch_imitation)):
            idx = batch_imitation[k][0]
            self.buffer_imitation.update(idx, adv_actor[k])

        self.buffer_imitation.reset_buffer()


    def replay(self, done):
        if sum(self.correct_termination) != (10 * self.max_return_reward):
            if self.buffer_online.buffer_len() >= self.batch_size:

                batch, imp_w = self.buffer_online.sample(self.batch_size, False)
                x, adv_actor, a_one_hot, y_critic = self._get_actor_critic_error(batch)

                _, __, self.ce_loss = self.main_model_nn.train(x, y_critic, a_one_hot, adv_actor)

                self.buffer_online.reset_buffer()

                for i in range(self.imitation_learning_steps):
                    self.train_imitation()

            elif done is True:

                batch, imp_w = self.buffer_online.sample(self.batch_size, False)
                x, adv_actor, a_one_hot, y_critic = self._get_actor_critic_error(batch)

                _, __, self.ce_loss = self.main_model_nn.train(x, y_critic, a_one_hot, adv_actor)

                self.buffer_online.reset_buffer()

                for i in range(self.imitation_learning_steps):
                    self.train_imitation()

            else:
                self.ce_loss = None

    def replay_imitation(self, done):
        if done:
            if sum(self.correct_termination) < (10 * self.max_return_reward):
                for i in range(self.imitation_learning_steps):
                    self.train_imitation()

    def set_name_file_2_save(self, filename):
        self.FILE_NAME = filename + " - "

    def reset(self):

        self.n_steps = 0
        self.n_episodes = 0
        del self.buffer_online
        del self.buffer_imitation
        self.buffer_online = ExperienceReplay(self.batch_size)
        self.buffer_imitation = PrioritizedExperienceReplay(self.imitation_buffer_size)
        self.S.clear()
