import numpy as np
from Agents.AbstractAgent import AbstractAgent
from Utils import ExperienceReplay, PrioritizedExperienceReplay

class GoalA2CSILAgent(AbstractAgent):

    def __init__(self, action_space, main_model_nn, gamma, batch_size, sil_batch_size, imitation_buffer_size, imitation_learnig_steps):

        self.batch_size = batch_size
        self.buffer_online = ExperienceReplay(self.batch_size)
        self.buffer_imitation = PrioritizedExperienceReplay(imitation_buffer_size)   # WARNING WARNING WARNING WARNING Experience Replay instead of Prioritized Experience Replay
        self.trajectory = []
        self.action_space = action_space
        self.main_model_nn = main_model_nn
        self.gamma = gamma
        self.imitation_learning_steps = imitation_learnig_steps
        self.sil_batch_size = sil_batch_size
        self.ce_loss = None

    def _get_actor_critic_error(self, batch):

        goals = np.array([o[1][7] for o in batch])
        starts = np.array([o[1][6] for o in batch])
        states_t = np.array([o[1][0] for o in batch])
        p = self.main_model_nn.prediction_critic(states_t, starts, goals)[:, 0]
        a_one_hot = np.zeros((len(batch), len(self.action_space)))
        dones = np.zeros((len(batch)))
        rewards = np.zeros((len(batch)))

        for k in range(len(batch)):
            o = batch[k][1]
            a = o[1]
            r = o[2]
            s_ = o[3]
            done = o[4]
            i = o[6]
            g = o[7]

            a_index = self.action_space.index(a)

            if done:
                dones[k] = 1
                p_ = [0]
            elif k == len(batch)-1:
                p_ = self.main_model_nn.prediction_critic([s_], [i], [g])[0]
            rewards[k] = r
            a_one_hot[k][a_index] = 1

        y_critic, adv_actor = self._returns_advantages(rewards, dones, p, p_)
        y_critic = np.expand_dims(y_critic, axis=-1)
        return states_t, starts, goals, adv_actor, a_one_hot, y_critic

    def _get_imitation_error(self, batch):

        goals = np.array([o[1][4] for o in batch])
        starts = np.array([o[1][3] for o in batch])
        states_t = np.array([o[1][0] for o in batch])
        p = self.main_model_nn.prediction_critic(states_t, starts, goals)[:, 0]   # not sure about the indexes here
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

        return states_t, starts, goals, clip_advantages, a_one_hot, y_critic

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

    def act(self, s, start, goal):
        predict = self.main_model_nn.prediction_actor([s], [start], [goal])[0]

        return np.random.choice(self.action_space, p=predict)

    def observe(self, sample):
        self.buffer_online.add(sample)
        self.add_multy_trajectory_memory(sample)

    def add_multy_trajectory_memory(self, sample):
        self.trajectory.append((sample[0], sample[1], sample[2], sample[6], sample[7]))
        if sample[4]:
            s = np.array([o[0] for o in self.trajectory])
            a = np.array([o[1] for o in self.trajectory])
            r = np.array([o[2] for o in self.trajectory])
            starts = np.array([o[3] for o in self.trajectory])
            goals = np.array([o[4] for o in self.trajectory])
            discounted_rewards = self._discount_rewards(r, self.gamma)
            for i in range(len(discounted_rewards)):
                self.buffer_imitation.add((s[i], a[i], discounted_rewards[i], starts[i], goals[i]))

            self.trajectory.clear()

    def replay(self):
        if self.buffer_online.buffer_len() >= self.batch_size:

            batch, imp_w = self.buffer_online.sample(self.batch_size, False)
            x, s, g, adv_actor, a_one_hot, y_critic = self._get_actor_critic_error(batch)

            _, __, self.ce_loss = self.main_model_nn.train(x, s, g, y_critic, a_one_hot, adv_actor)

            self.buffer_online.reset_buffer()

            if self.buffer_imitation.buffer_len() >= self.sil_batch_size:
                for i in range(self.imitation_learning_steps):
                    batch_imitation, imp_w = self.buffer_imitation.sample(self.sil_batch_size)
                    x, s, g, adv_actor, a_one_hot, y_critic = self._get_imitation_error(batch_imitation)

                    self.main_model_nn.train_imitation(x, s, g, y_critic, a_one_hot, adv_actor, imp_w)

                    # update errors
                    for k in range(len(batch_imitation)):
                        idx = batch_imitation[k][0]
                        self.buffer_imitation.update(idx, adv_actor[k])
