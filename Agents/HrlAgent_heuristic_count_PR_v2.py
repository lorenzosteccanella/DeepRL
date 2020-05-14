import random
from Agents.HrlAgent import HrlAgent
from Utils import Edge, Node, Graph, KeyDict
from collections import deque
import math
import copy
import numpy as np

class HrlAgent_heuristic_count_PR_v2(HrlAgent):

    """
    Inherits from HrlAgent, and augment HrlAgent class including a strategy for the end termination state
    """

    options_executed_episode = []     # a list just to keep in memory all the options executed
    heuristic_reward = []             # a list to keep in memory all the heuristic count reward in the episode
    as_visited = []                   # a list to include all the abstract state visited
    counter_as = 0
    samples = []                      # a list to collect the samples of the episode
    as_m2s_m = {}                     # a dictinory to keep in memory for each abstract state all the possible ending state and relative values
    n_steps_option = 0
    HER_experience_batch = []
    update_heuristic_count_values = False

    def pixel_manager_obs(self, s = None, sample = None):

        """
        a function to populate the dictionary "as_m2s_m" for each asbtract state we save all possible ending states

        Args:
            s : an abstract state
            sample: a (s, a, r, s', done, info) tuple
        """

        if s is not None:
            s = copy.deepcopy(s)
            node = Node(s["manager"], 0)

            if node not in self.as_m2s_m:
                self.as_m2s_m[node] = {}

        if sample is not None:

            sample = copy.deepcopy(sample)
            node1 = Node(sample[0]["manager"], 0)
            node2 = Node(sample[3]["manager"], 0)

            if node1 not in self.as_m2s_m:
                self.as_m2s_m[node1] = {}

            if node2 not in self.as_m2s_m:
                self.as_m2s_m[node2] = {}

    def act(self, s):

        """
        Overwrite of act HrlAgent to include pixel_manager_obs

        Args:
            s : the observation from the environment

        Returns:
            returns the option to perform
        """

        self.pixel_manager_obs(s=s)

        return super().act(s)

    def update_option(self, sample):

        self.update_option_online_version(sample)

    def HER_training(self, s, a, s_, info, s_m, s_m_):
        # HER style training
        edge = Edge(s_m, s_m_)
        option = self.get_option(edge)
        if KeyDict(s_) not in (self.as_m2s_m[s_m]):  # we check if we are in ended state already encountered
            HER_r = self.correct_option_end_reward
        else:  # we already visited this ending state
            HER_r = self.as_m2s_m[s_m][KeyDict(s_)][1]
        HER_done = True
        for sample in self.HER_experience_batch:
            option.observe_imitation(sample)
        option.observe_imitation((s, a, HER_r, s_, HER_done, info))

    def HER_training_wo_heuristic(self, s, a, s_, info, s_m, s_m_):
        # HER style training
        edge = Edge(s_m, s_m_)
        option = self.get_option(edge)
        HER_r = self.correct_option_end_reward
        HER_done = True
        for sample in self.HER_experience_batch:
            option.observe_imitation(sample)
        option.observe_imitation((s, a, HER_r, s_, HER_done, info))

    def update_option_offline_version(self, sample):

        max_d = 10
        weight_heuristic_reward = 0.2

        s = sample[0]["option"]
        a = sample[1]
        r = min(sample[2], 0.)                                 # remember this!!!!!
        s_ = sample[3]["option"]
        done = sample[4]
        info = sample[5]

        s_m = Node(sample[0]["manager"], 0)
        s_m_ = Node(sample[3]["manager"], 0)

        self.n_steps_option +=1

        if s_m != s_m_:
            if self.target is not None:
                if s_m_ == self.target:
                    self.as_visited.append(s_m_)
                    self.counter_as = len(set(self.as_visited))  # are we sure we should count here?
                    # self.counter_as += 1

                    r = self.correct_option_end_reward
                    done = True

                else:
                    self.HER_training_wo_heuristic(s, a, s_, info, s_m, s_m_)
                    self.counter_as = 0
                    self.as_visited.clear()
                    self.update_heuristic_count_values = True

                    r = self.wrong_end_option_reward
                    done = True
            else:
                self.HER_training_wo_heuristic(s, a, s_, info, s_m, s_m_)
                self.counter_as = 0
                self.as_visited.clear()
                self.update_heuristic_count_values = True

            self.HER_experience_batch.clear()                       # I'm transitioning from abstract states clear the offline experience batch
        else:
            self.HER_experience_batch.append((s, a, r, s_, done, info))

        if self.n_steps_option > 100:
            self.replan = True
            r = self.wrong_end_option_reward
            done = True

        if done:
            self.n_steps_option = 0

        self.heuristic_reward.append(self.counter_as)
        self.samples.append((s, a, r, s_, done, info))
        self.options_executed_episode.append(self.best_option_action)

        if self.counter_as >= max_d:

            rewards_h = [(min(element, max_d) / max_d) for element in self.heuristic_reward[::-1]]

            for i, p_sample, option in zip(range(len(self.samples)), self.samples, self.options_executed_episode):
                if rewards_h[i] == 1:
                    s = p_sample[0]
                    a = p_sample[1]
                    r = p_sample[2] + weight_heuristic_reward * rewards_h[i] if p_sample[2]>0 and p_sample[4]  else p_sample[2]
                    s_ = p_sample[3]
                    done = p_sample[4]
                    info = p_sample[5]
                    option.observe((s, a, r, s_, done, info))

                    del self.samples[i]
                    del self.options_executed_episode[i]
                    del self.heuristic_reward[-(i+1)]

                else:

                    break

        if sample[4] or self.update_heuristic_count_values is True:

            rewards_h = [(min(element, max_d) / max_d) for element in self.heuristic_reward[::-1]]

            for i, p_sample, option in zip(range(len(self.samples)), self.samples, self.options_executed_episode):
                s = p_sample[0]
                a = p_sample[1]
                r = p_sample[2] + weight_heuristic_reward * rewards_h[i] if p_sample[2]>0 and p_sample[4]  else p_sample[2]
                s_ = p_sample[3]
                done = p_sample[4]
                info = p_sample[5]
                option.observe((s, a, r, s_, done, info))

            self.samples.clear()
            self.options_executed_episode.clear()
            self.as_visited.clear()
            self.counter_as = 0
            self.heuristic_reward.clear()
            self.update_heuristic_count_values = False

    def update_option_online_version(self, sample):

        """
        overwrite of update_option in HrlAgent now we include as well heuristic count pseudo reward

        Args:
            sample: a (s, a, r, s', done, info) tuple to train the option on
        """

        max_d = 10                                                      # max number of abstract states to reach we are interested in
        weight_heuristic_reward = 0.2                                   # a weight value to decide how much the heuristic count reward value is worth

        s = sample[0]["option"]
        a = sample[1]
        r = min(sample[2], 0.)                                 # remember this!!!!!
        r_h_c = r                                              # a reward based on heuristic count
        s_ = sample[3]["option"]
        done = sample[4]
        info = sample[5]

        s_m = Node(sample[0]["manager"], 0)
        s_m_ = Node(sample[3]["manager"], 0)

        self.n_steps_option += 1

        if s_m != s_m_:
            if self.target is not None:
                if s_m_ == self.target:                                 # if we ended correctly
                    self.as_visited.append(s_m_)  # we add the abstract state to the list of abstract statas reached
                    self.counter_as = len(set(self.as_visited))  # we count how many singular abstract state we reached
                    r = self.correct_option_end_reward                  # we augment the reward with the correct end reward
                    done = True                                         # done is True we finished with the option

                    if KeyDict(s_) not in (self.as_m2s_m[s_m]):         # we check if we are in ended state already encountered
                        r_h_c = r
                    else:                                               # we already visited this ending state
                        r_h_c = self.as_m2s_m[s_m][KeyDict(s_)][1]

                else:                                                   # we endend wrongly
                    r = self.wrong_end_option_reward
                    done = True

                    r_h_c = self.wrong_end_option_reward

                    self.HER_training_wo_heuristic(s, a, s_, info, s_m, s_m_)
                    self.counter_as = 0
                    self.as_visited.clear()
                    self.update_heuristic_count_values = True
            else:
                self.HER_training_wo_heuristic(s, a, s_, info, s_m, s_m_)
                self.counter_as = 0
                self.as_visited.clear()
                self.update_heuristic_count_values = True

            self.HER_experience_batch.clear()  # I'm transitioning from abstract states clear the offline experience batch

        else:
            self.HER_experience_batch.append((s, a, r, s_, done, info))

        if self.n_steps_option > 100:
            self.replan = True
            r = min(self.wrong_end_option_reward, r)
            done = True

            if r < -1:
                r = -1

            r_h_c += self.wrong_end_option_reward

            if r_h_c < -1:
                r_h_c = -1

        if done:
            self.n_steps_option = 0
            print(self.best_option_action, r_h_c)

        self.option_rewards += r_h_c
        self.best_option_action.observe_online((s, a, r_h_c, s_, done, info))
        self.best_option_action.observe_imitation((s, a, r, s_, done, info))

        self.heuristic_reward.append(self.counter_as)
        self.samples.append((s, a, r, s_, done, info, s_m, s_m_))
        self.options_executed_episode.append(self.best_option_action)

        if self.counter_as >= max_d:                                   # if we reached the number of abstract state we are interested in we can start calculate the heuristic count pseudo reward

            rewards_h = [(min(element, max_d) / max_d) for element in self.heuristic_reward[::-1]]

            for i, p_sample, option in zip(range(len(self.samples)), self.samples, self.options_executed_episode):
                if rewards_h[i] == 1:
                    s = p_sample[0]
                    a = p_sample[1]
                    r = p_sample[2] + weight_heuristic_reward * rewards_h[i] if p_sample[2]>0 and p_sample[4] else p_sample[2]
                    s_ = p_sample[3]
                    done = p_sample[4]
                    info = p_sample[5]
                    s_m = p_sample[6]
                    s_m_ = p_sample[7]

                    if done and r >= self.correct_option_end_reward:  # this means that we are at the correct end of an option
                        if KeyDict(s_) not in (self.as_m2s_m[s_m]):
                            self.as_m2s_m[s_m][KeyDict(s_)] = [copy.deepcopy(s_), r]
                        else:
                            # if self.as_m2s_m[s_m][KeyDict(s_)][1] < r:
                            #     self.as_m2s_m[s_m][KeyDict(s_)][1] = r

                            self.as_m2s_m[s_m][KeyDict(s_)][1] = 0.6 * self.as_m2s_m[s_m][KeyDict(s_)][1] + 0.4 * r    # changed for max let's see
                            #self.as_m2s_m[s_m][KeyDict(s_)][1] = r

                    del self.samples[i]
                    del self.options_executed_episode[i]
                    del self.heuristic_reward[-(i+1)]

                else:
                    break

        if sample[4] or self.update_heuristic_count_values is True:     # if we reached the end of episode we can start calculate the heuristic count pseudo reward

            rewards_h = [(min(element, max_d) / max_d) for element in self.heuristic_reward[::-1]]

            for i, p_sample, option in zip(range(len(self.samples)), self.samples, self.options_executed_episode):
                s = p_sample[0]
                a = p_sample[1]
                r = p_sample[2] + weight_heuristic_reward * rewards_h[i] if p_sample[2]>0 and p_sample[4] else p_sample[2]
                s_ = p_sample[3]
                done = p_sample[4]
                info = p_sample[5]
                s_m = p_sample[6]
                s_m_ = p_sample[7]

                if done and r >= self.correct_option_end_reward:        # this means that we are at the correct end of an option
                    if KeyDict(s_) not in (self.as_m2s_m[s_m]):
                        self.as_m2s_m[s_m][KeyDict(s_)] = [copy.deepcopy(s_), r]
                    else:
                        # if self.as_m2s_m[s_m][KeyDict(s_)][1] < r:
                        #     self.as_m2s_m[s_m][KeyDict(s_)][1] = r

                        self.as_m2s_m[s_m][KeyDict(s_)][1] = 0.6 * self.as_m2s_m[s_m][KeyDict(s_)][1] + 0.4 * r    # changed for max let's see

                        #self.as_m2s_m[s_m][KeyDict(s_)][1] = r

            self.samples.clear()
            self.options_executed_episode.clear()
            self.as_visited.clear()
            self.counter_as = 0
            self.heuristic_reward.clear()
            self.update_heuristic_count_values = False

    def observe(self, sample):

        """
        Overwrite of HrlAgent observe to include pixel_manager_obs

        Args:
            sample: a (s, a, r, s', done, info) tuple
        """

        self.pixel_manager_obs(sample=sample)

        return super().observe(sample)


