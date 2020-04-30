import random
from Agents.HrlAgent import HrlAgent
from Utils import Edge, Node, Graph, KeyDict
from collections import deque
import math
import copy
import numpy as np

class HrlAgent_heuristic_count_PR(HrlAgent):

    """
    Inherits from HrlAgent, and augment HrlAgent class including a strategy for the end termination state
    """

    options_executed_episode = []     # a list just to keep in memory all the options executed
    heuristic_reward = []             # a list to keep in memory all the heuristic count reward in the episode
    as_visited = []                   # a list to include all the abstract state visited
    counter_as = 0
    samples = []                      # a list to collect the samples of the episode
    as_m2s_m = {}                     # a dictinory to keep in memory for each abstract state all the possible ending state and relative values

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

        """
        overwrite of update_option in HrlAgent now we include as well heuristic count pseudo reward

        Args:
            sample: a (s, a, r, s', done, info) tuple to train the option on
        """

        max_d = 10                                                      # max number of abstract states to reach we are interested in
        weight_heuristic_reward = 0.2                                   # a weight value to decide how much the heuristic count reward value is worth

        s = sample[0]["option"]
        a = sample[1]
        r = sample[2]
        r_h_c = sample[2]                                               # a reward based on heuristic count
        s_ = sample[3]["option"]
        done = sample[4]
        info = sample[5]

        s_m = Node(sample[0]["manager"], 0)
        s_m_ = Node(sample[3]["manager"], 0)

        if s_m != s_m_:
            if self.target is not None:
                if s_m_ == self.target:                                 # if we ended correctly
                    self.as_visited.append(s_m_)                        # we add the abstract state to the list of abstract statas reached
                    self.counter_as = len(set(self.as_visited))         # we count how many singular abstract state we reached
                    r += self.correct_option_end_reward                 # we augment the reward with the correct end reward
                    done = True                                         # done is True we finished with the option

                    if KeyDict(s_) not in (self.as_m2s_m[s_m]):         # we check if we are in ended state already encountered
                        r_h_c = r
                    else:                                               # we already visited this ending state
                        r_h_c = self.as_m2s_m[s_m][KeyDict(s_)][1]

                else:                                                   # we endend wrongly
                    r += self.wrong_end_option_reward
                    done = True

                    if r < -1:
                        r = -1

                    r_h_c += self.wrong_end_option_reward

                    if r_h_c < -1:
                        r_h_c = -1

        self.best_option_action.observe((s, a, r_h_c, s_, done, info))

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
                            self.as_m2s_m[s_m][KeyDict(s_)][1] = 0.8 * self.as_m2s_m[s_m][KeyDict(s_)][1] + 0.2 * r

                    del self.samples[i]
                    del self.options_executed_episode[i]
                    del self.heuristic_reward[-(i+1)]

                else:
                    break

        if sample[4]:                                                   # if we reached the end of episode we can start calculate the heuristic count pseudo reward

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
                        self.as_m2s_m[s_m][KeyDict(s_)][1] = 0.8 * self.as_m2s_m[s_m][KeyDict(s_)][1] + 0.2 * r

            self.samples.clear()
            self.options_executed_episode.clear()
            self.as_visited.clear()
            self.counter_as = 0
            self.heuristic_reward.clear()

    def observe(self, sample):

        """
        Overwrite of HrlAgent observe to include pixel_manager_obs

        Args:
            sample: a (s, a, r, s', done, info) tuple
        """

        self.pixel_manager_obs(sample=sample)

        return super().observe(sample)



