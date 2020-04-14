import random
from Agents.GoalHrlAgent import GoalHrlAgent
from Utils import Edge, Node, Graph
from collections import deque
import math
import copy

class GoalHrlAgent_heuristic_count_PR(GoalHrlAgent):

    options_executed_episode = []
    heuristic_reward = []
    counter_as = 0
    samples = []

    def update_option(self, sample):

        max_d = 10
        weight_heuristic_reward = 0.2

        s = sample[0]["option"]
        a = sample[1]
        r = sample[2]
        s_ = sample[3]["option"]
        done = sample[4]
        info = sample[5]

        s_m = Node(sample[0]["manager"], 0)
        s_m_ = Node(sample[3]["manager"], 0)

        if self.target is not None:
            start = self.as_m2s_m[s_m.state][0]
        else:
            start = None

        if self.target is not None:
            goal = self.as_m2s_m[self.target.state][0]
        else:
            goal = None

        if s_m != s_m_:
            if self.target is not None:
                if s_m_ == self.target:
                    self.counter_as += 1  # are we sure we should count here?
                    r += self.correct_option_end_reward
                    done = True

                else:

                    r += self.wrong_end_option_reward
                    done = True

        self.heuristic_reward.append(self.counter_as)
        self.samples.append((s, a, r, s_, done, info, start, goal, s_m, s_m_))
        self.options_executed_episode.append(self.best_option_action)

        if self.counter_as >= max_d:

            rewards_h = [(min(element, max_d) / max_d) for element in self.heuristic_reward[::-1]]

            for i, p_sample, option in zip(range(len(self.samples)), self.samples, self.options_executed_episode):
                if rewards_h[i] == 1:
                    s = p_sample[0]
                    a = p_sample[1]
                    r = p_sample[2] + weight_heuristic_reward * rewards_h[i] if p_sample[2] > 0 and p_sample[4] else p_sample[2]
                    s_ = p_sample[3]
                    done = p_sample[4]
                    info = p_sample[5]
                    start = p_sample[6]
                    goal = p_sample[7]
                    s_m = p_sample[8]
                    s_m_ = p_sample[9]

                    option.observe((s, a, r, s_, done, info, start, goal))
                    if done:
                        if r > self.as_m2s_m[s_m_.state][1]:
                            self.as_m2s_m[s_m_.state] = (copy.deepcopy(s_), r)

                    del self.samples[i]
                    del self.options_executed_episode[i]
                    del self.heuristic_reward[-(i+1)]

        if sample[4]:

            rewards_h = [ (min(element, max_d) / max_d) for element in self.heuristic_reward[::-1] ]

            for i, p_sample, option in zip(range(len(self.samples)), self.samples, self.options_executed_episode):
                s = p_sample[0]
                a = p_sample[1]
                r = p_sample[2] + weight_heuristic_reward * rewards_h[i] if p_sample[2]>0 and p_sample[4] else p_sample[2]
                s_ = p_sample[3]
                done = p_sample[4]
                info = p_sample[5]
                start = p_sample[6]
                goal = p_sample[7]
                s_m = p_sample[8]
                s_m_ = p_sample[9]

                option.observe((s, a, r, s_, done, info, start, goal))
                if done:
                    if r > self.as_m2s_m[s_m_.state][1]:
                        self.as_m2s_m[s_m_.state] = (copy.deepcopy(s_), r)

            self.samples.clear()
            self.options_executed_episode.clear()
            self.counter_as = 0
            self.heuristic_reward.clear()



