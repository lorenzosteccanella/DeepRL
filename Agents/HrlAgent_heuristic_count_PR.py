import random
from Agents.HrlAgent import HrlAgent
from Utils import Edge, Node, Graph, KeyDict
from collections import deque
import math
import copy
import numpy as np

class HrlAgent_heuristic_count_PR(HrlAgent):

    options_executed_episode = []
    heuristic_reward = []
    as_visited = []
    counter_as = 0
    samples = []

    def act(self, s):

        self.pixel_manager_obs(s=s)

        node = Node(s["manager"], 0)
        self.graph.node_update(node)
        # if structure to reduce computation cost

        if self.current_node is None:
            self.current_node = self.graph.get_current_node()
            distances = self.graph.find_distances(self.current_node)
            self.distances_2_print.append(distances)
            self.best_option_action, self.best_edge = self.exploration_fn(self.current_node, distances)

            # print(self.current_node, node)
            # print(self.best_option_action, self.best_edge)
            # print()
            # print()

        elif self.current_node != node:
            self.current_node = self.graph.get_current_node()
            distances = self.graph.find_distances(self.current_node)
            self.distances_2_print.append(distances)
            self.best_option_action, self.best_edge = self.exploration_fn(self.current_node, distances)

            # print(self.current_node, node)
            # print(self.best_option_action, self.best_edge)
            # print()
            # print()

        #for option in self.options:
        #    print(option, len(option.get_edge_list()))#, option.get_edge_list())

        return self.best_option_action.act(s["option"])

    def update_option(self, sample):

        max_d = 10
        weight_heuristic_reward = 0.2

        s = sample[0]["option"]
        a = sample[1]
        r = sample[2]
        r_h_c = sample[2]
        s_ = sample[3]["option"]
        done = sample[4]
        info = sample[5]

        s_m = Node(sample[0]["manager"], 0)
        s_m_ = Node(sample[3]["manager"], 0)

        if s_m != s_m_:
            if self.target is not None:
                if s_m_ == self.target:
                    self.as_visited.append(s_m_)
                    self.counter_as = len(set(self.as_visited)) # are we sure we should count here?
                    #self.counter_as += 1
                    r += self.correct_option_end_reward
                    done = True

                    r_h_c = max(r, max(self.as_m2s_m[s_m].values(), key=lambda x: x[1])[1])

                else:
                    r += self.wrong_end_option_reward
                    done = True

                    if r < -1:
                        r = -1

                    r_h_c += self.wrong_end_option_reward

                    if r_h_c < -1:
                        r_h_c = -1

        # here u should take the old episode reward for that state

        #print(r_h_c)

        self.best_option_action.observe((s, a, r_h_c, s_, done, info))

        self.heuristic_reward.append(self.counter_as)
        self.samples.append((s, a, r, s_, done, info, s_m, s_m_))
        self.options_executed_episode.append(self.best_option_action)

        if self.counter_as >= max_d:

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

                    if done:   # this means that we are at the ending of an option
                        if KeyDict(s_) not in (self.as_m2s_m[s_m]):
                            self.as_m2s_m[s_m][KeyDict(s_)] = [copy.deepcopy(s_), r]
                        else:
                            self.as_m2s_m[s_m][KeyDict(s_)][1] = 0.8 * self.as_m2s_m[s_m][KeyDict(s_)][1] + 0.2 * r

                    #option.observe_imitation((s, a, r, s_, done, info))

                    del self.samples[i]
                    del self.options_executed_episode[i]
                    del self.heuristic_reward[-(i+1)]

                else:
                    break

        if sample[4]:

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

                if done:  # this means that we are at the ending of an option
                    if KeyDict(s_) not in (self.as_m2s_m[s_m]):
                        self.as_m2s_m[s_m][KeyDict(s_)] = [copy.deepcopy(s_), r]
                    else:
                        self.as_m2s_m[s_m][KeyDict(s_)][1] = 0.8 * self.as_m2s_m[s_m][KeyDict(s_)][1] + 0.2 * r

                #option.observe_imitation((s, a, r, s_, done, info))

            self.samples.clear()
            self.options_executed_episode.clear()
            self.as_visited.clear()
            self.counter_as = 0
            self.heuristic_reward.clear()

    def observe(self, sample):  # in (s, a, r, s_, done, info) format

        self.pixel_manager_obs(sample=sample)

        self.n_steps += 1
        self.total_r += sample[2]

        if sample[4]:
            self.total_r_2_print.append(self.total_r)
            self.total_r = 0

        self.graph.abstract_state_discovery(sample, self.target)

        edges_from_current_node = self.graph.get_edges_of_a_node(self.current_node)

        self.create_options(edges_from_current_node)
        self.update_option(sample)
        #self.update_imitation(sample)

        self.update_manager(sample)

        self.statistics_options(sample)

        # slowly decrease Epsilon based on manager experience
        if not self.equal(sample[0]["manager"], sample[3]["manager"]):
            self.manager_exp += 1
            self.epsilon = self.MIN_EPSILON + (1 - self.MIN_EPSILON) * math.exp(-self.LAMBDA * self.manager_exp)

        if sample[4]:
            self.current_node = None
            self.best_edge = None
            self.target = None
            self.n_episodes += 1

        return self.n_steps, self.n_episodes



