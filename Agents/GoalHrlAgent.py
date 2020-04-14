import random
from Agents.HrlAgent import HrlAgent
from Utils import Edge, Node, Graph
from collections import deque
import math
import copy

class GoalHrlAgent(HrlAgent):

    def pixel_manager_obs(self, s = None, sample = None):
        if s is not None:

            s = copy.deepcopy(s)

            if s["manager"] not in self.as_m2s_m:
                self.as_m2s_m[s["manager"]] = (copy.deepcopy(s["option"]), 0.)

        if sample is not None:

            sample = copy.deepcopy(sample)

            if sample[0]["manager"] not in self.as_m2s_m:
                self.as_m2s_m[sample[0]["manager"]] = (copy.deepcopy(sample[0]["option"]), 0.)

            if sample[3]["manager"] not in self.as_m2s_m:
                self.as_m2s_m[sample[3]["manager"]] = (copy.deepcopy(sample[3]["option"]), 0.)

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

        elif self.current_node != node:
           self.current_node = self.graph.get_current_node()
           distances = self.graph.find_distances(self.current_node)
           self.distances_2_print.append(distances)
           self.best_option_action, self.best_edge = self.exploration_fn(self.current_node, distances)

        if self.target is not None:
            start = self.as_m2s_m[self.current_node.state][0]
        else:
            start = None

        if self.target is not None:
            goal = self.as_m2s_m[self.target.state][0]
        else:
            goal = None

        return self.best_option_action.act([s["option"], start, goal])

    def update_option(self, sample):
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

                    r += self.correct_option_end_reward
                    done = True                           # single network now, so maybe done = false here

                    if r > self.as_m2s_m[s_m_.state][1]:

                        self.as_m2s_m[s_m_.state] = (copy.deepcopy(s_), r)

                else:

                    r += self.wrong_end_option_reward
                    done = True                           # single network now, so maybe done = false here

        self.best_option_action.observe((s, a, r, s_, done, info, start, goal))

    def observe(self, sample):  # in (s, a, r, s_, done, info) format

        self.pixel_manager_obs(sample=sample)

        self.n_steps += 1
        self.total_r += sample[2]

        if sample[4]:
            self.total_r_2_print.append(self.total_r)
            self.total_r = 0

        self.graph.abstract_state_discovery(sample, self.target)

        if self.RESET_EXPLORATION_WHEN_NEW_NODE:
            if self.graph.new_node_encontered:
                print("new node discovered, resetting the exploration!!!")
                self.reset_exploration()

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

        #if sample[4]:
        #    self.path_2_print.clear()
        #    self.distances_2_print.clear()

        if sample[4]:
            self.current_node = None
            self.best_edge = None
            self.target = None
            self.n_episodes += 1

        return self.n_steps, self.n_episodes



