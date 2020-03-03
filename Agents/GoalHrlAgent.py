import random
from Agents.HrlAgent import HrlAgent
from Utils import Edge, Node, Graph
from collections import deque
import math

class GoalHrlAgent(HrlAgent):

    def act(self, s):
        node = Node(s["manager"], 0)
        self.graph.node_update(node)
        # if structure to reduce computation cost

        # if self.target is not None:
        #   if self.current_node != node:
        #       self.graph.print_node_list()
        #       print(self.current_node.state, "->", self.target.state, " - ", self.best_option_action)

        if self.current_node is None:
            self.current_node = self.graph.get_current_node()
            distances = self.graph.find_distances(self.current_node)
            self.distances_2_print.append(distances)
            self.best_option_action = self.exploration_fn(distances)
        elif self.current_node != node:
            self.current_node = self.graph.get_current_node()
            distances = self.graph.find_distances(self.current_node)
            self.distances_2_print.append(distances)
            self.best_option_action = self.exploration_fn(distances)

        if self.target is not None:
            goal = self.target.state
        else:
            goal = None

        return self.best_option_action.act([s["option"], goal])

    def update_option(self, sample):
        s = sample[0]["option"]
        a = sample[1]
        r = sample[2]
        s_ = sample[3]["option"]
        done = sample[4]
        info = sample[5]

        if self.target is not None:
            goal = self.target.state
        else:
            goal = None

        if not self.equal(sample[0]["manager"], sample[3]["manager"]):
            if self.target is not None:
                if self.equal(sample[3]["manager"], self.target.state):

                    # to keep performance statistics
                    self.number_of_options_executed += 1
                    self.number_of_successfull_option += 1

                else:
                    # to keep performance statistics
                    self.number_of_options_executed += 1

                    r += self.wrong_end_option_reward
                    done = True

        self.save_statistics()

        self.best_option_action.observe((s, a, r, s_, done, info, goal))



