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
            start = self.current_node.state
        else:
            start = None

        if self.target is not None:
            goal = self.target.state
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
            start = s_m.state
        else:
            start = None

        if self.target is not None:
            goal = self.target.state
        else:
            goal = None

        if s_m != s_m_:
            if self.target is not None:
                if s_m_ == self.target:

                    r += self.correct_option_end_reward
                    done = True                           # single network now, so maybe done = false here

                else:

                    r += self.wrong_end_option_reward
                    done = True                           # single network now, so maybe done = false here

        self.best_option_action.observe((s, a, r, s_, done, info, start, goal))



