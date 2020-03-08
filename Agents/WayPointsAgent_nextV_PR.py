import random
from Agents.HrlAgent import HrlAgent
from Utils import Edge, Node, Graph
from collections import deque
import math

class WayPointsAgent_nextV_PR(HrlAgent):

    def act(self, s):
        node = Node(s["manager"], 0)
        self.graph.node_update(node)

        if self.current_node is None:
            self.current_node = self.graph.get_current_node()
            distances = self.graph.find_distances(self.current_node)
            self.distances_2_print.append(distances)
            self.best_option_action = self.exploration_fn(distances)

        if self.target:

            if self.current_node == self.target:
               self.current_node = self.graph.get_current_node()
               distances = self.graph.find_distances(self.current_node)
               self.distances_2_print.append(distances)
               self.best_option_action = self.exploration_fn(distances)

        return self.best_option_action.act(s["option"])

    intra_reward_dictionary = {}

    def hash_function(self, a):
        if type(a).__name__ == "ndarray":
            return hash(a.tostring())
        else:
            return hash(a)

    def update_option(self, sample):

        max_memory = 20

        s = sample[0]["option"]
        a = sample[1]
        r = sample[2]
        s_ = sample[3]["option"]
        done = sample[4]
        info = sample[5]

        if sample[0]["manager"] != sample[3]["manager"]:
            if self.target is not None:
                if sample[3]["manager"] == self.target.state:

                    key = self.hash_function(sample[0]["manager"])
                    if key not in self.intra_reward_dictionary:
                        sum_of_value = deque(maxlen=max_memory)
                        self.intra_reward_dictionary.update({self.hash_function(sample[0]["manager"]): [0, sum_of_value]}) # number of updates, sum_of_value, sum_of_squared_value

                    # to keep performance statistics
                    self.number_of_options_executed += 1
                    self.number_of_successfull_option += 1

                    edges_from_current_node = self.graph.get_edges_of_a_node(self.target)

                    if len(edges_from_current_node)>0:
                        values = []
                        for i in range(len(edges_from_current_node)):
                            if i in range(len(self.options)):
                                values.append(self.options[i].get_state_value(s_))

                        max_value = max(values)

                    else:
                        max_value = 0

                    # let's update the dictionary first

                    self.intra_reward_dictionary[key][1].append(max_value)
                    self.intra_reward_dictionary[key][0] = len(self.intra_reward_dictionary[key][1])

                    if len(self.intra_reward_dictionary[key][1]) == max_memory:

                        # Min Max normalization

                        min_max_value = min(self.intra_reward_dictionary[key][1])
                        max_max_value = max(self.intra_reward_dictionary[key][1])

                        max_value_normalized = (max_value - min_max_value) / ((max_max_value - min_max_value) + 1e-7) # [0, 1] range

                    else:
                        max_value_normalized=0

                    r += (self.correct_option_end_reward + (0.1 * max_value_normalized))
                    done = True

        self.save_statistics()

        self.best_option_action.observe((s, a, r, s_, done, info))



