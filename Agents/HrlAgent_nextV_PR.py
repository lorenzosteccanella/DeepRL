import random
from Agents.HrlAgent import HrlAgent
from Utils import Edge, Node, Graph
from collections import deque
import math

class HrlAgent_nextV_PR(HrlAgent):

    intra_reward_dictionary = {}

    def update_option(self, sample):

        max_memory = 20

        s = sample[0]["option"]
        a = sample[1]
        r = sample[2]
        s_ = sample[3]["option"]
        done = sample[4]
        info = sample[5]

        s_m = Node(sample[0]["manager"], 0)
        s_m_ = Node(sample[3]["manager"], 0)

        if s_m != s_m_:
            if self.target is not None:
                if s_m_ == self.target:

                    key = s_m
                    if key not in self.intra_reward_dictionary:
                        sum_of_value = deque(maxlen=max_memory)
                        self.intra_reward_dictionary.update({s_m: [0, sum_of_value]}) # number of updates, sum_of_value, sum_of_squared_value

                    edges_from_current_node = self.graph.get_edges_of_a_node(self.target)

                    if len(edges_from_current_node)>0:

                        values = []
                        for i in range(len(edges_from_current_node)):
                            values.append(self.options[i].get_state_value(s_))

                        max_value = max(values)

                    else:
                        max_value = 0

                    self.intra_reward_dictionary[key][1].append(max_value)
                    self.intra_reward_dictionary[key][0] = len(self.intra_reward_dictionary[key][1])

                    if len(self.intra_reward_dictionary[key][1]) == max_memory:

                        # Min Max normalization

                        min_max_value = min(self.intra_reward_dictionary[key][1])
                        max_max_value = max(self.intra_reward_dictionary[key][1])

                        max_value_normalized = (max_value - min_max_value) / ((max_max_value - min_max_value) + 1e-7) # [0, 1] range
                        #max_value_normalized = (2 * max_value) - 1  # [-1, 1] range

                    else:
                        max_value_normalized=0

                    r += (self.correct_option_end_reward + (0.1 * max_value_normalized))

                    done = True

                else:

                    r += self.wrong_end_option_reward
                    done = True

                    if r < -1 :
                        r = -1

        self.best_option_action.observe((s, a, r, s_, done, info))



