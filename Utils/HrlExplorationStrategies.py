import random

def get_epsilon_count_exploration(self, current_node, distances):
    """
    this is the function that choose the option to execute with an epsilon exploration strategy, it differ from regular
    epsilon - greedy because we are keeping a different epsilon value for each abstract state

    Args:
        current_node: the abstract state we are in
        distances: the Q values or other values defining the utility of each edge
    """

    edge = None
    option = None

    if distances is not None:                                                                                           # if we still don't have some values for the edges
        edges_from_current_node = self.graph.get_edges_of_a_node(current_node)
        if len(edges_from_current_node) > 0:                                                                            # if we are not for the first time in this abstract states
            if random.random() < self.current_node.epsilon:                                                             # epsilon greedy choose
                random_edge_index = random.choice(range(len(edges_from_current_node) + 1))
                if random_edge_index >= len(edges_from_current_node):                                                   # if randomly we choose the exploration option
                    # here it means we choose the exploration option
                    self.target = None
                    self.path_2_print.append("exploration " + str(current_node.state) + " - " + str(self.target))
                    option = self.exploration_option

                else:                                                                                                   # if randomly we choose an edge
                    self.target = edges_from_current_node[random_edge_index].get_destination()
                    self.path_2_print.append(self.target)
                    edge = edges_from_current_node[random_edge_index]
                    option = self.get_option(edge)

            else:                                                                                                       # if we select the best option
                best_edge_index = random.choice(
                    self.graph.get_node_best_edge_index(current_node, distances, edges_from_current_node))
                best_edge = edges_from_current_node[best_edge_index]
                self.target = best_edge.get_destination()

                self.path_2_print.append("best choice " + str(current_node.state) + " - " + str(self.target.state))
                edge = edges_from_current_node[best_edge_index]
                option = self.get_option(edge)
        else:                                                                                                           # the first time in this abstract state so we have to explore for finding edges
            self.target = None
            self.path_2_print.append("exploration " + str(self.current_node.state) + " - " + str(self.target))
            option = self.exploration_option
    else:                                                                                                               # if we still don't have some values for the edges utility just choose exploration option
        self.target = None
        self.path_2_print.append("exploration " + str(self.current_node.state) + " - " + str(self.target))
        option = self.exploration_option

    return option, edge