import random

def get_best_action(self, distances):
    if distances is not None:
        edges_from_current_node = self.graph.get_edges_of_a_node(self.current_node)
        if len(edges_from_current_node) > 0:
            best_edge_index = random.choice(
                self.graph.get_node_best_edge_index(self.current_node, distances, edges_from_current_node, False))
            best_edge = edges_from_current_node[best_edge_index]
            self.target = best_edge.get_destination()

            for option in self.options:
                for edge in option.get_edge_list():
                    if edge == best_edge:
                        return option

            self.path_2_print.append("best choice " + str(self.current_node.state) + " - " + str(self.target.state))
            self.options[best_edge_index].add_edge(best_edge)
            return self.options[best_edge_index]
        else:
            self.target = None
            return self.exploration_option
    else:
        self.target = None
        return self.exploration_option


def get_epsilon_best_action(self, distances):
    if distances is not None:
        # print(distances)
        edges_from_current_node = self.graph.get_edges_of_a_node(self.current_node)
        # print(self.current_node, edges_from_current_node)
        if len(edges_from_current_node) > 0:
            if random.random() < self.epsilon:
                random_edge_index = random.choice(range(len(edges_from_current_node) + 1))
                if random_edge_index >= len(edges_from_current_node):
                    # here it means we choose the exploration option
                    self.target = None
                    self.path_2_print.append("exploration " + str(self.current_node.state) + " - " + str(self.target))
                    return self.exploration_option
                else:
                    self.target = edges_from_current_node[random_edge_index].get_destination()
                    self.path_2_print.append(self.target)
                    return self.options[random_edge_index]
            else:
                best_edge_index = random.choice(
                    self.graph.get_node_best_edge_index(self.current_node, distances, edges_from_current_node, False))
                best_edge = edges_from_current_node[best_edge_index]
                self.target = best_edge.get_destination()

                for option in self.options:
                    for edge in option.get_edge_list():
                        if edge == best_edge:
                            return option

                self.path_2_print.append("best choice " + str(self.current_node.state) + " - " + str(self.target.state))
                self.options[best_edge_index].add_edge(best_edge)
                return self.options[best_edge_index]
        else:
            self.target = None
            self.path_2_print.append("exploration " + str(self.current_node.state) + " - " + str(self.target))
            return self.exploration_option
    else:
        self.target = None
        self.path_2_print.append("exploration " + str(self.current_node.state) + " - " + str(self.target))
        return self.exploration_option


def get_epsilon_exploration(self, distances):
    if distances is not None:
        # print(distances)
        edges_from_current_node = self.graph.get_edges_of_a_node(self.current_node)
        # print(self.current_node, edges_from_current_node)
        if len(edges_from_current_node) > 0:
            if random.random() < self.epsilon:
                return self.exploration_option

            else:
                best_edge_index = random.choice(
                    self.graph.get_node_best_edge_index(self.current_node, distances, edges_from_current_node, False))
                best_edge = edges_from_current_node[best_edge_index]
                self.target = best_edge.get_destination()

                for option in self.options:
                    for edge in option.get_edge_list():
                        if edge == best_edge:
                            return option

                self.path_2_print.append("best choice " + str(self.current_node.state) + " - " + str(self.target.state))
                self.options[best_edge_index].add_edge(best_edge)
                return self.options[best_edge_index]
        else:
            self.target = None
            self.path_2_print.append("exploration " + str(self.current_node.state) + " - " + str(self.target))
            return self.exploration_option
    else:
        self.target = None
        self.path_2_print.append("exploration " + str(self.current_node.state) + " - " + str(self.target))
        return self.exploration_option


def get_epsilon_count_exploration(self, distances):
    if distances is not None:
        # print(distances)
        edges_from_current_node = self.graph.get_edges_of_a_node(self.current_node)
        # print(self.current_node, edges_from_current_node)
        if len(edges_from_current_node) > 0:
            if random.random() < self.current_node.epsilon:
                return self.exploration_option

            else:
                best_edge_index = random.choice(
                    self.graph.get_node_best_edge_index(self.current_node, distances, edges_from_current_node, False))
                best_edge = edges_from_current_node[best_edge_index]
                self.target = best_edge.get_destination()

                for option in self.options:
                    for edge in option.get_edge_list():
                        if edge == best_edge:
                            return option

                self.path_2_print.append("best choice " + str(self.current_node.state) + " - " + str(self.target.state))
                self.options[best_edge_index].add_edge(best_edge)
                return self.options[best_edge_index]
        else:
            self.target = None
            self.path_2_print.append("exploration " + str(self.current_node.state) + " - " + str(self.target))
            return self.exploration_option
    else:
        self.target = None
        self.path_2_print.append("exploration " + str(self.current_node.state) + " - " + str(self.target))
        return self.exploration_option