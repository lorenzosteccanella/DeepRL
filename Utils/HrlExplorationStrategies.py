import random

def get_best_action(self, current_node, distances, store_action_choice=False):

    edge = None
    option = None

    if self.precomputed_option_action is not None:
        option, edge = self.precomputed_option_action
        self.precomputed_option_action = None

        return option, edge

    if distances is not None:
        edges_from_current_node = self.graph.get_edges_of_a_node(current_node)
        if len(edges_from_current_node) > 0:
            best_edge_index = random.choice(
                self.graph.get_node_best_edge_index(self.current_node, distances, edges_from_current_node))
            best_edge = edges_from_current_node[best_edge_index]
            self.target = best_edge.get_destination()

            self.path_2_print.append("best choice " + str(current_node.state) + " - " + str(self.target.state))
            if best_edge not in self.options[best_edge_index].get_edge_list():
                self.options[best_edge_index].add_edge(best_edge)
                option = self.options[best_edge_index]
                edge = edges_from_current_node[best_edge_index]
        else:
            self.target = None
            option = self.exploration_option
    else:
        self.target = None
        option = self.exploration_option

    if store_action_choice:
        self.precomputed_option_action = (option, edge)

    return option, edge


def get_epsilon_best_action(self, current_node, distances, store_action_choice=False):
    edge = None
    option = None

    if self.precomputed_option_action is not None:
        option, edge = self.precomputed_option_action
        self.precomputed_option_action = None
        return option, edge

    if distances is not None:
        # print(distances)
        edges_from_current_node = self.graph.get_edges_of_a_node(current_node)
        # print(self.current_node, edges_from_current_node)
        if len(edges_from_current_node) > 0:
            if random.random() < self.epsilon:
                random_edge_index = random.choice(range(len(edges_from_current_node) + 1))
                if random_edge_index >= len(edges_from_current_node):
                    # here it means we choose the exploration option
                    self.target = None
                    self.path_2_print.append("exploration " + str(current_node.state) + " - " + str(self.target))
                    option = self.exploration_option

                else:
                    self.target = edges_from_current_node[random_edge_index].get_destination()
                    self.path_2_print.append(self.target)
                    option = self.options[random_edge_index]
                    edge = edges_from_current_node[random_edge_index]

            else:
                best_edge_index = random.choice(
                    self.graph.get_node_best_edge_index(self.current_node, distances, edges_from_current_node))
                best_edge = edges_from_current_node[best_edge_index]
                self.target = best_edge.get_destination()

                self.path_2_print.append("best choice " + str(current_node.state) + " - " + str(self.target.state))
                if best_edge not in self.options[best_edge_index].get_edge_list():
                    self.options[best_edge_index].add_edge(best_edge)
                option = self.options[best_edge_index]
                edge = edges_from_current_node[best_edge_index]

        else:
            self.target = None
            self.path_2_print.append("exploration " + str(current_node.state) + " - " + str(self.target))
            option = self.exploration_option

    else:
        self.target = None
        self.path_2_print.append("exploration " + str(current_node.state) + " - " + str(self.target))
        option = self.exploration_option

    if store_action_choice:
        self.precomputed_option_action = (option, edge)

    return option, edge


def get_epsilon_exploration(self, current_node, distances, store_action_choice=False):

    edge = None
    option = None

    if self.precomputed_option_action is not None:
        option, edge = self.precomputed_option_action
        self.precomputed_option_action = None

        return option, edge

    if distances is not None:
        # print(distances)
        edges_from_current_node = self.graph.get_edges_of_a_node(current_node)
        # print(self.current_node, edges_from_current_node)
        if len(edges_from_current_node) > 0:
            if random.random() < self.epsilon:
                option = self.exploration_option

            else:
                best_edge_index = random.choice(
                    self.graph.get_node_best_edge_index(current_node, distances, edges_from_current_node))
                best_edge = edges_from_current_node[best_edge_index]
                self.target = best_edge.get_destination()

                self.path_2_print.append("best choice " + str(current_node.state) + " - " + str(self.target.state))
                if best_edge not in self.options[best_edge_index].get_edge_list():
                    self.options[best_edge_index].add_edge(best_edge)
                option = self.options[best_edge_index]
                edge = edges_from_current_node[best_edge_index]

        else:
            self.target = None
            self.path_2_print.append("exploration " + str(current_node.state) + " - " + str(self.target))
            option = self.exploration_option
    else:
        self.target = None
        self.path_2_print.append("exploration " + str(current_node.state) + " - " + str(self.target))
        option = self.exploration_option

    if store_action_choice:
        self.precomputed_option_action = (option, edge)

    return option, edge


def get_epsilon_count_exploration(self, current_node, distances, store_action_choice=False):

    edge = None
    option = None

    if self.precomputed_option_action is not None:
        option, edge = self.precomputed_option_action
        self.precomputed_option_action = None

        return option, edge

    if distances is not None:
        edges_from_current_node = self.graph.get_edges_of_a_node(current_node)
        if len(edges_from_current_node) > 0:
            if random.random() < self.current_node.epsilon:
                random_edge_index = random.choice(range(len(edges_from_current_node) + 1))
                if random_edge_index >= len(edges_from_current_node):
                    # here it means we choose the exploration option
                    self.target = None
                    self.path_2_print.append("exploration " + str(current_node.state) + " - " + str(self.target))
                    option = self.exploration_option

                else:
                    self.target = edges_from_current_node[random_edge_index].get_destination()
                    self.path_2_print.append(self.target)
                    #option = self.options[random_edge_index]
                    edge = edges_from_current_node[random_edge_index]
                    option = self.get_option(edge)

            else:
                best_edge_index = random.choice(
                    self.graph.get_node_best_edge_index(current_node, distances, edges_from_current_node))
                best_edge = edges_from_current_node[best_edge_index]
                self.target = best_edge.get_destination()

                self.path_2_print.append("best choice " + str(current_node.state) + " - " + str(self.target.state))
                #option = self.options[best_edge_index]
                edge = edges_from_current_node[best_edge_index]
                option = self.get_option(edge)
        else:
            self.target = None
            self.path_2_print.append("exploration " + str(self.current_node.state) + " - " + str(self.target))
            option = self.exploration_option
    else:
        self.target = None
        self.path_2_print.append("exploration " + str(self.current_node.state) + " - " + str(self.target))
        option = self.exploration_option

    if store_action_choice:
        self.precomputed_option_action = (option, edge)

    #print(option)
    return option, edge