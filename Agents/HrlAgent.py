import random
from Agents.AbstractAgent import AbstractAgent
from Utils import Edge, Node, Graph
import time
import math

class HrlAgent(AbstractAgent):

    manager_exp = 0

    epsilon = 1

    def __init__(self, option_params, exploration_option, pseudo_count_exploration = 1000, LAMBDA=1000, MIN_EPSILON=0, correct_option_end_reward=1.1, wrong_option_end_reward=-1.1, SaveResult = False):

        self.option_params = option_params

        self.action_space = option_params["action_space"]

        self.graph = Graph()

        self.save_result = SaveResult


        # variables to keep statistics of the execution
        self.number_of_options_executed = 1
        self.number_of_successfull_option = 0
        self.list_percentual_of_successfull_options = []
        self.old_number_of_options = 0
        self.path_2_print = []
        self.distances_2_print = []

        self.best_option_action = None
        self.current_node = None
        self.exploration_option = exploration_option
        self.options = []
        self.target = None

        self.INITIAL_LAMBDA = LAMBDA
        self.MIN_EPSILON = MIN_EPSILON
        self.LAMBDA = LAMBDA
        self.correct_option_end_reward = correct_option_end_reward
        self.wrong_end_option_reward = wrong_option_end_reward
        self.pseudo_count_exploration(pseudo_count_exploration)

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
            self.best_option_action = self.get_epsilon_best_action(distances)
        elif self.current_node != node:
            self.current_node = self.graph.get_current_node()
            distances = self.graph.find_distances(self.current_node)
            self.distances_2_print.append(distances)
            self.best_option_action = self.get_epsilon_best_action(distances)

        return self.best_option_action.act(s["option"])

    def get_best_action(self, distances):
        if distances is not None:
            edges_from_current_node = self.graph.get_edges_of_a_node(self.current_node)
            if len(edges_from_current_node) > 0:
                    max_distance = - float("inf")
                    best_edge_index = []
                    for i, edge in zip(range(len(edges_from_current_node)), edges_from_current_node):
                        if distances[edge.destination]==max_distance:
                            best_edge_index.append(i)
                        elif distances[edge.destination] > max_distance:
                            best_edge_index.clear()
                            best_edge_index.append(i)
                            max_distance = distances[edge.destination]
                    #if random.random() < self.epsilon:              #
                    #    best_edge = random.choice(best_edge_index)  # Better to check carefully this part
                    #else:                                           # Changed  so now when exploration finish the plan will become deterministic
                    #    best_edge = best_edge_index[0]              #

                    best_edge = random.choice(best_edge_index)
                    self.target = edges_from_current_node[best_edge].get_destination()
                    self.options[best_edge].add_edge(edges_from_current_node[best_edge])
                    return self.options[best_edge]
            else:
                self.target = None
                return self.exploration_option
        else:
            self.target = None
            return self.exploration_option

    def get_epsilon_best_action(self, distances):
        if distances is not None:
            #print(distances)
            edges_from_current_node = self.graph.get_edges_of_a_node(self.current_node)
            #print(self.current_node, edges_from_current_node)
            if len(edges_from_current_node) > 0:
                if random.random() < self.epsilon:
                    random_edge_index = random.choice(range(len(edges_from_current_node)+1))
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
                    best_edge_index= self.graph.get_node_best_edge_index(self.current_node, distances, edges_from_current_node, False)
                    #max_distance = - float("inf")
                    #best_edge_index = []
                    #for i, edge in zip(range(len(edges_from_current_node)), edges_from_current_node):
                    #    if distances[edge.destination]==max_distance:
                    #        best_edge_index.append(i)
                    #    elif distances[edge.destination] > max_distance:
                    #        best_edge_index.clear()
                    #        best_edge_index.append(i)
                    #        max_distance = distances[edge.destination]

                    #if random.random() < self.epsilon:              #
                    #    best_edge = random.choice(best_edge_index)  # Better to check carefully this part
                    #else:                                           # Changed  so now when exploration finish the plan will become deterministic
                    #    best_edge = best_edge_index[0]              #

                    best_edge = random.choice(best_edge_index)
                    self.target = edges_from_current_node[best_edge].get_destination()
                    self.path_2_print.append("best choice " + str(self.current_node.state) + " - " + str(self.target.state))
                    self.options[best_edge].add_edge(edges_from_current_node[best_edge])
                    return self.options[best_edge]
            else:
                self.target = None
                self.path_2_print.append("exploration " + str(self.current_node.state) + " - " + str(self.target))
                return self.exploration_option
        else:
            self.target = None
            self.path_2_print.append("exploration " + str(self.current_node.state) + " - " + str(self.target))
            return self.exploration_option

    def pseudo_count_exploration(self, pseudo_count_factor):
        Node.set_pseudo_count(pseudo_count_factor)

    def create_options(self, edges_from_current_node):

        if len(edges_from_current_node) > len(self.options):

            option = self.option_params["option"]
            option = option(self.option_params)
            self.options.append(option)

    def update_option(self, sample):
        s = sample[0]["option"]
        a = sample[1]
        r = sample[2]
        s_ = sample[3]["option"]
        done = sample[4]
        info = sample[5]

        if sample[0]["manager"] != sample[3]["manager"]:
            if self.target is not None:
                if sample[3]["manager"] == self.target.state:

                    # to keep performance statistics
                    self.number_of_options_executed += 1
                    self.number_of_successfull_option += 1

                    r += self.correct_option_end_reward
                    done = True

                else:
                    # to keep performance statistics
                    self.number_of_options_executed += 1

                    r += self.wrong_end_option_reward
                    done = True

        if self.number_of_options_executed % 10000 == 0 and self.old_number_of_options != self.number_of_options_executed:
            if self.save_result is not False:
                message = ""
                for target in self.path_2_print:
                    message += (str(target) + "\n-> ")
                message += "\n\n"
                self.save_result.save_data("Path", message)
                self.path_2_print.clear()

            if self.save_result is not False:
                message = ""
                if self.distances_2_print is not None:
                    for distance in self.distances_2_print:
                        message += (str(distance) + " \n")
                    message += "\n\n"
                    self.save_result.save_data("Distances", message)
                    self.distances_2_print.clear()
            if self.save_result is not False:
                message = (str(self.number_of_options_executed) + " "
                           + str(self.number_of_successfull_option) + " "
                           + str(self.number_of_successfull_option/self.number_of_options_executed*100)
                           +"\n")
                self.save_result.save_data("Transitions_performance", message)
                message = ("number of options executed:  ", str(self.number_of_options_executed)
                           + "  number of succesfull termination  "
                           + str(self.number_of_successfull_option)
                           + "\n\n Nodes discovered: \n"
                           + self.graph.string_node_list()
                           + "\n\n Edges discovered: \n"
                           + self.graph.string_edge_list()
                           + "\n")
                self.save_result.save_data("Nodes_Edge_discovered", message)
            self.old_number_of_options = self.number_of_options_executed

        #print(r, done, end=" ")
        #if self.number_of_successfull_option > 0:
        #    print("percentage of successfull options", self.number_of_successfull_option/self.number_of_options_executed,
        #          " number of abstract states:", self.graph.get_number_of_nodes())
        #    self.graph.print_node_list()

        self.best_option_action.observe((s, a, r, s_, done, info))

    def reset_exploration(self):
        self.LAMBDA = self.INITIAL_LAMBDA

    def observe(self, sample):  # in (s, a, r, s_, done, info) format

        self.graph.abstract_state_discovery(sample)

        if self.graph.new_node_encontered:
            print("new node discovered, resetting the exploration!!!")
            self.reset_exploration()

        edges_from_current_node = self.graph.get_edges_of_a_node(self.current_node)
        self.create_options(edges_from_current_node)
        self.update_option(sample)

        # slowly decrease Epsilon based on manager experience
        if sample[0]["manager"] != sample[3]["manager"]:
            self.manager_exp += 1
            self.epsilon = self.MIN_EPSILON + (1 - self.MIN_EPSILON) * math.exp(-self.LAMBDA * self.manager_exp)

        if sample[4]:
            self.path_2_print.clear()
            self.distances_2_print.clear()

    def replay(self):
        pass
