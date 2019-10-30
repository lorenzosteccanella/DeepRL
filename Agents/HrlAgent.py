import random
from Agents.AbstractAgent import AbstractAgent
from Utils import Edge, Node, Graph
import time
import math

class HrlAgent(AbstractAgent):

    manager_exp = 0

    epsilon = 1

    def __init__(self, option_params, exploration_option, LAMBDA=0.0005, MIN_EPSILON=0.01, correct_option_end_reward=0.6, wrong_option_end_reward=-0.6):

        self.option_params = option_params

        self.action_space = option_params["action_space"]

        self.graph = Graph()

        self.best_option_action = None
        self.current_node = None
        self.exploration_option = exploration_option
        self.options = []
        self.target = None

        self.MIN_EPSILON = MIN_EPSILON
        self.LAMBDA = LAMBDA
        self.correct_option_end_reward = correct_option_end_reward
        self.wrong_end_option_reward = wrong_option_end_reward

    def act(self, s):
        node = Node(s["manager"], 0)
        self.graph.node_update(node)
        # if structure to reduce computation cost
        if self.current_node is None:
            self.current_node = self.graph.get_current_node()
            distances = self.graph.find_distances(self.current_node)
            self.best_option_action = self.get_epsilon_best_action(distances)
        elif self.current_node != node:
            self.current_node = self.graph.get_current_node()
            distances = self.graph.find_distances(self.current_node)
            self.best_option_action = self.get_epsilon_best_action(distances)

        if self.target is not None:
            #self.graph.print_node_list()
            print(self.current_node.state, "->", self.target.state, " - ", self.best_option_action, end=" ")

        return self.best_option_action.act(s["option"])

    def sub_path(self, distances, root):
        next_node = None
        edge_from_root = self.graph.get_edges_of_a_node(root)
        if distances is not None:
            next_node_min = - float("inf")
            for edge in edge_from_root:
                if distances[edge.destination] > next_node_min:
                    next_node_min = distances[edge.destination]
                    next_node = edge.destination
            return next_node
        elif len(edge_from_root) > 0:
            return random.choice(edge_from_root)
        else:
            return None

    def get_epsilon_best_action(self, distances):
        if distances is not None:
            edges_from_current_node = self.graph.get_edges_of_a_node(self.current_node)
            if len(edges_from_current_node) > 0:
                if random.random() < self.epsilon:
                    random_edge_index = random.choice(range(len(edges_from_current_node)))
                    self.target = edges_from_current_node[random_edge_index].get_destination()
                    return self.options[random_edge_index]
                else:
                    max_distance = - float("inf")
                    for i, edge in zip(range(len(edges_from_current_node)), edges_from_current_node):
                        if distances[edge.destination] > max_distance:
                            max_distance = distances[edge.destination]
                            best_edge_index = i
                            self.target = edges_from_current_node[best_edge_index].get_destination()
                    self.options[best_edge_index].add_edge(edges_from_current_node[best_edge_index])
                    return self.options[best_edge_index]
            else:
                self.target = None
                return self.exploration_option
        else:
            self.target = None
            return self.exploration_option

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
                    r += self.correct_option_end_reward
                    done = True

                else:
                    r += self.wrong_end_option_reward
                    done = True

        print(r, done)

        self.best_option_action.observe((s, a, r, s_, done, info))

    def observe(self, sample):  # in (s, a, r, s_, done, info) format

        self.graph.abstract_state_discovery(sample)
        edges_from_current_node = self.graph.get_edges_of_a_node(self.current_node)
        self.create_options(edges_from_current_node)
        self.update_option(sample)





        # slowly decrease Epsilon based on manager experience
        if sample[0]["manager"] != sample[3]["manager"]:
            self.manager_exp += 1
            self.epsilon = self.MIN_EPSILON + (1 - self.MIN_EPSILON) * math.exp(-self.LAMBDA * self.manager_exp)

    def replay(self):
        pass
