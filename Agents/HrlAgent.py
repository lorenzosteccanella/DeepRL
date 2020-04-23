import random
from Agents.AbstractAgent import AbstractAgent
from Utils import Edge, Node, Graph, KeyDict
import time
import math
import numpy as np
import dill
import matplotlib.pyplot as plt
import copy

class HrlAgent(AbstractAgent):

    manager_exp = 0

    epsilon = 1

    def __init__(self, option_params, exploration_option, exploration_fn, pseudo_count_exploration = 1000, LAMBDA=1000, MIN_EPSILON=0, correct_option_end_reward=1.1, wrong_option_end_reward=-1.1, SaveResult = False, graph=False, options_list=False, single_option=False, as_m2s_m=False):

        self.option_params = option_params

        self.action_space = option_params["action_space"]

        self.save_result = SaveResult

        if graph is False:
            self.graph = Graph(save_results = self.save_result)
        else:
            self.graph = graph

        #exploration variables
        self.MIN_EPSILON = MIN_EPSILON
        self.LAMBDA = LAMBDA


        # variables to keep statistics of the execution
        self.number_of_options_executed = 1
        self.number_of_successfull_option = 0
        self.count_edges = {}
        self.count_couple_edges = {}
        self.entropy_edges = {}
        self.list_percentual_of_successfull_options = []
        self.old_number_of_options = 0
        self.path_2_print = []
        self.distances_2_print = []
        self.total_r_2_print = []
        self.total_r = 0.
        self.old_edge = None
        self.n_steps = 0
        self.n_episodes = 0
        self.samples_imitation = []

        self.best_option_action = None
        self.precomputed_option_action = None
        self.best_edge = None
        self.current_node = None
        self.exploration_option = exploration_option
        if options_list is False:
            self.options = []
        else:
            self.options = options_list

        self.single_option = single_option

        #if single_option is not False:
        #    self.single_option = self.option_params["option"](self.option_params)
        #else:
        #    self.single_option = single_option

        self.target = None

        self.correct_option_end_reward = correct_option_end_reward
        self.wrong_end_option_reward = wrong_option_end_reward

        HrlAgent.exploration_fn = exploration_fn

        self.pseudo_count_exploration(pseudo_count_exploration)
        self.epsilon_count_exploration(self.LAMBDA, self.MIN_EPSILON)
        self.reward_manager = 0.
        if as_m2s_m is False:
            self.as_m2s_m = {}
        else:
            self.as_m2s_m = as_m2s_m

        self.w_o = {}
        self.old_edge = None

    def pixel_manager_obs(self, s = None, sample = None):
        if s is not None:
            s = copy.deepcopy(s)
            node = Node(s["manager"], 0)
            key = KeyDict(s["option"])

            if node not in self.as_m2s_m:
                self.as_m2s_m[node] = {}
                self.as_m2s_m[node][key] = [copy.deepcopy(s["option"]), 0.]

        if sample is not None:

            sample = copy.deepcopy(sample)
            node1 = Node(sample[0]["manager"], 0)
            node2 = Node(sample[3]["manager"], 0)
            key1 = KeyDict(sample[0]["option"])
            key2 = KeyDict(sample[3]["option"])

            if node1 not in self.as_m2s_m:
                self.as_m2s_m[node1] = {}
                self.as_m2s_m[node1][key1] = [copy.deepcopy(sample[0]["option"]), 0.]

            if node2 not in self.as_m2s_m:
                self.as_m2s_m[node2] = {}
                self.as_m2s_m[node2][key2] = [copy.deepcopy(sample[3]["option"]), 0.]

    def act(self, s):
        node = Node(s["manager"], 0)
        self.graph.node_update(node)
        # if structure to reduce computation cost

        if self.current_node is None:
            self.current_node = self.graph.get_current_node()
            distances = self.graph.find_distances(self.current_node)
            self.distances_2_print.append(distances)
            self.best_option_action, self.best_edge = self.exploration_fn(self.current_node, distances)

            # print(self.current_node, node)
            # print(self.best_option_action, self.best_edge)
            # print()
            # print()

        elif self.current_node != node:
            self.current_node = self.graph.get_current_node()
            distances = self.graph.find_distances(self.current_node)
            self.distances_2_print.append(distances)
            self.best_option_action, self.best_edge = self.exploration_fn(self.current_node, distances)

            # print(self.current_node, node)
            # print(self.best_option_action, self.best_edge)
            # print()
            # print()

        #for option in self.options:
        #    print(option, len(option.get_edge_list()))#, option.get_edge_list())

        return self.best_option_action.act(s["option"])

    def pseudo_count_exploration(self, pseudo_count_factor):
        Edge.set_pseudo_count(pseudo_count_factor)

    def epsilon_count_exploration(self, LAMBDA, min_epsilon=0):
        Node.set_lambda_node(LAMBDA, min_epsilon)

    def get_option(self, edge):
        if edge not in self.w_o:
            self.w_o[edge] = self.option_params["option"](self.option_params)

        return self.w_o[edge]

    def save_statistics(self):

        if self.number_of_options_executed % 1000 == 0 and self.old_number_of_options != self.number_of_options_executed:
            if self.save_result is not False:
                message = ""
                for tot_reward in self.total_r_2_print:
                    message += (str(tot_reward) + "\n")
                message += "\n\n"
                self.save_result.save_data(self.FILE_NAME + "Total Reward", message)
                self.total_r_2_print.clear()

            if self.save_result is not False:
                message = ""
                for target in self.path_2_print:
                    message += (str(target) + "\n-> ")
                message += "\n\n"
                self.save_result.save_data(self.FILE_NAME + "Path", message)
                self.path_2_print.clear()

            if self.save_result is not False:
                message = ""
                if self.distances_2_print is not None:
                    for distance in self.distances_2_print:
                        message += (str(distance) + " \n")
                    message += "\n\n"
                    self.save_result.save_data(self.FILE_NAME + "Distances", message)
                    self.distances_2_print.clear()
            if self.save_result is not False:
                message = (str(self.number_of_options_executed) + " "
                           + str(self.number_of_successfull_option) + " "
                           + str(self.number_of_successfull_option/self.number_of_options_executed*100)
                           +"\n")
                self.save_result.save_data(self.FILE_NAME + "Transitions_performance", message)
                message = ("number of options executed:  ", str(self.number_of_options_executed)
                           + "  number of succesfull termination  "
                           + str(self.number_of_successfull_option)
                           + "\n\n Nodes discovered: \n"
                           + self.graph.string_node_list()
                           + "\n\n Edges discovered: \n"
                           + self.graph.string_edge_list()
                           + "\n")
                self.save_result.save_data(self.FILE_NAME + "Nodes_Edge_discovered", message)
            self.old_number_of_options = self.number_of_options_executed

            if self.save_result is not False:

                self.save_result.save_pickle_data(self.FILE_NAME + "edge_option_stats.pkl", self.count_edges)
                self.save_result.save_pickle_data(self.FILE_NAME + "edge_entropy_stats.pkl", self.entropy_edges)
                self.save_result.save_pickle_data(self.FILE_NAME + "edgeXedge_option_stats.pkl", self.count_couple_edges)

            if self.save_result is not False:
                names = []
                values = []
                for k, v in self.count_edges.items():
                    names.append(str(k))
                    values.append(float((v[1] / v[0]) * 100))

                plt.bar(range(len(names)), values)
                plt.savefig(self.save_result.get_path() + "/edgeXedge_transition_prob", format="PNG")
                plt.close()

    def update_option(self, sample):
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

                    r += self.correct_option_end_reward
                    done = True

                else:
                    r += self.wrong_end_option_reward
                    if r < -1:
                        r = -1
                    done = True

        self.best_option_action.observe((s, a, r, s_, done, info))

    def update_manager(self, sample):

        self.reward_manager += sample[2]

        s = Node(sample[0]["manager"], 0)
        r = self.reward_manager
        s_ = Node(sample[3]["manager"], 0)
        a = Edge(s, s_)
        done = sample[4]
        info = sample[5]

        right_termination_option = False

        if s != s_:
            if a is not None:
                #if s_ == self.target:
                    #self.graph.tabularQ((s, a, r, s_, done, info))
                    #self.graph.WatkinsQ((s, a, r, s_, done, info), self.exploration_fn)
                right_termination_option = True

                self.graph.tabularMC((s, a, r, s_, done, right_termination_option))

            self.reward_manager = 0

        if done:
            self.reward_manager = 0

    def has_method(self, o, name):
        return callable(getattr(o, name, None))

    def statistics_options(self, sample):

        edge = self.best_edge
        s_m = Node(sample[0]["manager"], 0)
        s_m_ = Node(sample[3]["manager"], 0)

        if s_m != s_m_:
            if edge is not None:
                if str(edge) not in self.count_edges:
                    self.count_edges[str(edge)] = [0, 0]

                if str(edge) not in self.entropy_edges:
                    self.entropy_edges[str(edge)] = []

                if self.old_edge is not None:
                    if edge != self.old_edge:
                        if (str(self.old_edge) + str(edge)) not in self.count_couple_edges:
                            self.count_couple_edges[(str(self.old_edge) + str(edge))] = [0, 0]

                        self.count_couple_edges[(str(self.old_edge) + str(edge))][0] += 1

                        if s_m_ == self.target:
                            self.count_couple_edges[(str(self.old_edge) + str(edge))][1] += 1
                            self.old_edge = edge
                        else:
                            self.old_edge = None

                if s_m_ == self.target:

                    # to keep performance statistics
                    self.number_of_successfull_option += 1
                    self.count_edges[str(edge)][1] += 1

                # to keep performance statistics
                self.number_of_options_executed += 1

                self.count_edges[str(edge)][0] += 1

                if (self.has_method(self.best_option_action, 'get_ce_loss')):

                    ce_loss = self.best_option_action.get_ce_loss()

                    if ce_loss is not None:
                        self.entropy_edges[str(edge)].append(ce_loss)

        self.save_statistics()

    def observe(self, sample):  # in (s, a, r, s_, done, info) format

        self.n_steps += 1
        self.total_r += sample[2]

        if sample[4]:
            self.total_r_2_print.append(self.total_r)
            self.total_r = 0

        self.graph.abstract_state_discovery(sample, self.target)

        self.update_option(sample)

        self.update_manager(sample)

        self.statistics_options(sample)

        # slowly decrease Epsilon based on manager experience
        if not self.equal(sample[0]["manager"], sample[3]["manager"]):
            self.manager_exp += 1
            self.epsilon = self.MIN_EPSILON + (1 - self.MIN_EPSILON) * math.exp(-self.LAMBDA * self.manager_exp)

        if sample[4]:
            self.current_node = None
            self.best_edge = None
            self.target = None
            self.n_episodes += 1

        return self.n_steps, self.n_episodes

    def replay(self):
        pass

    def reset_exploration(self):
        self.manager_exp = 0

    def equal(self, a, b):
        if type(a).__name__ == "ndarray":
            return np.array_equal(a, b)
        else:
            return a == b

