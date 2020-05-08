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

    """
    The main Hierarchical Class
    """

    manager_exp = 0

    def __init__(self, option_params, exploration_option, exploration_fn, LAMBDA=1000, MIN_EPSILON=0,
                 correct_option_end_reward=1.1, wrong_option_end_reward=-1.1, SaveResult = False):

        """
        __init__

        Args:
            option_params : all the attributes needed to create an option. i.e. and A2Coption
            exploration_option : an instance of an exploration option to perform random exploration
            exploration_fn : an instance of an exploration function, that determines how the manager choose the option to execute i.e. epsilon-greedy
            LAMBDA: parameter for exploration
            MIN_EPSILON: parameter for exploration
            correct_option_end_reward: the positive value we use to augment the reward in case the option ended correctly
            wrong_option_end_reward: the negative value we use to augment the reward in case the option ended badly
            SaveResult: an instance used to save statistics of performance
        """

        self.option_params = option_params
        self.action_space = option_params["action_space"]
        self.save_result = SaveResult
        self.graph = Graph(save_results = self.save_result)

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
        self.option_rewards = 0

        # variables for manager
        self.best_option_action = None
        self.best_edge = None
        self.current_node = None
        self.target = None
        self.reward_manager = 0.
        self.replan = False
        HrlAgent.exploration_fn = exploration_fn

        # variables for options
        self.correct_option_end_reward = correct_option_end_reward
        self.wrong_end_option_reward = wrong_option_end_reward
        self.options = []
        self.exploration_option = exploration_option
        self.epsilon_count_exploration(self.LAMBDA, self.MIN_EPSILON)
        self.options = {}

    def act(self, s):
        """
        act of the manager takes an environment obs "s" in input and choose the option to perform based
        on the exploration function "exploration_fn"

        Args:
            s : the observation from the environment

        Returns:
            returns the option to perform
        """

        node = Node(s["manager"], 0)
        self.graph.node_update(node)

        # if structure to reduce computation cost
        if self.current_node is None:
            self.replan = True

        elif self.current_node != node:
            self.replan = True

        if self.replan:
            self.current_node = self.graph.get_current_node()
            distances = self.graph.find_distances(self.current_node)
            self.distances_2_print.append(distances)
            self.best_option_action, self.best_edge = self.exploration_fn(self.current_node, distances)
            self.replan = False

        return self.best_option_action.act(s["option"])

    def epsilon_count_exploration(self, LAMBDA, min_epsilon=0):
        """
        set the epsilon count exploration static variables in Node class

        Args:
            LAMBDA : value for exploration
            min_epsilon : min exploration prob
        """
        Node.set_lambda_node(LAMBDA, min_epsilon)

    def get_option(self, edge):
        """
        create a new option for every edge encountered used by "exploration_fn"

        Args:
            edge: the edge "exploration_fn" decided to perform
        """
        if edge not in self.options:
            self.options[edge] = self.option_params["option"](self.option_params)

        return self.options[edge]

    def save_statistics(self):

        """
        just statistics of the run
        """

        if self.number_of_options_executed % 10000 == 0 and self.old_number_of_options != self.number_of_options_executed:
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

            # if self.save_result is not False:
            #     message = ""
            #     if self.distances_2_print is not None:
            #         for distance in self.distances_2_print:
            #             message += (str(distance) + " \n")
            #         message += "\n\n"
            #         self.save_result.save_data(self.FILE_NAME + "Distances", message)
            #         self.distances_2_print.clear()

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

        """
        this is the function that updates the option

        Args:
            sample: a (s, a, r, s', done, info) tuple to train the option on
        """

        s = sample[0]["option"]
        a = sample[1]
        r = sample[2]
        s_ = sample[3]["option"]
        done = sample[4]
        info = sample[5]

        s_m = Node(sample[0]["manager"], 0)                             # the manager abstract state
        s_m_ = Node(sample[3]["manager"], 0)                            # the manager abstract state at time t+1

        if s_m != s_m_:                                                 # if we are at the transitioning from an abstract state to another
            if self.target is not None:                                 # if we are using a real option and not an exploration option
                if s_m_ == self.target:                                 # if we reached the new abstract state successfully
                    r += self.correct_option_end_reward                 # we augment the reward with the correct option end reward
                    done = True                                         # we set done to True the option has finished

                else:                                                   # we didn't finish in the abstract state the manager told us
                    r += self.wrong_end_option_reward                   # we augment the reward with the wrong_option_end reward
                    if r < -1:                                          # this is just to clip the negative value to -1
                        r = -1
                    done = True                                         # the option ended

        self.option_rewards += r
        self.best_option_action.observe((s, a, r, s_, done, info))      # here we train the option selected on this experience

    def update_manager(self, sample):

        """
        this is the function that updates the manager

        Args:
            sample: a (s, a, r, s', done, info) tuple to train the manager on
        """

        #beta = 0.2

        self.reward_manager += sample[2]                                # here we keep a sum of all the reward collected in these abstract state
        s = self.graph.get_node(sample[0]["manager"])                   # the abstract state
        r = self.reward_manager #+ (beta/math.sqrt(s.visit_count))       # the reward of the manager
        #print((beta/math.sqrt(s.visit_count)) )
        s_ = self.graph.get_node(sample[3]["manager"])                  # the abstact state at time t+1
        a = Edge(s, s_)                                                 # the Edge i'm in, this is a trick to define the edge I executed as always the wanted one, even when I'm ending in wrong abstract state
        done = sample[4]                                                # the done returned from the environment
        info = sample[5]

        if s != s_:
            if a is not None:
                self.graph.tabularMC((s, a, r, s_, done, True))
            self.reward_manager = 0

        if done:
            self.reward_manager = 0

    def statistics_options(self, sample):
        """
        just to keep statistics of options

        Args:
            sample: a (s, a, r, s', done, info)
        """
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

        self.save_statistics()

    def observe(self, sample):

        """
        this is the function called from Environment that call updated manager and update option and updates the Graph

        Args:
            sample: a (s, a, r, s', done, info) tuple
        """

        # just statistics for performance
        self.n_steps += 1
        self.total_r += sample[2]

        # just statistics for performance
        if sample[4]:
            self.total_r_2_print.append(self.total_r)
            self.total_r = 0

        self.graph.abstract_state_discovery(sample, self.target)           # the function that update the graph
        self.update_option(sample)                                         # update option
        self.update_manager(sample)                                        # update manager
        self.statistics_options(sample)                                    # update performance statistics

        if not self.equal(sample[0]["manager"], sample[3]["manager"]):     # count number of steps of manager level time
            self.manager_exp += 1

        if sample[4]:                                                      # if we are at the end of the episode
            self.current_node = None
            self.best_edge = None
            self.target = None
            self.n_episodes += 1
            self.option_rewards = 0

        return self.n_steps, self.n_episodes

    def replay(self):                                                       # just needed for the Environment doesn't do nothing
        pass

    def equal(self, a, b):


        """
        this is the function that define equality depending on the type I'm passing in

        Args:
            a : first element
            b : second element
        """

        if type(a).__name__ == "ndarray":
            return np.array_equal(a, b)
        else:
            return a == b

