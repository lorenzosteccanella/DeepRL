import math
import random
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (20, 20)
#plt.ion() # enable interactivity
#plt.show()
#fig = plt.figure() # make a figure

class KeyDict:

    def __init__(self, s):
        self.s = s

    def __eq__(self, other):
        if type(self.s).__name__ == "ndarray":
            if np.array_equal(self.s, other.s):
                return True
            else:
                return False
        else:
            if self.s == other.s:
                return True
            else:
                return False

    def __hash__(self):
        if type(self.s).__name__ == "ndarray":
            return hash(self.s.tostring())
        else:
            return hash(self.s)




class Distances:

    def __init__(self, node_list):
        self.distances = {}
        for node in node_list:
            self.distances[node] = 0 # distances for all node setted to 0
            edges = self.get_edges_of_a_node(node)
            if (len(edges) == 0):
                self.distances[node] = node.value

    def update(self, graph):
        for node in graph.node_list:
            if node not in self.distances:
                self.distances[node] = 0
            edges = graph.get_edges_of_a_node(node)
            if (len(edges) == 0):
                self.distances[node] = node.value

class Edge:

    pseudo_count_factor = 1000

    def __init__(self, origin, destination, value=0, edge_cost = 0):

        self.succes_execution_counter = 0
        self.origin = origin
        self.visit_count = 1
        self.destination = destination
        self.edge_cost = edge_cost
        self.value = value + self.edge_cost #+ (Edge.pseudo_count_factor/math.sqrt(self.visit_count)) #+ round(0.5 * (math.exp(-Edge.pseudo_count_factor * self.visit_count)), 3)
        self.option = None

    def get_value(self):
        return self.value

    def get_origin(self):
        return self.origin

    def get_destination(self):
        return self.destination

    def set_value(self, value):
        self.value = value + self.edge_cost #+ (Edge.pseudo_count_factor/math.sqrt(self.visit_count)) #+ round(0.5 * (math.exp(-Edge.pseudo_count_factor * self.visit_count)), 3)

    def visited(self):
        self.visit_count += 1
        self.value = self.value #+ (Edge.pseudo_count_factor/math.sqrt(self.visit_count)) #+ self.edge_cost + round(0.5 * (math.exp(-Edge.pseudo_count_factor * self.visit_count)),3)

    def set_option(self, option):
        self.option = option

    def get_option(self):
        return self.option

    def update_succes_execution_counter(self, value):
        self.succes_execution_counter += 1

    def __eq__(self, other):
        if type(self.origin).__name__ == "ndarray":
            if np.array_equal(self.origin, other.origin) and np.array_equal(self.destination, other.destination):
                return True
            else:
                return False
        else:
            if self.origin == other.origin and self.destination == other.destination:
                return True
            else:
                return False

    def __repr__(self):
        return "Edge ["+repr(self.origin) + ", " + repr(self.destination) + "]"

    def __str__(self):
        if type(self.origin.state).__name__ == "ndarray":
            return "[" + str(hash(self.origin.state.tostring())) + ", " + str(hash(self.destination.state.tostring())) + "]"
        else:
            return "["+str(self.origin.state) + ", " + str(self.destination.state) + "]"

    def __hash__(self):
        if type(self.origin.state).__name__ == "ndarray":
            return hash((self.origin.state.tostring(), self.destination.state.tostring()))
        else:
            return hash((self.origin, self.destination))

    @staticmethod
    def set_pseudo_count(pseudo_count_factor):
        Edge.pseudo_count_factor = pseudo_count_factor


class Node:

    lambda_node = 1000
    min_epsilon = 0

    def __init__(self, state, value=0):
        self.state = state
        self.visit_count = 1
        self.value = value #+ 1 * (math.exp(-Node.pseudo_count_factor * self.visit_count)) #(Node.pseudo_count_factor * (self.visit_count ** -1))
        self.lambda_node = Node.lambda_node
        self.min_epsilon = Node.min_epsilon
        self.epsilon = self.min_epsilon + (1 - self.min_epsilon) * (math.exp(-self.lambda_node * self.visit_count))

    def get_value(self):
        return self.value

    def get_state(self):
        return self.state

    def set_value(self, value):
        self.value = value #+ 1 * (math.exp(-Node.pseudo_count_factor * self.visit_count)) #(Node.pseudo_count_factor * (self.visit_count ** -1))

    def visited(self):
        self.visit_count += 1
        self.epsilon = self.min_epsilon + (1 - self.min_epsilon) * (math.exp(-self.lambda_node * self.visit_count))
        #print(self.state, self.epsilon, self.visit_count)


        #print(self.state, self.epsilon)

    def get_n_visits(self):
        return self.visit_count

    def reset_n_visits(self):
        self.visit_count=1

    def __eq__(self, other):
        if type(self.state).__name__ == "ndarray":
            return np.array_equal(self.state, other.state)
        else:
            return self.state == other.state

    def __repr__(self):
        if type(self.state).__name__ == "ndarray":
            return "Node ["+str(hash(self.state.tostring()))+"]"
        else:
            return "Node ["+str(self.state)+"]"

    def __str__(self):
        if type(self.state).__name__ == "ndarray":
            return "[" + str(hash(self.state.tostring())) + ", " + str(self.value) + ", " + str(self.visit_count) + "]"
        else:
            return "["+str(self.state) + ", " + str(self.value)+", " + str(self.visit_count) + "]"

    def __hash__(self):
        if type(self.state).__name__ == "ndarray":
            return hash(self.state.tostring())
        else:
            return hash(self.state)

    @staticmethod
    def set_lambda_node(lambda_node, min_epsilon=0):
        Node.lambda_node = lambda_node
        Node.min_epsilon = min_epsilon


class Graph:

    def __init__(self, edge_list = list(), node_list = list(), Q = {}, E = {}, distances = {}, node_edges_dictionary = {}, destination_node_edges_dictionary = {}, save_results = False):
        self.edge_list = edge_list
        self.node_list = node_list
        self.current_node = None
        self.current_edge = None
        self.new_node_encontered = False
        self.new_edge_encontered = False
        self.index_4_bestpathprint = 0
        self.Q = Q
        self.E = E
        self.distances = distances
        self.total_reward_node = 0
        self.total_reward_edge = 0
        self.node_edges_dictionary = node_edges_dictionary # this is the graph representation with nodes as key and eges list as values, this structure is just used to speed up computation
        self.destination_node_edges_dictionary = destination_node_edges_dictionary
        self.path = []
        self.i = 0
        self.batch = []
        self.save_results = save_results

    def print_networkx_graph(self, root, route, distances):
        if self.save_results:
            self.i += 1
            if self.i % 1000 == 0:
                self.i = 0
                from time import sleep
                import networkx as nx

                def V(node):
                    maxQ = - float("inf")
                    for edge in distances[node]:
                        if maxQ < (distances[node][edge]):
                            maxQ = (distances[node][edge])
                    return maxQ

                G = nx.MultiDiGraph()
                edge_lab={}

                for node in self.node_list:
                    G.add_node((node))

                for edge in self.edge_list:
                    G.add_edge(edge.origin, edge.destination, weight= edge.value)
                    edge_lab.update({(edge.origin, edge.destination): str(self.Q[edge.origin][edge])})
                pos = nx.drawing.nx_agraph.graphviz_layout(G)
                nx.draw(G, pos, edge_color='black', width=1, linewidths=1, connectionstyle='arc3, rad = 0.1', \
                        node_size=500, node_color='pink', alpha=0.9, font_size=10, \
                        labels={node: (repr(node) + "\n" + str(round(V(node), 5))) for node in G.nodes()})

                nx.draw_networkx_nodes(G, pos, nodelist=[(root)], node_color='y', alpha=1)
                nx.draw_networkx_edge_labels(G, pos, connectionstyle='arc3, rad = 0.1', edge_labels = edge_lab, font_color='red', label_pos=0.3)
                nx.draw_networkx_edges(G, pos, connectionstyle='arc3, rad = 0.1', edgelist=route, edge_color='g', width=5)

                plt.draw()
                plt.savefig(self.save_results.get_path() + "/Graph.png", format="PNG")
                plt.clf()

    def edge_update(self, old_node, new_node, reward, target):

        self.new_edge_encontered = False

        if old_node != new_node:
            edge = Edge(old_node, new_node)

            self.total_reward_edge += reward

            # if is a new edge I add it
            if edge not in self.node_edges_dictionary[old_node]:
                self.new_edge_encontered = True
                edge.set_value(self.total_reward_edge)
                self.edge_list.append(edge)
                self.node_edges_dictionary[old_node].append(edge)
                self.destination_node_edges_dictionary[new_node].append(edge)  # this structure is just to speed up at the cost of memory
                a_q_s = {edge: edge.value}
                self.Q[old_node].update(a_q_s)
                a_e_s = {edge: 0}
                self.E[old_node].update(a_e_s)

            edge = self.node_edges_dictionary[old_node][self.node_edges_dictionary[old_node].index(edge)]
            edge.visited()

            if target is not None:
                if new_node == target:
                    edge.set_value(0.8 * edge.get_value() + 0.2 * self.total_reward_edge)

            self.total_reward_edge = 0
            self.current_edge = edge

        elif target is not None:
            self.total_reward_edge += reward



    def node_update(self, old_node, new_node=False, reward=False, done=False):

        self.new_node_encontered = False

        # if the nodes are new I add them to the list
        if old_node not in self.node_list:
            self.node_list.append(old_node)
            self.new_node_encontered = True
            self.node_edges_dictionary[old_node] = []
            self.destination_node_edges_dictionary[old_node] = [] # this structure is just to speed up at the cost of memory
            self.Q[old_node] = {}
            self.E[old_node] = {}

        if new_node:
            if new_node not in self.node_list:
                self.node_list.append(new_node)
                self.new_node_encontered = True
                self.node_edges_dictionary[new_node] = []
                self.destination_node_edges_dictionary[new_node] = [] # this structure is just to speed up at the cost of memory
                self.Q[new_node] = {}
                self.E[new_node] = {}

        #setting the value for the specific abstract node
        if new_node: # if we recived the new node together with the old node as parameters
            node = self.node_list[self.node_list.index(new_node)]

        #this is just used to add the first abstract state at the beginning of a epoch
        else: # if we recived just the old node as parameter i.e. new_node = False and reward = False
            node = self.node_list[self.node_list.index(old_node)]

        if new_node:
            if old_node != new_node:
                node.visited() # to augment the visit counter

        #else:
        #    node.visited()


        self.current_node = node

        return self.node_list[self.node_list.index(old_node)], self.current_node

    def abstract_state_discovery(self, sample, target):
        old_node = Node(sample[0]["manager"], 0)

        new_node = Node(sample[3]["manager"], 0)

        done = sample[4]

        r = sample[2]

        old_node_in_list, new_node_in_list = self.node_update(old_node, new_node, r, done)

        self.edge_update(old_node_in_list, new_node_in_list, r, target)

        if done:
            self.total_reward_node = 0
            self.total_reward_edge = 0
            self.current_edge = None
            self.current_node = None
            #self.print_edge_list()
            #self.print_networkx_graph()

        return self.new_node_encontered, self.new_edge_encontered

    def print_edge_list(self):
        for edge in self.edge_list:
            print(edge)

    def print_node_list(self):
        for node in self.node_list:
            print(node)

    def string_node_list(self):
        string = ""

        for node in self.node_list:
            string += node.__str__() + "\n"

        return string

    def string_edge_list(self):
        string = ""

        for edge in self.edge_list:
            string += edge.__str__() + "\n"

        return string

    def get_number_of_nodes(self):
        return len(self.node_list)

    def get_node_best_edge_index(self, root, distances, edges=False, verbose=False):
        max_distance = - float("inf")
        best_edge_index = []
        if not edges:
            edges = self.get_edges_of_a_node(root)
        for i, edge in zip(range(len(edges)), edges):
            if self.distances[edge.origin][edge] == max_distance:
                best_edge_index.append(i)
            elif self.distances[edge.origin][edge] > max_distance:
                best_edge_index.clear()
                best_edge_index.append(i)
                max_distance = self.distances[edge.origin][edge]

        if verbose:
            print("root -> ", root.state)
            for i in best_edge_index:
                print("destination ->", edges[i].destination.state)

        return best_edge_index

    def best_path(self, root, distances):
        max_distance = - float("inf")
        best_edge_index = []
        self.index_4_bestpathprint += 1
        if self.index_4_bestpathprint > 100:
            self.index_4_bestpathprint = 0
            return self.path

        edges = self.get_edges_of_a_node(root)

        if len(edges) == 0:
            return self.path


        for i, edge in zip(range(len(edges)), edges):
            if self.distances[edge.origin][edge] == max_distance:
                best_edge_index.append(i)
            elif self.distances[edge.origin][edge] > max_distance:
                best_edge_index.clear()
                best_edge_index.append(i)
                max_distance = self.distances[edge.origin][edge]

        for i in best_edge_index:
            self.path.append((edges[i].origin, edges[i].destination))

        return self.best_path(edges[random.choice(best_edge_index)].destination, distances)

    def print_best_path(self, root, distances):
        self.index_4_bestpathprint += 1
        if self.index_4_bestpathprint > 20:
            self.index_4_bestpathprint = 0
            return False
        edges = self.get_edges_of_a_node(root)
        max_distance = - float("inf")
        best_edge_array = []
        if(len(edges)==0):
            return False
        for edge in edges:
            if self.distances[edge.origin][edge] == max_distance:
                best_edge_array.append(edge)
            elif self.distances[edge.origin][edge] > max_distance:
                best_edge_array.clear()
                best_edge_array.append(edge)
                max_distance = self.distances[edge.origin][edge]

        best_edge = random.choice(best_edge_array)
        print(best_edge.destination.state)
        return self.print_best_path(best_edge.destination, distances)

    def tabularMC(self, sample):

        learning_rate = 0.6
        gamma = 0.95

        done = sample[4]

        self.batch.append(sample)

        if done:
            rewards = np.array([o[2] for o in self.batch])
            #if sum(rewards) >= 0:
            discounted_r = np.zeros_like(rewards)
            running_add = 0
            for t in reversed(range(rewards.shape[0])):
                running_add = running_add * gamma + rewards[t]
                discounted_r[t] = running_add
            #if sum( discounted_r >= 0.):
            for i, sample in zip(range(len(self.batch)), self.batch):
                s = sample[0]
                a = sample[1]
                correct_termination = sample[5]
                if correct_termination:
                    newQ = discounted_r[i]
                    if newQ > self.Q[s][a]:
                        self.Q[s][a] = newQ
                    #td_error = (discounted_r[i] - self.Q[s][a])
                    #self.Q[s][a] = self.Q[s][a] + learning_rate * td_error
                    #if 1e-4 > self.Q[s][a] > - 1e-4:
                    #   self.Q[s][a] = 0.  # round(self.Q[s][a], 7)
                    #print(actions, right_termination)
            self.batch.clear()



    def tabularQ(self, sample):

        learning_rate = 0.9
        gamma = 0.95
        N = 6

        done = sample[4]

        self.batch.append(sample)

        if done or len(self.batch)==N:
            rewards = np.array([o[2] for o in self.batch])
            dones = np.array([o[4] for o in self.batch])
            s1_t_N = self.batch[-1][3]
            if len(self.Q[s1_t_N]) > 0:
                maxQs1_t_N = - float("inf")
                for edge in self.Q[s1_t_N]:
                    if maxQs1_t_N < (self.Q[s1_t_N][edge]):
                        maxQs1_t_N = (self.Q[s1_t_N][edge])
            else:
                maxQs1_t_N = 0

            returns = np.append(np.zeros_like(rewards), [maxQs1_t_N], axis=-1)
            for t in reversed(range(rewards.shape[0])):
                returns[t] = rewards[t] + gamma * returns[t + 1] * (1 - dones[t])

            returns = returns[:-1]

            #if sum(returns)>=0.:    # WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING
            for i, sample in zip(range(len(self.batch)), self.batch):
                s = sample[0]
                a = sample[1]
                correct_termination = sample[5]
                if correct_termination:
                    td_error = (returns[i] - self.Q[s][a])
                    self.Q[s][a] = self.Q[s][a] + learning_rate * td_error
                    if 1e-4 > self.Q[s][a] > - 1e-4:
                        self.Q[s][a] = 0. #round(self.Q[s][a], 7)

            self.batch.clear()


    def WatkinsQ(self, sample, exploration_fn):

        learning_rate = 0.9
        gamma = 0.99
        lamb = 0.8

        s = sample[0]
        a = sample[1]
        r = sample[2]
        s_ = sample[3]
        done = sample[4]

        if len(self.Q[s_])>0:
            o_, edge_ = exploration_fn(s_, self.Q, True)
            if edge_ is not None:
                QS1 = self.Q[s_][edge_]
            else:
                QS1 = 0
        else:
            QS1 = 0

        if len(self.Q[s_])>0:
            maxQS1 = - float("inf")
            for edge in self.Q[s_]:
                if maxQS1 < (self.Q[s_][edge]):
                    maxQS1 = (self.Q[s_][edge])
        else:
            maxQS1 = 0

        td_error = r + gamma * maxQS1 - self.Q[s][a]

        # Check wether the action choosen was exploratory or not
        if len(self.Q[s_]) > 0:
            exploratory_action = 0 if maxQS1 == QS1 else 1
        else:
            exploratory_action = 1

        self.E[s][a] += 1  # Update trace for the current state

        for s_i in self.node_list:
            edges = self.get_edges_of_a_node(s_i)
            for a_i in edges:
                self.Q[s_i][a_i] += learning_rate * td_error * self.E[s_i][a_i]

                # Update traces
                if exploratory_action:
                    self.E[s_i][a_i] = 0
                else:
                    self.E[s_i][a_i] *= gamma * lamb


    def find_distances(self, root):
        """
        Bellman-Ford algorithm to get the longest path (highest value)
        :param root: the origin of the path
        :return: distances.
        """
        if len(self.node_list) > 0 and len(self.edge_list) > 0:

            #if len(self.distances) < len(self.node_list):

            #root_origin=self.node_list[0]
            #if self.new_edge_encontered:

                #self.distances= self.value_iteration(0.0001) #distances= self.value_iteration(0.0001) #

            self.distances = self.Q

            self.path.clear() # used by best_path function
            path = self.best_path(root, self.distances)
            self.print_networkx_graph(root, path, self.distances)

            return self.distances

        else:
            return None

    def get_edges_of_a_node(self, node):

        return self.node_edges_dictionary[node]

    def get_current_node(self):
        return self.current_node

    def get_current_edge(self):
        return self.current_edge

    def reset_distances(self):
        self.distances = None

    def add_node(self, node):
        self.node_list.append(node)
        self.node_edges_dictionary[node] = []

    def add_edge(self, edge):
        self.edge_list.append(edge)
        self.node_edges_dictionary[edge.origin].append(edge)
