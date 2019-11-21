import math
import random

class Edge:

    def __init__(self, origin, destination, value=0, edge_cost = 0):

        self.origin = origin
        self.destination = destination
        self.edge_cost = edge_cost
        self.value = self.edge_cost
        self.option = None

    def get_value(self):
        return self.value

    def get_origin(self):
        return self.origin

    def get_destination(self):
        return self.destination

    def set_value(self, value):
        self.value = value + self.edge_cost

    def refresh_value(self):
        self.value =  self.edge_cost

    def update_value(self, value):
        self.value = (self.value + self.edge_cost + value)/2

    def set_option(self, option):
        self.option = option

    def get_option(self):
        return self.option

    def __eq__(self, other):
        if self.origin == other.origin and self.destination == other.destination:
            return True
        else:
            return False

    def __repr__(self):
        return "Edge ["+str(self.origin) + ", " + str(self.destination) + ", " + str(self.value)+"]"

    def __str__(self):
        return "["+str(self.origin.state) + ", " + str(self.destination.state) + ", " + str(self.value)+"]\n"


class Node:

    pseudo_count_factor = 1000

    def __init__(self, state, value=0):
        self.state = state
        self.visit_count = 1
        self.value = value + 1 * (math.exp(-Node.pseudo_count_factor * self.visit_count)) #(Node.pseudo_count_factor * (self.visit_count ** -1))

    def get_value(self):
        return self.value

    def get_state(self):
        return self.state

    def set_value(self, value):
        self.value = value + 1 *(math.exp(-Node.pseudo_count_factor * self.visit_count)) #(Node.pseudo_count_factor * (self.visit_count ** -1))

    def visited(self):
        self.visit_count += 1

    def __eq__(self, other):
        return self.state == other.state

    def __repr__(self):
        return "Node ["+str(self.state) + ", " + str(self.value)+"]"

    def __str__(self):
        return "["+str(self.state) + ", " + str(self.value)+", " + str(self.visit_count) + "]"

    def __hash__(self):
        return hash(self.state)

    @staticmethod
    def set_pseudo_count(pseudo_count_factor):
        Node.pseudo_count_factor = pseudo_count_factor


class Graph:

    def __init__(self):
        self.edge_list = []
        self.node_list = []
        self.current_node = None
        self.current_edge = None
        self.new_node_encontered = False
        self.new_edge_encontered = False
        self.index_4_bestpathprint = 0

    def edge_update(self, old_node, new_node, reward):

        self.new_edge_encontered = False

        if old_node != new_node:
            current_edge = Edge(old_node, new_node)
            if current_edge not in self.edge_list:
                self.new_edge_encontered = True
                self.edge_list.append(current_edge)
                #current_edge.set_value(reward)
                current_edge.refresh_value()
                self.current_edge = current_edge
            else:
                edge = self.edge_list[self.edge_list.index(current_edge)]
                edge.refresh_value()
                #edge.update_value(reward)
                self.current_edge = edge

    def node_update(self, old_node, new_node=False, reward=False):

        self.new_node_encontered = False

        if old_node not in self.node_list:
            self.node_list.append(old_node)
            self.new_node_encontered = True

        if new_node:
            if new_node not in self.node_list:
                self.node_list.append(new_node)
                self.new_node_encontered = True
        if new_node:
            node = self.node_list[self.node_list.index(new_node)]
            node.set_value((node.get_value() + reward) / 2)
        else:
            node = self.node_list[self.node_list.index(old_node)]

        node.visited() # to augment the visit counter

        self.current_node = node

    def abstract_state_discovery(self, sample):
        old_node = Node(sample[0]["manager"], 0)

        new_node = Node(sample[3]["manager"], sample[2])

        self.node_update(old_node, new_node, sample[2])
        self.edge_update(old_node, new_node, sample[2])

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
            if distances[edge.destination] == max_distance:
                best_edge_index.append(i)
            elif distances[edge.destination] > max_distance:
                best_edge_index.clear()
                best_edge_index.append(i)
                max_distance = distances[edge.destination]

        if verbose:
            print("root -> ", root.state)
            for i in best_edge_index:
                print("destination ->", edges[i].destination.state)

        return best_edge_index

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
            if distances[edge.destination] == max_distance:
                best_edge_array.append(edge)
            elif distances[edge.destination] > max_distance:
                best_edge_array.clear()
                best_edge_array.append(edge)
                max_distance = distances[edge.destination]

        best_edge = random.choice(best_edge_array)
        print(best_edge.destination.state)
        return self.print_best_path(best_edge.destination, distances)

    def value_iteration(self, n_steps):
        distances = {}
        for node in self.node_list:
            distances[node] = 0 # distances for all node setted to 0
        # distances[root] = 0. # the source distance is set to 0
        for _ in range(n_steps):
            for node in self.node_list:
                edges = self.get_edges_of_a_node(node)
                values = []
                if (len(edges) > 0):
                    for edge in edges:
                        origin = edge.get_origin()
                        destination = edge.get_destination()
                        # if((distances[node] + edge.get_value())<distances[node]):
                        V = edge.get_value() + 0.95 * distances[destination]
                        values.append(V)

                    distances[node] = max(values)
                else:
                    distances[node] = node.value

        return distances

    def bellman_ford(self, root):
        distances = {}
        for node in self.node_list:
            distances[node] = - float("inf")  # distances for all node setted to - inf
        distances[root] = 0. # the source distance is set to 0

        for i in range(len(self.node_list) - 1):
            #print("LOOP ", i)
            for edge in self.edge_list:
                origin = edge.get_origin()
                destination = edge.get_destination()
                if distances[destination] < distances[origin] + edge.get_value() + destination.value :
                    distances[destination] = distances[origin] + edge.get_value() + destination.value

        return distances


    def find_distances(self, root):
        """
        Bellman-Ford algorithm to get the longest path (highest value)
        :param root: the origin of the path
        :return: distances.
        """
        if len(self.node_list) > 0 and len(self.edge_list) > 0:

            root_origin=self.node_list[0]

            distances = self.value_iteration(1000)


            #print("DISTANCES")
            #print("\nroot", root.state)
            #for node in self.node_list:
            #  print(node.state, " = ", distances[node])
            #print()
            #self.print_best_path(root, distances)

            return distances

        else:
            return None

    def get_edges_of_a_node(self, node):
        edge_node = []
        for edge in self.edge_list:
            if edge.origin == node:
                edge_node.append(edge)

        return edge_node

    def get_current_node(self):
        return self.current_node

    def get_current_edge(self):
        return self.current_edge

    def reset_distances(self):
        self.distances = None

    def add_node(self, node):
        self.node_list.append(node)

    def add_edge(self, edge):
        self.edge_list.append(edge)
