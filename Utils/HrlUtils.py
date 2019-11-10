import math

class Edge:

    def __init__(self, origin, destination, value=0, edge_cost = -1000):

        self.origin = origin
        self.destination = destination
        self.edge_cost = edge_cost
        self.value = float(self.destination.get_value() or 0) + self.edge_cost
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
        self.value = float(self.destination.get_value() or 0) + self.edge_cost

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

    pseudo_count_factor = 0

    def __init__(self, state, value=0):
        self.state = state
        self.visit_count = 1
        self.value = round(value + (1 * math.exp(-Node.pseudo_count_factor * self.visit_count)), 6) #(Node.pseudo_count_factor * (self.visit_count ** -1))

    def get_value(self):
        return self.value

    def get_state(self):
        return self.state

    def set_value(self, value):
        self.value = round(value + (1 * math.exp(-Node.pseudo_count_factor * self.visit_count)), 6) #(Node.pseudo_count_factor * (self.visit_count ** -1))

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

    def edge_update(self, old_node, new_node, reward):

        if old_node != new_node:
            current_edge = Edge(old_node, new_node)
            if current_edge not in self.edge_list:
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
            self.new_node_encontered = False
        else:
            node = self.node_list[self.node_list.index(old_node)]
            self.new_node_encontered = False

        node.visited() # to augment the visit counter

        self.current_node = node

    def abstract_state_discovery(self, sample):
        old_node = Node(sample[0]["manager"], 0)

        new_node = Node(sample[3]["manager"], sample[2])

        self.node_update(old_node, new_node, sample[2])
        self.edge_update(old_node, new_node, sample[2])

        return self.new_node_encontered

    def print_edge_list(self):
        for edge in self.edge_list:
            print(edge)

    def print_node_list(self):
        for node in self.node_list:
            print(node)

    def get_number_of_nodes(self):
        return len(self.node_list)

    def find_distances(self, root):
        """
        Bellman-Ford algorithm to get the longest path (highest value)
        :param root: the origin of the path
        :return: distances.
        """

        if len(self.node_list) > 0 and len(self.edge_list) > 0:
            distances = {}
            for node in self.node_list:
                distances[node] = - float("inf") # distances for all node setted to - inf
            distances[root] = 0. # the source distance is set to 0

            for _ in range(len(self.node_list) - 1):
                for edge in self.edge_list:
                    origin = edge.get_origin()
                    destination = edge.get_destination()

                    if distances[destination] < distances[origin] + edge.get_value():
                        distances[destination] = distances[origin] + edge.get_value()

            # Step 3: check for negative - weight cycles

            for edge in self.edge_list:
                origin = edge.get_origin()
                destination = edge.get_destination()
                if distances[destination] < (distances[origin] + edge.get_value()):
                    print( "Graph contains a negative-weight cycle")
            #        return None

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
