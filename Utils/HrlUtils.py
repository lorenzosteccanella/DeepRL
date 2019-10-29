class Edge:

    def __init__(self, origin, destination, value=0, edge_cost = -0.001):

        self.origin = origin
        self.destination = destination
        self.edge_cost = edge_cost
        self.value = float(destination.get_value() or 0) + self.edge_cost
        self.option = None

    def get_value(self):
        return self.value

    def get_origin(self):
        return self.origin

    def get_destination(self):
        return self.destination

    def set_value(self, value):
        self.value = value + self.edge_cost

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

    def __init__(self, state, value=0):
        self.state = state
        self.value = value
        self.visit_count = 1

    def get_value(self):
        return self.value

    def get_state(self):
        return self.state

    def set_value(self, value):
        self.value = value

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


class Graph:

    def __init__(self):
        self.edge_list = []
        self.node_list = []
        self.current_node = None
        self.current_edge = None

    def edge_update(self, old_node, new_node, reward):

        if old_node != new_node:
            current_edge = Edge(old_node, new_node)
            if current_edge not in self.edge_list:
                self.edge_list.append(current_edge)
                #current_edge.set_value(reward)
                self.current_edge = current_edge
            else:
                edge = self.edge_list[self.edge_list.index(current_edge)]
                #edge.update_value(reward)
                self.current_edge = edge

    def node_update(self, old_node, new_node=None, reward=None):
        if old_node not in self.node_list:
            self.node_list.append(old_node)

        if new_node:
            if new_node not in self.node_list:
                self.node_list.append(new_node)
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

    def print_edge_list(self):
        for edge in self.edge_list:
            print(edge)

    def print_node_list(self):
        for node in self.node_list:
            print(node)

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
                    return None

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
