import abc


class AbstractOption(abc.ABC):

    id = 0

    def __init__(self):
        AbstractOption.id += 1
        self.edge_list = []

    @abc.abstractmethod
    def act(self, s):
        pass

    @abc.abstractmethod
    def observe(self, s):
        pass

    def observe_online(self, s):
        pass

    def observe_imitation(self, s):
        pass

    def getID(self):
        return self.id

    def add_edge(self, edge):
        if edge not in self.edge_list:
            self.edge_list.append(edge)

    def get_edge_list(self):
        return self.edge_list

    def get_state_value(self, s):
        pass
