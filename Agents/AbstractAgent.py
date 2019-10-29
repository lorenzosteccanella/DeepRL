import abc


class AbstractAgent(abc.ABC):

    @abc.abstractmethod
    def act(self, s):
        pass

    @abc.abstractmethod
    def observe(self, s):
        pass

    @abc.abstractmethod
    def replay(self):
        pass