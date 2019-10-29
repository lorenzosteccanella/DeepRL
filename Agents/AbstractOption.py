import abc


class AbstractOption(abc.ABC):

    @abc.abstractmethod
    def act(self, s):
        pass

    @abc.abstractmethod
    def observe(self, s):
        pass