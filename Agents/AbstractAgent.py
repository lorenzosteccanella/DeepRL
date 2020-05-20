import abc


class AbstractAgent(abc.ABC):

    FILE_NAME = ""

    @abc.abstractmethod
    def act(self, s):
        pass

    @abc.abstractmethod
    def observe(self, s):
        pass

    @abc.abstractmethod
    def replay(self, done):
        pass

    def set_name_file_2_save(self, filename):
        self.FILE_NAME = filename + " - "