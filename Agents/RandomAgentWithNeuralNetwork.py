import random
from Agents.AbstractAgent import AbstractAgent


class RandomAgentWithNeuralNetwork(AbstractAgent):

    exp = 0

    def __init__(self, action_space, main_model_nn, buffer=None):
        self.action_space = action_space
        self.buffer = buffer
        self.main_model_nn = main_model_nn

    def act(self, s):
        return random.choice(self.action_space)

    def observe(self, sample):  # in (s, a, r, s_) format
        # error = abs(sample[2])  # reward
        if self.buffer is not None:
            self.buffer.add(sample)
        self.exp += 1

    def replay(self):
        pass

    def get_observation_encoding(self, s):
        return self.main_model_nn.prediction_h([s])