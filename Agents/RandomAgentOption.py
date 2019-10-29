from Agents.AbstractOption import AbstractOption
from Agents import RandomAgent


class RandomAgentOption(AbstractOption):

    def __init__(self, action_space):
        self.agent = RandomAgent(action_space)

    def act(self, s):
        return self.agent.act(s)

    def observe(self, sample):
        return self.agent.observe(sample)
