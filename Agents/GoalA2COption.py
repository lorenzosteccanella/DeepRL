from Agents.AbstractOption import AbstractOption
from Agents import GoalA2CAgent
from Models.GoalA2CnetworksEager import *

class GoalA2COption(AbstractOption):

    def __init__(self, parameters):

        super(GoalA2COption, self).__init__()

        self.id = self.getID()

        self.a2cDNN = GoalA2CEagerSync(parameters["h_size"], len(parameters["action_space"]), parameters["critic_network"],
                                   parameters["actor_network"], parameters["learning_rate"], parameters["weight_mse"],
                                   parameters["weight_ce_exploration"], parameters["shared_representation"],
                                   parameters["learning_rate_reduction_obs"], parameters["shared_goal_representation"])

        self.agent = GoalA2CAgent(parameters["action_space"], self.a2cDNN, parameters["gamma"], parameters["batch_size"])

        self.preprocessing = parameters["preprocessing"]

        self.parameters = parameters


    def act(self, s):
        state = s[0]
        goal = s[1]
        if self.preprocessing:
            s = self.preprocessing.preprocess_image(state)
            g = self.preprocessing.preprocess_image(goal)
        else:
            s = state
            g = goal

        return self.agent.act(s, g)

    def observe(self, sample):  # in (s, a, r, s_, done, info, goal) format
        if self.preprocessing:
            s = self.preprocessing.preprocess_image(sample[0])
            s_ = self.preprocessing.preprocess_image(sample[3])
            g = self.preprocessing.preprocess_image(sample[6])
        else:
            s = sample[0]
            s_ = sample[3]
            g = sample[6]
        sample = (s, sample[1], sample[2], s_, sample[4], sample[5], g)
        #print(sample[2])
        self.agent.observe(sample)
        self.agent.replay()
        if self.preprocessing:
            if sample[4]:
                self.preprocessing.reset(sample[4])

    def __str__(self):
        return "option " + str(self.id)

    def get_state_value(self, s):
        pass

    def get_ce_loss(self):
        return self.agent.ce_loss
