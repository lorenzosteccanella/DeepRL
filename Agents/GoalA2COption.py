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
        start = s[1]
        goal = s[2]
        if self.preprocessing:
            s = self.preprocessing.preprocess_image(state)
            i = self.preprocessing.preprocess_image(start)
            g = self.preprocessing.preprocess_image(goal)
        else:
            s = state
            i = start
            g = goal

        return self.agent.act(s, i, g)

    def observe(self, sample):  # in (s, a, r, s_, done, info, goal) format
        if self.preprocessing:
            s = self.preprocessing.preprocess_image(sample[0])
            s_ = self.preprocessing.preprocess_image(sample[3])
            i = self.preprocessing.preprocess_image(sample[6])
            g = self.preprocessing.preprocess_image(sample[7])
        else:
            s = sample[0]
            s_ = sample[3]
            i = sample[6]
            g = sample[7]
        sample = (s, sample[1], sample[2], s_, sample[4], sample[5], i, g)
        #print(sample[2])
        self.agent.observe(sample)
        self.agent.replay()
        if self.preprocessing:
            if sample[4]:
                self.preprocessing.reset(sample[4])

    def __str__(self):
        return "option " + str(self.id)

    def get_state_value(self, s):
        state = s[0]
        start = s[1]
        goal = s[2]
        if self.preprocessing:
            s = self.preprocessing.preprocess_image(state)
            i = self.preprocessing.preprocess_image(start)
            g = self.preprocessing.preprocess_image(goal)
        else:
            s = state
            i = start
            g = goal

        return self.a2cDNN.prediction_critic([s],[i],[g])[0][0]

    def get_ce_loss(self):
        return self.agent.ce_loss
