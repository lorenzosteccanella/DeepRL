from Agents.AbstractOption import AbstractOption
from Agents import A2CAgent
from Models.A2CnetworksEager import *
from copy import deepcopy

class A2COption(AbstractOption):

    def __init__(self, parameters):

        super(A2COption, self).__init__()

        self.id = self.getID()

        self.a2cDNN = A2CEagerSeparate(parameters["h_size"], len(parameters["action_space"]), parameters["critic_network"],
                                   parameters["actor_network"], parameters["learning_rate"], parameters["weight_mse"],
                                   parameters["weight_ce_exploration"], parameters["shared_representation"], parameters["learning_rate_reduction_obs"])

        self.agent = A2CAgent(parameters["action_space"], self.a2cDNN, parameters["gamma"], parameters["batch_size"])

        self.preprocessing = deepcopy(parameters["preprocessing"]) # remember these are options u need to use deepcopy
        self.preprocessing1 = deepcopy(parameters["preprocessing"]) # remember these are options u need to use deepcopy
        self.preprocessing2 = deepcopy(parameters["preprocessing"]) # remember these are options u need to use deepcopy

        self.parameters = parameters


    def act(self, s):
        if self.preprocessing:
            s = self.preprocessing.preprocess_image(s)

        #print(s, self.id)

        return self.agent.act(s)

    def observe(self, sample):  # in (s, a, r, s_, done, info) format
        if self.preprocessing:
            s = self.preprocessing1.preprocess_image(sample[0])
            s_ = self.preprocessing2.preprocess_image(sample[3])
        else:
            s = sample[0]
            s_ = sample[3]
        sample = (s, sample[1], sample[2], s_, sample[4], sample[5])
        self.agent.observe(sample)
        self.agent.replay(sample[4])
        if self.preprocessing:
            if sample[4]:
                self.preprocessing.reset(sample[4])
                self.preprocessing1.reset(sample[4])
                self.preprocessing2.reset(sample[4])

    def __str__(self):
        return "option " + str(self.id)

    def get_state_value(self, s):
        if self.preprocessing:
            s = self.preprocessing.preprocess_image(s)

        return self.a2cDNN.prediction_critic([s])[0][0]

    def get_ce_loss(self):
        return self.agent.ce_loss
