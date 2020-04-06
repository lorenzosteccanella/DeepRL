from Agents.AbstractOption import AbstractOption
from Agents import DoubleDQNAgent
from Models.QnetworksEager import *
from copy import deepcopy

class DoubleDQNOption(AbstractOption):

    def __init__(self, parameters):

        super(DoubleDQNOption, self).__init__()

        self.id = self.getID()

        self.MainQlearningDNN = QnetworkEager(parameters["h_size"], len(parameters["action_space"]), parameters["head_model"],
                                    parameters["learning_rate"], parameters["main_shared_representation"],
                                    parameters["learning_rate_reduction_obs"])

        self.TargetQlearningDNN = QnetworkEager(parameters["h_size"], len(parameters["action_space"]), parameters["head_model"],
                                    parameters["learning_rate"], parameters["target_shared_representation"],
                                    parameters["learning_rate_reduction_obs"])

        self.buffer = parameters["buffer_type"](parameters["buffer_size"])

        self.update_fn = parameters["update_fn"](weights=self.MainQlearningDNN, model=self.TargetQlearningDNN, tau=parameters["tau"])

        self.agent = DoubleDQNAgent(parameters["action_space"], self.buffer, self.MainQlearningDNN, self.TargetQlearningDNN,
                                    parameters["lambda"], parameters["update_target_freq"], self.update_fn, parameters["gamma"],
                                    parameters["batch_size"], parameters["min_epsilon"])

        self.preprocessing = deepcopy(parameters["preprocessing"]) # remember these are options u need to use deepcopy
        self.preprocessing1 = deepcopy(parameters["preprocessing"]) # remember these are options u need to use deepcopy
        self.preprocessing2 = deepcopy(parameters["preprocessing"]) # remember these are options u need to use deepcopy

        self.parameters = parameters


    def act(self, s):
        if self.preprocessing:
            s = self.preprocessing.preprocess_image(s)

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
        self.agent.replay()
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

        predictions = self.MainQlearningDNN.Qprediction([s])

        return max(predictions)

    def get_ce_loss(self):
        pass
