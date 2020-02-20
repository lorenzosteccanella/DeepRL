import random
from Agents.AbstractAgent import AbstractAgent
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os




class ExplorationEffortRepresentativeStates(AbstractAgent):

    exp = 0

    def __init__(self, action_space, nn, distance_cluster, name):
        self.action_space = action_space
        self.nn = nn
        self.list_of_repr_states = []
        self.distance_cluster = distance_cluster
        self.directory="results/Exploration_Effort_Representative_states_"+ name + "_" + str(self.distance_cluster)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def act(self, s):
        return random.choice(self.action_space)

    def distance_vectors(v1, v2):
        return np.linalg.norm(v1 - v2)  # l2 norm

    def observe(self, sample):  # in (s, a, r, s_) format
        if not self.list_of_repr_states:
            self.list_of_repr_states.append(sample[0])
            matplotlib.image.imsave(self.directory+"/"+str(len(self.list_of_repr_states))+'.png', sample[0])

        s1 = sample[3]

        distances = []
        for state in self.list_of_repr_states:
            distances.append(np.linalg.norm(self.nn.prediction_distance([state], [s1])))

        d = min(distances)



        if d>self.distance_cluster :
            self.list_of_repr_states.append(s1)
            matplotlib.image.imsave(self.directory+"/"+str(len(self.list_of_repr_states))+'.png', s1)

            # distances = []
            # for state in self.list_of_repr_states:
            #     d1 = self.distance_vectors(self.nn.prediction_distance([state], [s]) - self.nn.prediction_distance([state], [s1]))
            #     d2 = self.distance_vectors(self.nn.prediction_distance([s], [state]) - self.nn.prediction_distance([s1], [state]))
            #     distances.append(max([d1,d2]))
            #
            # d = max(distances)
            #
            # if d > self.distance_cluster:
            #     self.list_of_repr_states.append(s)
            #     plt.imshow(s)
            #     plt.show()

        self.exp += 1

    def replay(self):
        pass
