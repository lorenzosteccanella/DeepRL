import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
import csv
import tensorflow as tf

tf.enable_eager_execution()

import sys
sys.path.append("..")

from Models.ExplorationEffortnetworksEager import *

ACTION_SPACE = [0, 1, 2, 3, 4]
learning_rate = 0.0001
observation = SharedDenseLayers()

nn = EffortExplorationNN(len(ACTION_SPACE), learning_rate, observation)
nn.load_weights()

def distance_vectors(v1, v2):
    return np.linalg.norm(v1 - v2) # l2 norm


starting_point = (1, 8, 0)
starting_point_EE = np.zeros(5)

width = 10
height = 10
depth = 2
np_EE = np.zeros((width, height, depth))
np_clustered_EE = np.zeros((width, height, depth))


print("")
print("ORIGIN EE DISTANCES")
print("")

for k in range(depth):
    print("")
    print("-"*30+"  "+str(k)+"  "+30*"-")
    print("")
    for i in range(width):
        for j in range(height):

            s1s2 = (starting_point, (j, i, k))

            distance = distance_vectors(starting_point_EE, nn.prediction_distance([starting_point], [(j,i,k)]))

            print(str(round(distance, 5)).ljust(5), end=" , ")
            np_EE[j, i, k] = distance

        print("")

plt.imshow(np_EE[:,:,0])
plt.show()
plt.imshow(np_EE[:,:,1])
plt.show()