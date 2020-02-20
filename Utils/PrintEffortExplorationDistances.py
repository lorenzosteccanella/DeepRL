import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
import csv

def load_csv(file_name):
    with open(file_name, mode='r') as infile:
        reader = csv.reader(infile)
        mydict = {rows[0]: rows[1] for rows in reader}

        return mydict

    return False

def load_pkl(file_name):
    with open(file_name, 'rb') as file:
        mydict = pickle.load(file)

        return mydict
    return False

def distance_vectors(v1, v2):
    return np.linalg.norm(v1 - v2) # l2 norm

dictionary_effort_exploration = load_pkl("../ExplorationEffort.pkl")

print(dictionary_effort_exploration.keys())

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

            if s1s2 in dictionary_effort_exploration.keys():

                distance = distance_vectors(starting_point_EE, dictionary_effort_exploration[(s1s2)][1])

                print(str(round(distance, 5)).ljust(5), end=" , ")
                np_EE[j, i, k] = distance
            else:
                np_EE[j, i, k] = -1
                print(str(None).ljust(5), end=" , ")

        print("")

N = 0.1

print("")
print("")
print("CLUSTERED DIFFUSION TIME WITH N = " + str(N) )
print("")

for k in range(depth):
    print("")
    print("-"*30+"  "+str(k)+"  "+30*"-")
    print("")
    for i in range(width):
        for j in range(height):
            s1s2 = (starting_point, (j, i, k))

            if s1s2 in dictionary_effort_exploration.keys():

                distance = distance_vectors(starting_point_EE, dictionary_effort_exploration[(s1s2)][1])

                cluster_value = math.modf(distance / N)[1]
                np_clustered_EE[j,i,k] = cluster_value
                print(str(cluster_value).ljust(5), end=" , ")
            else:
                np_clustered_EE[j,i,k] = -1
                print(str(None).ljust(5), end=" , ")

        print("")

plt.imshow(np_EE[:,:,0])
plt.show()
plt.imshow(np_EE[:,:,1])
plt.show()

plt.imshow(np_clustered_EE[:,:,0])
plt.show()
plt.imshow(np_clustered_EE[:,:,1])
plt.show()