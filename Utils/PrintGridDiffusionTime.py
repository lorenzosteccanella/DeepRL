import pickle
import math
import numpy as np
import matplotlib.pyplot as plt

with open('../step_distances_position_observation.pkl', 'rb') as file:
    dictionary_diffusion_time = pickle.load(file)

width = 30
height = 30
depth = 2
np_diffusion = np.zeros((width, height, depth))
np_clustered_diffusion = np.zeros((width, height, depth))

print("")
print("ORIGINAL DIFFUSION TIMES")
print("")
for k in range(depth):
    print("")
    print("-"*30+"  "+str(k)+"  "+30*"-")
    print("")
    for i in range(width):
        for j in range(height):
            if (j,i,k) in dictionary_diffusion_time.keys():
                np_diffusion[j, i, k] = dictionary_diffusion_time[(j,i,k)][0]
                print(str(dictionary_diffusion_time[(j,i,k)][0]).ljust(5), end=" , ")
            else:
                np_diffusion[j, i, k] = -1
                print(str(None).ljust(5), end=" , ")

        print("")


N = 10

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
            if (j,i,k) in dictionary_diffusion_time.keys():
                cluster_value = math.modf(dictionary_diffusion_time[(j,i,k)][0] / N)[1]
                np_clustered_diffusion[j,i,k] = cluster_value
                print(str(cluster_value).ljust(5), end=" , ")
            else:
                np_clustered_diffusion[j,i,k] = -1
                print(str(None).ljust(5), end=" , ")

        print("")


if True:
    np.save("../Wrappers_Env/Diffusion_Cluster_"+str(N), np_clustered_diffusion)


plt.imshow(np_clustered_diffusion[:,:,0])
plt.show()
plt.imshow(np_clustered_diffusion[:,:,1])
plt.show()
