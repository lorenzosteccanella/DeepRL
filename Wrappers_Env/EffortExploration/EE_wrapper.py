import gym
from Utils import normalize, get_pixels_from_obs, np_in_list, ssim_in_list, SSIM_equal, sample_colors, make_gray_scale, hash_numpy_array_equal, make_downsampled_image
from collections import deque
import numpy as np
from numpy import array_equal
import matplotlib.pyplot as plt
from time import sleep

class EE_wrapper(gym.Wrapper):


    def __init__(self, env, parameters):

        super().__init__(env)
        self.parameters = parameters
        self.nn = self.parameters["nn"]
        self.distance_cluster = self.parameters["distance_cluster"]
        self.images_stack = deque([], maxlen=self.parameters["stack_images_length"])
        self.list_of_repr_states = []
        self.KEY = False
        self.total_r = 0.

    def reset(self, **kwargs):
        self.images_stack.clear()
        self.total_r = 0.

        observation = self.env.reset(**kwargs)
        observation = self.observation(observation)

        observation["manager"] = self.get_abstract_state(normalize(observation["vanilla"]), 0.)

        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        observation = self.observation(observation)

        observation["manager"] = self.get_abstract_state(normalize(observation["vanilla"]), reward)

        return observation, reward, done, info

    def observation(self, observation):
        image = observation

        #preprocessed observation
        img_ = self.get_obs(image.copy())

        # render it
        return {"vanilla": image, "manager": None, "option": img_}

    def get_obs(self, image):
        img_option = normalize(image)
        self.images_stack.append(img_option)

        img_option_stacked = np.zeros((img_option.shape[0], img_option.shape[1],
                                       img_option.shape[2] * self.parameters["stack_images_length"]), dtype=np.float32)
        index_image = 0
        for i in range(0, self.images_stack.__len__()):
            img_option_stacked[..., index_image:index_image + img_option.shape[2]] = self.images_stack[i]
            index_image = index_image + img_option.shape[2]

        return img_option_stacked

    # test for exact equality
    def arreq_in_list(self, myarr, list_arrays):
        return next((True for elem in list_arrays if array_equal(elem, myarr)), False)

    def get_abstract_state(self, s, r):

        self.total_r += r

        if not self.list_of_repr_states:
            self.list_of_repr_states.append(s)

        distances = []
        for state in self.list_of_repr_states:
            distances.append(np.linalg.norm(self.nn.prediction_distance([state], [s])))

        d = min(distances)

        index = distances.index(d)

        if d>self.distance_cluster:
            self.list_of_repr_states.append(s)
            index = len(self.list_of_repr_states) - 1 # just to save computation otherwise self.list_of_repr_states.index(s)
            plt.imshow(s)
            plt.show(block=False)
            sleep(2)
            plt.close()

        return (index, self.total_r) #self.list_of_repr_states[index] # index
