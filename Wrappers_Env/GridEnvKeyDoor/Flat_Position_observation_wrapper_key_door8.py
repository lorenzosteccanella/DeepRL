import gym
from Utils import normalize, get_pixels_from_obs, np_in_list, ssim_in_list, SSIM_equal, sample_colors, make_gray_scale, hash_numpy_array_equal, make_downsampled_image
from collections import deque
import numpy as np
import gym.spaces as spaces

class Flat_Position_observation_wrapper_key_door8(gym.Wrapper):


    def __init__(self, env, parameters):

        super().__init__(env)
        self.parameters = parameters
        self.width = self.parameters["width"]
        self.height = self.parameters["height"]
        self.total_reward = 0.
        self.Key = 0
        self.Door = 0

    def reset(self, **kwargs):

        self.Key = 0
        self.Door = 0
        self.total_reward = 0.

        obs = self.env.reset(**kwargs)
        observation = self.get_position(None, 0.)
        return observation

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        observation = self.get_position(info["position"], reward)

        return observation, reward, done, info

    def normalize_position(self, x, y):
        x = ((x / (self.width -1)) * 10) + 1
        y = ((y / (self.height -1)) * 10) + 1

        return x, y

    def get_position(self, position, reward):
        if reward > 0:
            self.total_reward += reward

        # if self.total_reward == 1:
        #     self.Key = 1
        #
        # if self.total_reward == 2:
        #     self.Door = 1

        if position is None:
            x = 1
            y = 6
        else:
            x = position[0]
            y = position[1]

        x, y = self.normalize_position(x, y)

        return (x, y, self.Key, self.total_reward)