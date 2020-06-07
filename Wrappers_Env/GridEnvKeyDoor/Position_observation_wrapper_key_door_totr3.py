import gym
from Utils import normalize, get_pixels_from_obs, np_in_list, ssim_in_list, SSIM_equal, sample_colors, make_gray_scale, hash_numpy_array_equal, make_downsampled_image
from collections import deque
import numpy as np
import gym.spaces as spaces
import random

class Position_observation_wrapper_key_door_totr3(gym.Wrapper):


    def __init__(self, env, parameters):

        super().__init__(env)
        self.parameters = parameters
        self.width = self.parameters["width"]
        self.height = self.parameters["height"]
        self.n_zones = self.parameters["n_zones"]
        self.total_reward = 0.
        self.Key = 0
        self.Door = 0

    def reset(self, **kwargs):

        self.Key = 0
        self.Door = 0
        self.total_reward = 0.

        obs, info = self.env.reset(**kwargs)
        observation = self.get_position(info["position"], 0.)

        observation["vanilla"] = obs
        observation["manager"] = self.get_position_abstract_state_gridenv_GE_MazeKeyDoor_v0(info["position"], 0., False)

        return observation

    def step(self, action):

        if random.random() > 0.8:
            a = action
            while a == action:
                action = self.env.action_space.sample()

        obs, reward, done, info = self.env.step(action)
        observation = self.get_position(info["position"], reward)

        observation["vanilla"] = obs
        observation["manager"] = self.get_position_abstract_state_gridenv_GE_MazeKeyDoor_v0(info["position"], reward, done)

        if self.total_reward >= 1 and reward == 1:
            reward = 1.
        elif reward > 0.:
            reward = 0.

        return observation, reward, done, info

    def normalize_position(self, x, y):
        x = ((x / (self.width -1)) * 10) + 1
        y = ((y / (self.height -1)) * 10) + 1

        return x, y

    def get_position(self, position, reward):

        x = position[0]
        y = position[1]

        x, y = self.normalize_position(x, y)

        return {"vanilla": (x, y, self.Key), "manager": None, "option": (x, y)}

    def get_position_abstract_state_gridenv_GE_MazeKeyDoor_v0(self, position, reward, done):

        if reward > 0:
            self.total_reward += reward

        step_x = self.width // self.n_zones
        step_y = self.height // self.n_zones

        x = position[0]
        y = position[1]

        return (x//step_x, y//step_y, self.total_reward)
