import gym
from Utils import normalize, get_pixels_from_obs, np_in_list, ssim_in_list, SSIM_equal, sample_colors, make_gray_scale, hash_numpy_array_equal, make_downsampled_image
from collections import deque
import numpy as np
import gym.spaces as spaces

class Position_observation_wrapper(gym.Wrapper):


    def __init__(self, env, parameters):

        super().__init__(env)
        self.parameters = parameters
        self.width = self.parameters["width"]
        self.height = self.parameters["height"]
        self.n_zones = self.parameters["n_zones"]
        self.total_reward = 0.
        self.Key = 0

    def reset(self, **kwargs):

        self.Key = 0
        self.total_reward = 0.

        obs = self.env.reset(**kwargs)
        observation = self.get_position(None, 0.)

        observation["vanilla"] = obs
        observation["manager"] = self.get_position_abstract_state_gridenv_GE_MazeKeyDoor_v0(None, 0., False)

        return observation

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        observation = self.get_position(info["position"], reward)

        observation["vanilla"] = obs
        observation["manager"] = self.get_position_abstract_state_gridenv_GE_MazeKeyDoor_v0(info["position"], reward, done)

        return observation, reward, done, info

    def get_position(self, position, reward):

        if reward > 0:
            self.Key = 1

        if position is None:
            x = 1
            y = 8
        else:
            x = position[0]
            y = position[1]

        return {"vanilla": (x, y, self.Key), "manager": None, "option": (x, y, self.Key)}

    def get_position_abstract_state_gridenv_GE_MazeKeyDoor_v0(self, position, reward, done):

        self.total_reward += reward

        step_x = int(self.width / self.n_zones)
        step_y = int(self.height / self.n_zones)

        #initial state returned when the environments is resetted
        if position is None:
            x = 1
            y = 8
        else:
            x = position[0]
            y = position[1]

        s = (x//step_x, y//step_y)

        return (x//step_x, y//step_y, self.total_reward, reward)