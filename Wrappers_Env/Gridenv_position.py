import gym
from Utils import normalize, get_pixels_from_obs, np_in_list, ssim_in_list, SSIM_equal, sample_colors, make_gray_scale, hash_numpy_array_equal, make_downsampled_image
from collections import deque
import numpy as np

class Gridenv_position(gym.Wrapper):


    def __init__(self, env, parameters):

        super().__init__(env)
        self.parameters = parameters
        self.width = self.parameters["width"]
        self.height = self.parameters["height"]

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.get_position_gridenv_GE_MazeKeyDoor_v0(None)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = self.get_position_gridenv_GE_MazeKeyDoor_v0(info["position"])
        return observation, reward, done, info

    def get_position_gridenv_GE_MazeKeyDoor_v0(self, position):

        if position is None:
            return (1, self.height-2)

        else:

            x = position[0]
            y = position[1]

            return (x,y)