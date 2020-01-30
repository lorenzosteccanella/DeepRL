import gym
from Utils import normalize, get_pixels_from_obs, np_in_list, ssim_in_list, SSIM_equal, sample_colors, make_gray_scale, hash_numpy_array_equal, make_downsampled_image
from collections import deque
import numpy as np

class Origin_wrapper(gym.Wrapper):


    def __init__(self, env, parameters = None):

        super().__init__(env)
        self.parameters = parameters

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info