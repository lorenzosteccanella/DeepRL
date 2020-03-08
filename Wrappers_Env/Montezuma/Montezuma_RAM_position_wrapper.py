import gym
from Utils import normalize, get_pixels_from_obs, np_in_list, ssim_in_list, SSIM_equal, sample_colors, make_gray_scale, hash_numpy_array_equal, make_downsampled_image
from collections import deque
import numpy as np
import math

_hex_to_int = lambda x:int(x, 16)-128

class Montezuma_RAM_position_wrapper(gym.Wrapper):


    def __init__(self, env, parameters):

        super().__init__(env)
        self.parameters = parameters
        self.n_zones = self.parameters["n_zones"] # the parameter to which we will divide the position
        self.images_stack = deque([], maxlen=self.parameters["stack_images_length"])
        self.KEY = False
        self.total_reward = 0
        self.old_position = None

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        observation = self.observation(observation)

        observation["manager"] = self.get_position_montezuma(observation, 0, False)
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        observation = self.observation(observation)

        observation["manager"] = self.get_position_montezuma(observation, reward, done)

        return observation, reward, done, info

    def observation(self, observation):
        ram = observation

        # option observation
        ram_option = ram.copy()

        # render it
        return {"vanilla": ram, "manager": None, "option": ram_option}

    def get_xy(self, obs):

        x = obs["vanilla"][_hex_to_int('0xAA')]
        y = obs["vanilla"][_hex_to_int('0xAB')]
        return x, y

    def get_position_montezuma(self, ram, reward, done):

        if done:
            self.total_reward = 0
            self.old_position = None

        self.total_reward += reward

        x, y = self.get_xy(ram)

        d_x = math.modf(x / self.n_zones)[1] # you have to get the int of this

        d_y = math.modf(y / self.n_zones)[1] # you have to get the int of this

        if self.old_position:

            if self.old_position != (d_x, d_y, self.total_reward):
                print(d_x, d_y, self.total_reward)

        else:

            print(d_x, d_y, self.total_reward)

        self.old_position = (d_x, d_y, self.total_reward)

        return (d_x, d_y, self.total_reward)

        #if total_reward > 0 :
        #    return (d_x, d_y, reward)

        #else :
        #    return (d_x, d_y, 0)
