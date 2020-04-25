import gym
from Utils import normalize, get_pixels_from_obs, np_in_list, ssim_in_list, SSIM_equal, sample_colors, make_gray_scale, hash_numpy_array_equal, make_downsampled_image
from collections import deque
import numpy as np
import math

_hex_to_int = lambda x:int(x, 16)-128

class Montezuma_position_wrapper_only_1key(gym.Wrapper):


    def __init__(self, env, parameters):

        super().__init__(env)
        self.parameters = parameters
        self.n_zones = self.parameters["n_zones"] # the parameter to which we will divide the position
        self.images_stack = deque([], maxlen=self.parameters["stack_images_length"])
        self.total_reward = 0.
        self.direction_of_skull = 0
        self.old_position = None

    def reset(self, **kwargs):
        self.images_stack.clear()
        self.total_reward = 0.
        observation = self.env.reset(**kwargs)
        observation, ram = self.observation(observation)

        observation["manager"] = self.get_position_montezuma(ram, 0., False)
        return observation

    def step(self, action):

        acc_reward = 0
        for i in range(4):
            observation, reward, done, info = self.env.step(action)
            acc_reward += reward
            if done:
                break

        reward = acc_reward
        if reward == 100:
            done = True

        observation, ram = self.observation(observation)

        observation["manager"] = self.get_position_montezuma(ram, reward, done)

        return observation, reward, done, info

    def observation(self, observation):
        ram = observation

        pixels = self.env.unwrapped._get_image()
        self.width, self.height, self.depth = pixels.shape

        obs_option = self.get_obs(ram)

        # render it
        return {"vanilla": pixels, "manager": None, "option": obs_option}, ram

    def get_obs(self, ram):

        x, y = self.get_xy(ram)
        #xy_skull, self.direction_of_skull = self.get_skull(ram, self.direction_of_skull)

        return (x, y)


    def get_xy(self, obs):

        x = obs[_hex_to_int('0xAA')]
        y = obs[_hex_to_int('0xAB')]
        return x, y

    def get_skull(self, obs, direction):
        pos = obs[_hex_to_int('0xAF')] - _hex_to_int('0x16')
        if pos == 0:
            direction = 0
        elif pos == 50:
            direction = 1
        return pos, direction

    def get_position_montezuma(self, ram, reward, done):

        self.total_reward += reward

        step_x = self.n_zones #int(self.width / self.n_zones)
        step_y = self.n_zones #int(self.height / self.n_zones)

        x, y = self.get_xy(ram)

        d_x = x//step_x
        d_y = y//step_y

        # if self.old_position:
        #
        #     if self.old_position != (d_x, d_y, self.total_reward):
        #         print(d_x, d_y, self.total_reward)
        #
        # else:
        #
        #     print(d_x, d_y, self.total_reward)

        #print(x, y, s)

        return (x//step_x, y//step_y, self.total_reward, reward)
