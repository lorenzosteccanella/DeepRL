import gym
from Utils import normalize, get_pixels_from_obs, np_in_list, ssim_in_list, SSIM_equal, sample_colors, make_gray_scale, hash_numpy_array_equal, make_downsampled_image
from collections import deque
import numpy as np
import math

_hex_to_int = lambda x:int(x, 16)-128

class Montezuma_Pixel_position_wrapper(gym.Wrapper):


    def __init__(self, env, parameters):

        super().__init__(env)
        self.parameters = parameters
        self.n_zones = self.parameters["n_zones"] # the parameter to which we will divide the position
        self.images_stack = deque([], maxlen=self.parameters["stack_images_length"])
        self.total_reward = 0.

    def reset(self, **kwargs):
        self.images_stack.clear()
        self.total_reward = 0.
        observation = self.env.reset(**kwargs)
        observation, ram = self.observation(observation)

        observation["manager"] = self.get_position_montezuma(ram, 0., False)
        return observation

    def step(self, action):

        acc_reward = 0
        for i in range(2):
            observation, reward, done, info = self.env.step(action)
            acc_reward += reward
            if done:
                break

        reward = acc_reward
        observation, ram = self.observation(observation)

        observation["manager"] = self.get_position_montezuma(ram, reward, done)
        return observation, reward, done, info

    def observation(self, observation):
        ram = observation

        pixels = self.env.unwrapped._get_image()
        self.width, self.height, self.depth = pixels.shape

        obs_option = self.get_obs(pixels)

        # render it
        return {"vanilla": pixels, "manager": None, "option": obs_option}, ram

    def get_obs(self, image):
        img_option = normalize(image)
        self.images_stack.append(img_option)

        img_option_stacked = np.zeros((img_option.shape[0], img_option.shape[1],
                                       img_option.shape[2] * self.parameters["stack_images_length"]), dtype=np.float32)
        index_image = 0
        for i in range(0, self.images_stack.__len__()):
            img_option_stacked[..., index_image:index_image + img_option.shape[2]] = self.images_stack[i]
            index_image = index_image + img_option.shape[2]

        # self.show_stacked_frames(img_option_stacked)

        return img_option_stacked

    def get_xy(self, obs):

        x = obs[_hex_to_int('0xAA')]
        y = obs[_hex_to_int('0xAB')]
        return x, y

    def get_position_montezuma(self, ram, reward, done):

        self.total_reward += reward

        step_x = self.n_zones #int(self.width / self.n_zones)
        step_y = self.n_zones #int(self.height / self.n_zones)

        x, y = self.get_xy(ram)

        s = (x//step_x, y//step_y)

        #print(x, y, s)

        return (s, self.total_reward, reward, done)
