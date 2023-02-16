import gym
from Utils import normalize, get_pixels_from_obs, np_in_list, ssim_in_list, SSIM_equal, sample_colors, make_gray_scale, hash_numpy_array_equal, make_downsampled_image
from collections import deque
import numpy as np
import math

_hex_to_int = lambda x:int(x, 16)-128

class Montezuma_position_wrapper_only_1key(gym.Wrapper):


    def __init__(self, env, parameters, noop_max=30):

        super().__init__(env)
        self.parameters = parameters
        self.n_zones = self.parameters["n_zones"] # the parameter to which we will divide the position
        self.images_stack = deque([], maxlen=self.parameters["stack_images_length"])
        self.direction_of_skull = 0
        self.old_position = None

        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        self.reset_FLAG = False
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def NoopReset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def FireReset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def reset(self, **kwargs):
        self.images_stack.clear()
        observation = self.NoopReset(**kwargs)
        #observation = self.FireReset(**kwargs)
        observation, ram = self.observation(observation)

        observation["manager"] = self.get_position_montezuma(ram, 0., False)

        return observation

    def step(self, action):

        acc_reward = 0
        observation_ram = self.env.unwrapped._get_ram()
        if observation_ram[56] != 0:
            self.env.step(0)

        for i in range(4):
            self.env.unwrapped._get_image()
            observation, reward, done, info = self.env.step(action)
            acc_reward += reward           
            if done:
                break

        if self.has_key(observation):
            reward = 100
        else:
            reward = acc_reward

        if reward == 100:
            done = True

        if reward == 100:
            done = True

        observation, ram = self.observation(observation)

        if self.reset_FLAG and self.has_key(ram):
            done = True
            self.reset_FLAG = False
        
        if self.reset_FLAG and not self.has_key(ram):
            self.reset_FLAG = False

        if self.has_key(ram):
            self.reset_FLAG = True

        observation["manager"] = self.get_position_montezuma(ram, reward, done)

        return observation, reward, done, info

    def is_even(self, x):
        _, r = divmod(x, 2)
        return r == 0

    def scale_obs(self, x, y):

        x = (x - 9)
        y = (y - 148)

        return x, y

    def to_network(self, x, y):

        max_x = 145 - 9
        max_y = 252 - 148

        x = (x / (max_x)) * 100
        y = (y / (max_y)) * 100

        return x, y

    def observation(self, observation):
        ram = observation

        pixels = self.env.unwrapped._get_image()
        self.width, self.height, self.depth = pixels.shape

        obs_option = self.get_obs(ram)

        # render it
        return {"vanilla": pixels, "manager": None, "option": obs_option}, ram

    def get_obs(self, ram):

        x, y = self.get_xy(ram)

        obs_option = self.to_network(x, y)
        xy_skull, self.direction_of_skull = self.get_skull(ram, self.direction_of_skull)
        obs_option = self.stack_obs((obs_option[0], obs_option[1]))
        return obs_option

    def stack_obs(self, obs):
        obs_option = np.asarray(obs)
        self.images_stack.append(obs_option)

        shape_obs = obs_option.shape

        img_option_stacked = np.zeros((shape_obs[-1] * self.parameters["stack_images_length"]), dtype=np.float32)
        index_image = 0
        for i in range(0, self.images_stack.__len__()):
            img_option_stacked[index_image:index_image + shape_obs[-1]] = self.images_stack[i]
            index_image = index_image + shape_obs[-1]

        return img_option_stacked

    def get_xy(self, obs):

        x = obs[_hex_to_int('0xAA')]
        y = obs[_hex_to_int('0xAB')]
        x, y = self.scale_obs(x, y)
        return x, y

    def get_skull(self, obs, direction):
        pos = obs[_hex_to_int('0xAF')] - _hex_to_int('0x16')
        if pos == 0:
            direction = 0
        elif pos == 50:
            direction = 1
        return pos, direction

    def has_key(self, obs):
        idx = _hex_to_int('0xc1')
        inventory = obs[idx]
        key = _hex_to_int('0x1e')
        _has_key = (inventory & key) != 0
        return _has_key

    def get_position_montezuma(self, ram, reward, done):

        step_x = self.n_zones #int(self.width / self.n_zones)
        step_y = self.n_zones #int(self.height / self.n_zones)

        x, y = self.get_xy(ram)
        key = int(self.has_key(ram))

        x = x if self.is_even(x) else x + 1
        y = y if self.is_even(y) else y + 1

        return (x//step_x, y//step_y, key)
