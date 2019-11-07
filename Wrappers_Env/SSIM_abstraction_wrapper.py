import gym
from Utils import normalize, get_pixels_from_obs, np_in_list, ssim_in_list, SSIM_equal, sample_colors, make_gray_scale, hash_numpy_array_equal, make_downsampled_image
from collections import deque
import numpy as np

class PositionGridenv_GE_MazeKeyDoor_v0(gym.Wrapper):


    def __init__(self, env, parameters):

        super().__init__(env)
        self.parameters = parameters
        self.images_stack = deque([], maxlen=self.parameters["stack_images_length"])
        self.KEY = False

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        observation = self.observation(observation)

        observation["manager"] = self.get_position_abstract_state_gridenv_GE_MazeKeyDoor_v0(None, None)
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        observation = self.observation(observation)

        observation["manager"] = self.get_position_abstract_state_gridenv_GE_MazeKeyDoor_v0(info["position"], reward)

        return observation, reward, done, info

    def observation(self, observation):
        image = observation

        # option observation
        img_option = self.get_option_obs(image.copy())

        # render it
        return {"vanilla": image, "manager": None, "option": img_option}

    def get_option_obs(self, image):
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

    def get_position_abstract_state_gridenv_GE_MazeKeyDoor_v0(self, position, reward):
        #initial state returned when the environments is resetted
        if position is None:
            self.KEY = False
            return "abstract state 1"

        x = position[0]
        y = position[1]
        r = reward

        # dying
        if x == 0 or y == 0:
            return "died"
        # dying
        elif x == 9 or y == 9:
            return "died"

        if self.KEY == False:
            # key taken
            if x == 8 and y == 1 and r == 1:
                self.KEY = True
                return "key taken"
            # door close
            elif x == 1 and y == 1:
                return "door close"
            #abstract state 4
            elif x <= 8 / 2 and y <= 8 / 2:
                return "abstract state 4"
            # abstract state 2
            elif x > 8 / 2 and y > 8 / 2:
                return "abstract state 2"
            # abstract state 1
            elif x <= 8 / 2 and y >= 8 / 2:
                return "abstract state 1"
            #abstract state 3
            elif x >= 8 / 2 >= y:
                return "abstract state 3"

        if self.KEY == True:
            # door close
            if x == 1 and y == 1:
                return "door open"
            #abstract state 4
            elif x <= 8 / 2 and y <= 8 / 2:
                return "abstract state 4 with key"
            # abstract state 2
            elif x > 8 / 2 and y > 8 / 2:
                return "abstract state 2 with key"
            # abstract state 1
            elif x <= 8 / 2 and y >= 8 / 2:
                return "abstract state 1 with key"
            #abstract state 3
            elif x >= 8 / 2 >= y:
                return "abstract state 3 with key"