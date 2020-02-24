import gym
from Utils import normalize, get_pixels_from_obs, np_in_list, ssim_in_list, SSIM_equal, sample_colors, make_gray_scale, hash_numpy_array_equal, make_downsampled_image
from collections import deque
import numpy as np

class PositionGridenv_GE_MazeKeyDoor_v0(gym.Wrapper):


    def __init__(self, env, parameters):

        super().__init__(env)
        self.parameters = parameters
        self.width = self.parameters["width"]
        self.height = self.parameters["height"]
        self.n_zones = self.parameters["n_zones"]
        self.images_stack = deque([], maxlen=self.parameters["stack_images_length"])
        self.KEY = False

    def reset(self, **kwargs):
        self.images_stack.clear()

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
        img_option = self.get_obs(image.copy())

        # render it
        return {"vanilla": image, "manager": None, "option": img_option}

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

    def get_position_abstract_state_gridenv_GE_MazeKeyDoor_v0(self, position, reward):
        step_x = int((self.width - 2) / self.n_zones)
        step_y = int((self.height - 2) / self.n_zones)

        #initial state returned when the environments is resetted
        if position is None:
            self.KEY = False
            return "abstract state %i %i" % (step_x, self.height-2)

        x = position[0]
        y = position[1]
        r = reward


        if self.KEY == False:
            # if x == 0 or y == 0:
            #     return "died"
            # if x == self.width - 1 or y == self.height - 1:
            #     return "died"


            # door close
            if x == 1 and y == 1:
                return "door close"

            # key taken
            if r == 1:
                self.KEY = True
                return "key taken"

            for grid_x in range(step_x, (self.width - 1), step_x):
                for grid_y in range(step_y, (self.height - 1), step_y):

                    # dying
                    if x == 0:
                        if y<=grid_y:
                            return "died %i %i" % (x + grid_x, grid_y)
                    elif x == self.width - 1:
                        if y<=grid_y:
                            return "died %i %i" % (x -1, grid_y)
                    # dying
                    if y == 0:
                        if x <= grid_x:
                            return "died %i %i" % (grid_x, y + grid_y)
                    elif y == self.height - 1:
                        if x <= grid_x:
                            return "died %i %i" % (grid_x, y -1)

                    if x <= grid_x and y <= grid_y:
                        return "abstract state %i %i" % (grid_x, grid_y)

        if self.KEY == True:
            # if x == 0 or y == 0:
            #     return "died with key"
            # if x == self.width - 1 or y == self.height - 1:
            #     return "died with key"

            # door open
            if r == 1:
                return "door open"

            for grid_x in range(step_x, (self.width - 1), step_x):
                for grid_y in range(step_y, (self.height - 1), step_y):

                    # dying
                    if x == 0:
                        if y<=grid_y:
                            return "died with key %i %i" % (x + grid_x, grid_y)
                    elif x == self.width - 1:
                        if y<=grid_y:
                            return "died with key %i %i" % (x -1, grid_y)
                    # dying
                    if y == 0:
                        if x <= grid_x:
                            return "died with key %i %i" % (grid_x, y + grid_y)
                    elif y == self.height - 1:
                        if x <= grid_x:
                            return "died with key %i %i" % (grid_x, y -1)

                    if x <= grid_x and y <= grid_y:
                        return "abstract state with key %i %i" % (grid_x, grid_y)
