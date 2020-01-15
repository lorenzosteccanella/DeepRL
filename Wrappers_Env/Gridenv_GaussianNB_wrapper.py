import gym
from Utils import normalize, get_pixels_from_obs, np_in_list, ssim_in_list, SSIM_equal, sample_colors, make_gray_scale, hash_numpy_array_equal, make_downsampled_image
from collections import deque
import numpy as np
from sklearn.naive_bayes import GaussianNB
import pickle

class Gridenv_GaussianNB_wrapper(gym.Wrapper):


    def __init__(self, env, parameters):

        super().__init__(env)
        self.parameters = parameters
        self.width = self.parameters["width"]
        self.height = self.parameters["height"]
        self.images_stack = deque([], maxlen=self.parameters["stack_images_length"])
        self.KEY = False
        self.model = self.load_model('/home/lorenzo/Documenti/UPF/DeepRL/Wrappers_Env/encoder.pkl')

    def load_model(self, path):
        with open(path, 'rb') as fid:
            return pickle.load(fid)

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

    def encode(self,s):
        return self.model.predict(s)

    def get_position_abstract_state_gridenv_GE_MazeKeyDoor_v0(self, position, reward):

        #initial state returned when the environments is resetted
        if position is None:
            x = 1
            y = self.height-2
            r = 0
        else:
            x = position[0]
            y = position[1]
            r = reward

        state = np.asarray([x,y], dtype=np.float32).reshape(1, -1)
        return self.encode(state)



