import gym
from Utils import normalize, get_pixels_from_obs, np_in_list, ssim_in_list, SSIM_equal, sample_colors, make_gray_scale, hash_numpy_array_equal, make_downsampled_image
import numpy as np
from collections import deque

class WrapperObsHRL(gym.ObservationWrapper):
    """
    An abstract class for a pixel based observation wrapper.
    By default, the observations of option and manager are :
    - downsampled
    - gray-scaled
    In the subclasses, we can also sample the colors
    """

    def __init__(self, env, parameters):

        super().__init__(env)
        self.parameters = parameters
        self.images_stack = deque([], maxlen=self.parameters["stack_images_length"])
        #self.old_abstract_state = 0
        #self.abstract_states = []


    def observation(self, observation):
        image = observation

        # manager observation
        img_manager = self.get_manager_obs(image.copy())

        # option observation
        img_option = self.get_option_obs(image.copy())

        # render it
        return {"vanilla": image, "manager": img_manager, "option": img_option}

    def get_manager_obs(self, image):

        if self.parameters["GRAY_SCALE"]:
            image = make_gray_scale(image)

        #img_manager = ObsPixelWrapper.sample_colors(img_manager, self.parameters["THRESH_BINARY_MANAGER"])

        img_manager = make_downsampled_image(image, self.parameters["ZONE_SIZE_MANAGER_X"],
                                                             self.parameters["ZONE_SIZE_MANAGER_Y"])

        return img_manager

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


