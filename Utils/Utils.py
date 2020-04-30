import scipy.misc
import random
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
from collections import deque
import cv2
import gym
from gym import spaces
import math
from Environment import Environment


class LoadEnvironment:

    def Load(self, environment, preprocessing, display_env = True):
        if display_env:
            from Utils import ShowRenderHRL
            rendering = ShowRenderHRL
        else:
            rendering = False

        env = Environment(environment, preprocessing=preprocessing, rendering_custom_class=rendering)

        return env



class Preprocessing:

    def __init__(self, image_width, image_height, image_depth, stacked_frames = 1, normalization = True):
        self.image_width = image_width
        self.image_height = image_height
        self.image_depth = image_depth
        self.stacked_frames = stacked_frames
        self.images_stack = deque([], maxlen=self.stacked_frames)
        self.normalization = normalization

    def preprocess_image(self, img):
        #img = Preprocessing.image_resize(img, self.image_width, self.image_height)
        #img = Preprocessing.gray_scale(img)
        if self.normalization:
            img = img / 255
        img_stacked = self.stack_frames(img)

        return img_stacked

    def stack_frames(self, img):
        self.images_stack.append(img)

        img_stacked = np.zeros((img.shape[0], img.shape[1],
                                img.shape[2]*self.stacked_frames), dtype=np.float32)
        index_image = 0
        for i in range(0, self.images_stack.__len__()):
            img_stacked[..., index_image:index_image+self.image_depth] = self.images_stack[i]
            index_image=index_image+self.image_depth

        return img_stacked

    def reset(self, done):
        if done:
            self.images_stack.clear()

    @staticmethod
    def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)

        # return the resized image
        return resized

    @staticmethod
    def gray_scale(image):
        im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if True:
            plt.imshow(im)
            plt.show()
        im = im.reshape((*im.shape, 1))  # reshape for the neural network for a2c
        return im

class ExperienceReplay:

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=self.max_size)
        self.max_reward = float('-inf')

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size, random=True):
        buffer_size = len(self.buffer)
        if random:
            index = np.random.choice(np.arange(buffer_size),
                                     size=batch_size,
                                     replace=False)
        else:
            index = np.arange(buffer_size)

        sample = [(i, self.buffer[i]) for i in index]

        imp_w = np.ones((batch_size, 1), dtype=np.float32)

        return sample, imp_w

    def buffer_len(self):
        return len(self.buffer)

    def reset_buffer(self):
        self.buffer.clear()

    def reset_size(self, max_size):
        if self.max_size != max_size:
            self.max_size = max_size
            self.buffer = deque(maxlen=self.max_size)

    def update(self, idx, error):
        None

class AnaliseResults:

    @staticmethod
    def reward_over_episodes(x, y):

        plt.plot(x, y)
        plt.show()

    @staticmethod
    def save_data(path_folder, file_name, data):
        directory = os.path.dirname(path_folder)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(path_folder+file_name, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load_data(path_folder, file_name):

        with open(path_folder+file_name, 'rb') as f:
            mynewlist = pickle.load(f)

        return mynewlist


class UpdateWeightsModels:

    def __init__(self, weights, model):
        self.operation = []
        self.model = model
        self.weights = weights

    def _set_update(self):
        for ln1, ln2 in (zip(self.weights, self.model)):
            self.operation.append(ln2.assign(ln1))

    def set_operations(self):
        self._set_update()
        print("OPERATION TO UPDATE WEIGHTS SET UP")

    def update(self, sess):
        for op in self.operation:
            sess.run(op)

    def get_weights(self):
        return self.weights

    def get_model(self):
        return self.model

    def set_weights(self, weights):
        self.weights = weights

    def set_model(self, model):
        self.model = model


class SoftUpdateWeightsEager:

    #θ_target = τ * θ_local + (1 - τ) * θ_target

    def __init__(self, weights, model, tau=1e-3):
        self.operation = []
        self.model = model
        self.weights = weights
        self.tau = tau

    def update(self):
        weights_main_network = np.array(self.weights.model.get_weights())
        weights_target_network = np.array(self.model.model.get_weights())

        self.model.model.set_weights(self.tau * weights_main_network +
                                     (1 - self.tau) * weights_target_network)

    def exact_copy(self):
        weights_main_network = np.array(self.weights.model.get_weights())

        self.model.model.set_weights(weights_main_network)

class SoftUpdateWeightsPPO:

    #θ_target = τ * θ_local + (1 - τ) * θ_target

    def __init__(self, weights, model, tau=1e-3):
        self.operation = []
        self.model = model
        self.weights = weights
        self.tau = tau

    def update(self):
        weights_main_network = np.array(self.weights.get_weights())
        weights_target_network = np.array(self.model.get_weights())

        self.model.set_weights(self.tau * weights_main_network +
                                     (1 - self.tau) * weights_target_network)

    def exact_copy(self):
        weights_main_network = np.array(self.weights.get_weights())

        self.model.set_weights(weights_main_network)


class UpdateWeightsEager:

    steps = 0

    def __init__(self, weights, model):
        self.operation = []
        self.model = model
        self.weights = weights

    def update(self):
        self.model.model.set_weights(self.weights.model.get_weights())


class AnalyzeMemory:

    memory_distribution = []
    reward_distribution = []

    def add_batch(self, batch):
        for i in range(len(batch)):
            episode_n = batch[i][1][-1]
            r = batch[i][1][2]
            self.memory_distribution.append(episode_n)
            self.reward_distribution.append(r)

    def plot_memory_distribution(self):
        print(len(self.memory_distribution))
        plt.hist(self.memory_distribution)
        plt.show()
        plt.hist(self.reward_distribution)
        plt.show()

class ToolEpsilonDecayExploration:
    @staticmethod
    def epsilon_decay_end_steps(MIN_EPSILON, LAMBDA):

        exp=0

        while True:
            exp +=1
            epsilon = MIN_EPSILON + (1 - MIN_EPSILON) * math.exp(-LAMBDA * exp)
            if epsilon == MIN_EPSILON:
                break

        print("MAXIMUM STEPS OF EXPLORATION:", exp)
