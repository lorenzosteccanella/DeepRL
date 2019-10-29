import scipy.misc
from Utils.SumTree import SumTree
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


class Preprocessing:

    def __init__(self, image_width, image_height, image_depth, stacked_frames = 1):
        self.image_width = image_width
        self.image_height = image_height
        self.image_depth = image_depth
        self.stacked_frames = stacked_frames
        self.images_stack = deque([], maxlen=self.stacked_frames)

    def preprocess_image(self, img):
        img = Preprocessing.image_resize(img, self.image_width, self.image_height)
        #img = Preprocessing.gray_scale(img)

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
        self.buffer = deque(maxlen=max_size)
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
        self.buffer = deque(maxlen=max_size)

    def update(self, idx, error):
        None

#  Slightly modified from https://github.com/jaromiru
class PrioritizedExperienceReplay:  # stored as ( s, a, r, s_ ) in SumTree

    e = 0.01
    a = 0.6

    absolute_error_upper = 1.

    PER_b = 0.1  # importance-sampling, from initial value increasing to 1

    PER_b_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _get_priority(self, error):
        return (error + self.e) ** self.a

    def add(self, sample):  # removed error argument, to new experience we give always the max
        #p = self._get_priority(error)
        #self.tree.add(p, sample)

        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, sample)  # set the max p for new p

    def sample(self, n):

        # Create a sample array that will contains the minibatch
        batch = []

        b_ISWeights = np.empty((n, 1), dtype=np.float32)

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total() / n  # priority segment

        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1

        # Calculating the max_weight
        p_min = np.min(self.tree.tree[-self.tree.capacity:][:self.tree.occupancy]) / self.tree.total()
        max_weight = (p_min * n) ** (-self.PER_b)

        for i in range(n):
            """
            A value is uniformly sample from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            """
            Experience that correspond to each value is retrieved
            """
            index, priority, data = self.tree.get(value)

            # P(j)
            sampling_probabilities = priority / self.tree.total()

            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b) / max_weight
            batch.append((index, data))

        return batch, b_ISWeights

    def update(self, idx, error):
        clipped_errors = min(error, self.absolute_error_upper)  # clipping the error is this right?
        p = self._get_priority(clipped_errors)
        self.tree.update(idx, p)

    def buffer_len(self):
        return self.tree.occupancy


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

class Tools4DQN:

    def epsilon_decay_end_steps(self, MIN_EPSILON, LAMBDA):

        exp=0

        while True:
            exp +=1
            epsilon = MIN_EPSILON + (1 - MIN_EPSILON) * math.exp(-LAMBDA * exp)
            if epsilon == MIN_EPSILON:
                break

        print("MAXIMUM STEPS OF EXPLORATION:", exp)
