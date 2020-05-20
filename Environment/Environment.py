import gym
import matplotlib.pyplot as plt
import numpy as np
import pickle


class Environment:

    def __init__(self, env, preprocessing=False, rendering_custom_class=False, display_env=False):
        self.env = env  # the environment
        self.n_step = 0
        self.n_episodes = 0
        self.total_r_episode = 0
        self.trajectory = []
        self.number_of_trajectory_saved = 0
        self.display_env = display_env

        if not rendering_custom_class:
            self.rendering = self.env
        else:
            self.rendering = rendering_custom_class()

        self.preprocessing = preprocessing

        print("action space of the environment: " + str(self.env.action_space))

    def run(self, agent):

        img = self.env.reset()  # the state of the environment
        # position = (1,8)

        if not self.preprocessing:
            s = img
        else:
            s = self.preprocessing.preprocess_image(img)  # reshape and preprocess the image

        self.total_r_episode = 0  # total reward
        while True:

            a = agent.act(s)
            img, r, done, info = self.env.step(a)
            if self.display_env: #and r > 0:  # just to check the correct episodes
                self.rendering.render(img)

            if not self.preprocessing:
                s_ = img
            else:
                s_ = self.preprocessing.preprocess_image(img)  # reshape and preprocess the image

            r = float(r)
            #Stocastich_reward = np.random.normal(1.0, 1.0)
            r_a = np.clip(r, -1, 1)

            self.n_step, self.n_episodes = agent.observe((s, a, r_a, s_, done, info))   # qui stai passando la reward clipped!!!
            agent.replay(done)

            self.total_r_episode += r
            #self.n_step += 1

            s = s_

            if done:
                if self.preprocessing:
                    self.preprocessing.reset(done)
                break

        #self.n_episodes += 1

        #agent.nn.save_weights()

        return self.n_episodes, self.n_step, self.total_r_episode

    def close(self):
        self.env.reset()
        self.n_episodes = 0
        self.n_step = 0
        self.total_r_episode = 0

    def save_trajectory(self, sample, total_r_episode, done):

        self.trajectory.append(sample)

        if(done):
            with open('experts_trajectories.pkl', 'ab') as f:
                pickle.dump(self.trajectory, f)
                self.trajectory.clear()
                self.number_of_trajectory_saved += 1

        if done:
            self.trajectory.clear()

        if(self.number_of_trajectory_saved>=1000):
            raise NameError('Finished to Collect data')

