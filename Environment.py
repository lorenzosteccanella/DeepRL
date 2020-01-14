import gym
import matplotlib.pyplot as plt
import numpy as np
import pickle


class Environment:

    def __init__(self, env, preprocessing=False, rendering_custom_class=False):
        self.env = env  # the environment
        self.n_step = 0
        self.n_episodes = 0
        self.total_r_episode = 0
        self.trajectory = []
        self.number_of_trajectory_saved = 0

        if not rendering_custom_class:
            self.rendering = self.env
        else:
            self.rendering = rendering_custom_class()

        self.preprocessing = preprocessing

        print("action space of the environment: " + str(self.env.action_space))

    def run(self, agent):

        img = self.env.reset()  # the state of the environment
#        position = (1,8)

        if not self.preprocessing:
            s = img
        else:
            s = self.preprocessing.preprocess_image(img)  # reshape and preprocess the image

        self.total_r_episode = 0  # total reward
        while True:
            a = agent.act(s)
            img, r, done, info = self.env.step(a)
            if False: #and r > 0:  # just to check the correct episodes
                self.rendering.render(img)

            if not self.preprocessing:
                s_ = img
            else:
                s_ = self.preprocessing.preprocess_image(img)  # reshape and preprocess the image

#            position_ = info["position"]
            r = float(r)
            #Stocastich_reward = np.random.normal(1.0, 1.0)
            r = np.clip(r, -1, 1)

            #if done:  # terminal state
            #    s_ = None

            agent.observe((s, a, r, s_, done, info))
            agent.replay()

            self.total_r_episode += r
            self.n_step += 1

#            h_s = agent.get_observation_encoding(s)
#            h_s_ = agent.get_observation_encoding(s_)

#            self.save_trajectory((s, h_s, a, r, s_, h_s_, done), self.total_r_episode, done)

            s = s_
#            position = position_

            if done:
                if self.preprocessing:
                    self.preprocessing.reset(done)
                break

        self.n_episodes += 1

        return self.n_episodes, self.n_step, self.total_r_episode

    def close(self):
        self.env.reset()
        self.n_episodes = 0
        self.n_step = 0
        self.total_r_episode = 0

#    def save_trajectory(self, sample, total_r_episode, done):
#
#        self.trajectory.append(sample)
#
#        if(total_r_episode >= 2 and done and self.number_of_trajectory_saved<30):
#            with open('experts_trajectories.pkl', 'wb') as f:
#                pickle.dump(self.trajectory, f)
#                self.trajectory.clear()
#                self.number_of_trajectory_saved += 1
#
#        if done:
#            self.trajectory.clear()
