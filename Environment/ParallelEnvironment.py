import gym
import matplotlib.pyplot as plt
import numpy as np
import pickle


class ParallelEnvironment:

    def __init__(self, preprocessing=False, rendering_custom_class=False):
        self.env = None
        self.n_step = 0
        self.n_episodes = 0
        self.total_r_episode = None
        self.trajectory = []
        self.number_of_trajectory_saved = 0
        self.first_loop = True
        self.s = None

        if not rendering_custom_class:
            self.rendering = self.env
        else:
            self.rendering = rendering_custom_class()

    def run(self, agent):

        if self.first_loop:
            self.s = self.env.reset()  # the state of the environment
            self.first_loop = False

        while True:

            A = []

            for i in range(len(self.s["vanilla"])):
                o = {"vanilla": self.s["vanilla"][i], "manager": tuple(self.s["manager"][i]), "option": self.s["option"][i]}
                A.append(agent[i].act(o))
            s_, R, DONE, INFO = self.env.step(A)

            DONE = DONE.tolist()

            #Stocastich_reward = np.random.normal(1.0, 1.0)
            R_A = np.clip(R, -1, 1).tolist()

            steps = []
            episodes = []

            for i in range(len(self.s["vanilla"])):
                o = {"vanilla": self.s["vanilla"][i], "manager": tuple(self.s["manager"][i]), "option": self.s["option"][i]}
                a = A[i]
                r = R_A[i]
                done = DONE[i]
                info = INFO[i]

                if done is False:
                    o_ = {"vanilla": s_["vanilla"][i], "manager": tuple(s_["manager"][i]), "option": s_["option"][i]}
                else:
                    o_ = {"vanilla": info["terminal_observation"]["vanilla"], "manager": info["terminal_observation"]["manager"], "option": info["terminal_observation"]["option"]}

                step, ep = agent[i].observe((o, a, r, o_, done, info))

                steps.append(step)
                episodes.append(ep)

                agent[i].replay()

            self.n_step = sum(steps)
            self.n_episodes = sum(episodes)

            if self.total_r_episode is None:
                self.total_r_episode = R
            else:
                self.total_r_episode += R

            self.s = s_

            # print("*" * 30)
            # print(o["manager"], o_["manager"])
            # print("*" * 30)

            out_tot_reward = self.total_r_episode.copy()

            for i, d in zip(range(len(DONE)), DONE):
                if d is True:
                    self.total_r_episode[i] = 0

            if DONE[0] is True:
                break

        return self.n_episodes, self.n_step, out_tot_reward[0]

    def set_env(self, environment):

        self.env = environment

    def close(self):
        if self.env is not None:
            self.env.reset()
            self.n_episodes = 0
            self.n_step = 0
            self.total_r_episode = 0
            self.first_loop = True

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



