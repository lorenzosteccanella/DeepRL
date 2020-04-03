import random
from typing import Dict, Any

from Agents.AbstractAgent import AbstractAgent
import numpy as np
import csv
import pickle


class ExplorationEffortAgent_tabular(AbstractAgent):
    exp = 0

    def __init__(self, action_space, buffer):
        self.action_space = action_space
        self.buffer = buffer
        self.effort_exploration_vector = []
        self.max_n_step_partition = 1000
        self.n_step_partition = random.choice(range(1, self.max_n_step_partition))
        self.state_1 = None
        self.Q = {}
        self.EE = {}
        self.gamma = 0.95

    def act(self, s):
        return random.choice(self.action_space)

    def auxiliary_return(self, r, gamma=0.8):
        """Takes 1d float array of rewards and computes discounted reward
        e.g. f([1, 1, 1], 0.99) -> [2.9701, 1.99, 1]
        """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, len(r))):
            running_add = running_add + pow(gamma, t) * r[t]
            discounted_r[t] = running_add
        return discounted_r[0]

    def tabularQ(self, sample):

        learning_rate = 0.9
        gamma = 0.99

        s = sample[0]
        a = sample[1]
        r = sample[2]
        s_ = sample[3]
        done = sample[4]

        if s not in self.Q:
            q_s = {s: np.zeros((len(self.action_space)))}  # updated the first elements
            self.Q.update(q_s)
        elif s_ not in self.Q:
            q_s_ = {s_: np.zeros((len(self.action_space)))}  # updated the first elements
            self.Q.update(q_s_)

        else:
            self.Q[s][a] = (
                        (1 - learning_rate) * self.Q[s][a] + learning_rate * (r + (1 - done) * gamma * max(self.Q[s_])))

    def print_value_from_Q(self):

        width = 10
        height = 10
        depth = 2

        print("")
        print("Values")
        print("")
        for k in range(depth):
            print("")
            print("-" * 30 + "  " + str(k) + "  " + 30 * "-")
            print("")
            for i in range(width):
                for j in range(height):
                    if (j, i, k) in self.Q:
                        print(str(round(np.average(self.Q[(j, i, k)]), 4)).ljust(5), end=" , ")
                    else:
                        print(str(None).ljust(5), end=" , ")

                print("")

    def observe_Q(self, sample):

        self.exp += 1

        effort_exploration_vector_t = -1 * (
                    np.ones(len(self.action_space)) / len(self.action_space))  # uniform random policy

        if not sample[4]:
            if self.exp % self.n_step_partition != 0 or self.exp == 0:

                done = False
            else:
                done = True
        else:
            done = True

        sampleQ = [sample[0], sample[1], effort_exploration_vector_t[sample[1]], sample[3], done]

        self.tabularQ(sampleQ)

        if sample[4]:
            self.exp = 0

            self.print_value_from_Q()

            w = csv.writer(open("ExplorationEffort.csv", "w"))
            for key, val in self.Q.items():
                w.writerow([key, val])

    def observe_EE(self, sample):


        self.exp += 1

        # let's start by calculating the Exploration Effort for now the policy is just uniform random
        if not sample[4]:
            if self.exp % self.n_step_partition != 0:

                if self.state_1 is None:
                    self.state_1 = sample[0]

                effort_exploration_vector_t = -1 * (
                        np.ones(len(self.action_space)) / len(self.action_space))  # uniform random policy
                effort_exploration_vector_t[sample[1]] += 1  # I define the effort exploration for this specific value
                self.effort_exploration_vector.append(effort_exploration_vector_t)

                # let's construct the sample to save in the buffer

                s1 = self.state_1
                s2 = sample[3]

                sample_t = [s1, self.effort_exploration_vector.copy(), s2]

                self.buffer.add(sample_t)

            elif self.exp % self.n_step_partition == 0:

                if self.state_1 is None:
                    self.state_1 = sample[0]

                effort_exploration_vector_t = -1 * (
                        np.ones(len(self.action_space)) / len(self.action_space))  # uniform random policy
                effort_exploration_vector_t[sample[1]] += 1  # I define the effort exploration for this specific value
                self.effort_exploration_vector.append(effort_exploration_vector_t)

                # let's construct the sample to save in the buffer

                s1 = self.state_1
                s2 = sample[3]

                sample_t = [s1, self.effort_exploration_vector.copy(), s2]

                self.buffer.add(sample_t)

                self.state_1 = sample[3]
                self.effort_exploration_vector.clear()
                self.n_step_partition = random.choice(range(1, self.max_n_step_partition))

        else:

            if self.state_1 is None: # this could happen when the first step directly is a terminal state
                self.state_1 = sample[0]

            effort_exploration_vector_t = -1 * (
                    np.ones(len(self.action_space)) / len(self.action_space))  # uniform random policy
            effort_exploration_vector_t[sample[1]] += 1  # I define the effort exploration for this specific value
            self.effort_exploration_vector.append(effort_exploration_vector_t)
            # let's construct the sample to save in the buffer

            s1 = self.state_1
            s2 = sample[3]

            sample_t = [s1, self.effort_exploration_vector.copy(), s2]
            self.buffer.add(sample_t)

            self.effort_exploration_vector.clear()
            self.exp = 0
            self.state_1 = None

            f = open("ExplorationEffort.pkl", "wb")
            pickle.dump(self.EE, f)
            f.close()

            w = csv.writer(open("ExplorationEffort.csv", "w"))
            for key, val in self.EE.items():
                w.writerow([key, val])


    def observe(self, sample):  # (s, a, r, s_, done, info)

        self.observe_EE(sample)

    def replay(self):
        if self.buffer.buffer_len() >= 32:
            batch, _ = self.buffer.sample(32)

            for index, sample in batch:

                s1s2 = (sample[0], sample[2])
                if s1s2 not in self.EE:
                    mc_return = self.auxiliary_return(sample[1], self.gamma)

                    #one_step_return = sample[1][0] + self.gamma * self.auxiliary_return(sample[1][1::], self.gamma)
                    #R = (1 - 0.6) * one_step_return + 0.6 * mc_return
                    R = mc_return
                    ee_s = {s1s2: [1, R]}
                    self.EE.update(ee_s)

                else:

                    mc_return = self.auxiliary_return(sample[1], self.gamma)

                    #one_step_return = sample[1][0] + self.gamma * self.auxiliary_return(sample[1][1::], self.gamma)

                    #R = (1 - 0.6) * one_step_return + 0.6 * mc_return
                    R = mc_return
                    self.EE[s1s2][1] = ((self.EE[s1s2][1] * self.EE[s1s2][0]) + R)/(self.EE[s1s2][0] + 1) #0.9 * self.EE[s1s2] + (1 - 0.9) * sample[1]

                    self.EE[s1s2][0] += 1 # the number of visits


                #if s1s2 == ((1,6,0), (2,6,0)):
                #    print(s1s2, self.EE[s1s2])