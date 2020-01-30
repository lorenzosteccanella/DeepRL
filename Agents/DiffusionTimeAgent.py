import random
from Agents.AbstractAgent import AbstractAgent
import pickle
import csv


class DiffusionTimeAgent(AbstractAgent):

    exp = 0

    def __init__(self, action_space, env):
        self.action_space = action_space
        self.env = env
        self.number_of_steps = 0
        self.diffusion_dict = {}
        self.resetted = False
        self.key_reset_state = []
        self.total_reward_episode = 0.0
        self.key_point_steps = 10

    def act(self, s):

        key1 = s

        if key1 not in self.diffusion_dict: #  just the first state
            self.diffusion_dict.update({key1: [self.number_of_steps, self.total_reward_episode]})

        self.number_of_steps += 1
        return random.choice(self.action_space)

    def observe(self, sample):  # in (s, a, r, s_, done, info) format

        #print(sample[0], sample[3], self.total_reward_episode)

        # to reset to saved states with 0,25 probability
        if self.resetted == False and len(self.key_reset_state) > 10 and sample[4] is False:
            self.resetted = True
            if random.choice([True, False, False, False]):
                #print("*" * 50 + "resetting" + "*" * 50)

                state_to_reset = random.choice(self.key_reset_state)

                self.env.reset_env_to_state(state_to_reset[0], state_to_reset[1], state_to_reset[2], state_to_reset[3])

        self.total_reward_episode += sample[2]

        # just to save a state for resetting it
        if self.number_of_steps % self.key_point_steps == 0 and sample[4] is False:

            exist_in_list = any(sample[3] == sublist[2] for sublist in self.key_reset_state)

            if exist_in_list:
                for reset_states in self.key_reset_state:
                    if reset_states[2] == sample[3]:
                        if reset_states[1] > self.number_of_steps:
                            reset_states[0] = self.env.env.env.clone_state()
                            reset_states[1] = self.number_of_steps
                            reset_states[2] = sample[3]
                            reset_states[3] = self.total_reward_episode

                    if reset_states[1] % self.key_point_steps != 0:
                        self.key_reset_state.remove(reset_states)
            else:
                self.key_reset_state.append([self.env.env.env.clone_state(), self.number_of_steps, sample[3], self.total_reward_episode])

        key2 = (sample[3])

        if key2 not in self.diffusion_dict: #  the following states
            self.diffusion_dict.update({key2: [self.number_of_steps, self.total_reward_episode]})
        else:
            if self.diffusion_dict[key2][0] > self.number_of_steps:
                self.diffusion_dict[key2][0] = self.number_of_steps

        if sample[4]: # end of the episode resetting number of steps
            self.number_of_steps = 0
            self.total_reward_episode = 0.0
            self.resetted = False

            #print(self.diffusion_dict)
            f = open("step_distances_position_observation.pkl", "wb")
            pickle.dump(self.diffusion_dict, f)
            f.close()
            w = csv.writer(open("step_distances_position_observation.csv", "w"))
            for key, val in self.diffusion_dict.items():
                w.writerow([key, val])

    def replay(self):
        pass
