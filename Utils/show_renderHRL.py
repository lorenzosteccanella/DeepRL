import cv2
import pyglet
import numpy as np
import time

class ShowRenderHRL(object):

    def __init__(self):
        from gym.envs.classic_control import rendering  # causes problem with the cluster

        self.display_learning = True

        self.vanilla_view = True
        self.option_view = False
        self.agent_view = False
        self.combined_view = False
        self.slow_display = 0

        self.viewer = rendering.SimpleImageViewer()
        self.viewer.width = 512
        self.viewer.height = 512
        self.viewer.window = pyglet.window.Window(width=self.viewer.width, height=self.viewer.height,
                                                  display=self.viewer.display, vsync=False, resizable=True)

        self.viewer.window.on_key_press = self.key_press
        self.Print=True

    def render(self, observation):
        """
        :param observation: a dictionary containing the observations:
        - observation vanilla
        - observation manager
        - observation option
        :return:
        """
        if self.display_learning:
            if type(observation).__name__ == "ndarray":
                self.display(observation)

            else:
                assert list(observation.keys()) == ["vanilla", "manager", "option"], \
                    "observation must be a dictionary with 3 keys : vanilla, manager and option"

                if self.vanilla_view:
                    self.display(observation["vanilla"])

                elif self.agent_view:
                    self.display(observation["manager"])

                elif self.option_view:
                    self.display(observation["option"])

                elif self.combined_view:
                    self.display_combined_view(observation["vanilla"], observation["manager"])

        else:
            self.viewer.imshow(np.array([[[0]]]))

    def display_combined_view(self, obs_vanilla, obs_manager):
        img_vanilla = cv2.resize(obs_vanilla, (256, 256), interpolation=cv2.INTER_AREA)
        img_manager = cv2.resize(obs_manager, (256, 256), interpolation=cv2.INTER_AREA)

        len_shape_vanilla = len(img_vanilla.shape)
        len_shape_manager = len(img_manager.shape)

        if len_shape_vanilla > len_shape_manager:
            if self.Print:
                print("The dimension of vanilla and manager observation are not matching, we are concatenating manager to display it")
                self.Print=False
            img_manager=np.stack((img_manager,)*img_vanilla.shape[2], axis=-1)

        img = np.hstack((img_vanilla, img_manager))
        self.viewer.imshow(img)
        time.sleep(self.slow_display)

    def display(self, image_pixel):
        img = cv2.resize(image_pixel, (512, 512), interpolation=cv2.INTER_AREA)
        len_shape= len(img.shape)
        if len_shape < 3:
            img = np.stack((img,) * 3, axis=-1)

        self.viewer.imshow(img)
        time.sleep(self.slow_display)

    def close(self):
        self.viewer.close()

    def key_press(self, key, mod):
        if key == ord("d"):
            print("press d to display the observation")
            self.display_learning = not self.display_learning

        if key == ord("o"):
            self.set_option_view()

        if key == ord("v"):
            self.set_vanilla_view()

        if key == ord("a"):
            self.set_agent_view()

        if key == ord("m"):
            self.set_combined_view()

        if key == ord("s"):
            self.set_slow_display()

        if key == ord("f"):
            self.set_fast_display()

        if key == ord(" "):
            if self.option_view:
                self.set_combined_view()

            elif self.vanilla_view:
                self.set_agent_view()

            elif self.agent_view:
                self.set_option_view()

            elif self.combined_view:
                self.set_vanilla_view()

    def set_vanilla_view(self):
        print("original view")
        self.vanilla_view = True
        self.option_view = False
        self.agent_view = False
        self.combined_view = False

    def set_option_view(self):
        print("option's view")
        self.vanilla_view = False
        self.option_view = True
        self.agent_view = False
        self.combined_view = False

    def set_agent_view(self):
        print("manager's view")
        self.vanilla_view = False
        self.option_view = False
        self.agent_view = True
        self.combined_view = False

    def set_combined_view(self):
        print("combined view")
        self.vanilla_view = False
        self.option_view = False
        self.agent_view = False
        self.combined_view = True

    def set_slow_display(self):
        print("slowing display frame rate")
        if self.slow_display<=10:
            self.slow_display += 0.1

    def set_fast_display(self):
        print("speeding up display frame rate")
        if self.slow_display>0.3:
            self.slow_display -= 0.1
        else:
            self.slow_display = 0