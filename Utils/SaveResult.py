import os
import time
import matplotlib.pyplot as plt
import pickle
import numpy as np


class SaveResult:

    def __init__(self, dir_name):
        self.dir_name = dir_name
        self.dir_path = self.make_dir_path(self.dir_name)
        self.dir_path_seed = None
        self.all_seed_paths = []

    def set_seed(self, seed):
        self.dir_path_seed = self.dir_path + "/seed_" + str(seed)
        self.all_seed_paths.append(self.dir_path_seed)
        os.mkdir(self.dir_path_seed)

    @staticmethod
    def make_dir_path(dir_name):
        dir_name = "results/" + dir_name
        if not os.path.exists("results/"):
            os.mkdir("results/")
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        dir_name += time.asctime(time.localtime(time.time())).replace(" ", "_")
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        return dir_name

    def save_overwrite_data(self, file_name, data):
        with open(self.dir_path_seed + "/" + file_name, 'w') as f:
            for message in data:
                f.write(message)
        f.close

    def save_data(self, file_name, data):
        with open(self.dir_path_seed + "/" + file_name, 'a') as f:
            for message in data:
                f.write(message)
        f.close

    def get_path(self):
        return self.dir_path_seed

    def save_pickle_data(self, file_name, data):
        with open(self.dir_path_seed + "/" + file_name, 'wb') as f:
            pickle.dump(data, f)


    def save_settings(self, parameters):
        """
        writes the parameters in a file
        """
        f = open(self.dir_path + "/" + "setting", "w")
        for key in parameters:
            f.write(key + " : " + str(parameters[key]) + "\n")

        f.close()

    def plot_results(self, file_name, title, xlabel, ylabel):
        x, y = [], []
        try:
            with open(str(self.dir_path_seed) + "/" + file_name) as f:
                print(str(self.dir_path_seed) + "/" + file_name)
                for line in f:
                    x.append(float(line.split()[0]))
                    y.append(float(line.split()[2]))

                plt.plot(x, y)
                plt.title(title)
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                # plt.draw()
                # plt.pause(0.01)
                plt.savefig(str(self.dir_path_seed) + "/" + file_name)
                plt.close()

        except IOError:
            print("Error: File does not appear to exist.")
            return 0

    def plot_multiple_seeds(self, file_name, title, xlabel, ylabel):
        list_results_x = []
        list_results_y = []

        for p in self.all_seed_paths:
            x, y = list(), list()
            with open(p + "/" + file_name) as f:
                for line in f:
                    x.append(float(line.split()[0]))
                    y.append(float(line.split()[2]))

            list_results_x.append(x)
            list_results_y.append(y)

        min_length = min([len(result) for result in list_results_x])

        x = list_results_x[0][:min_length]

        list_results_y = [a[:min_length] for a in list_results_y]
        st = np.vstack(list_results_y)

        y_mean = np.mean(st, axis=0)
        y_max = np.max(st, axis=0)
        y_min = np.min(st, axis=0)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.plot(x, y_mean, color='#CC4F1B')
        plt.fill_between(x, y_min, y_max, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
        plt.savefig(str(self.dir_path) + "/" + file_name)

        plt.close()

    def plot_success_rate_transitions(self, file_name):
        self.plot_results(file_name, "success rate of options' transitions",
                          "number of options executed", "% of successful option executions")

    def plot_manager_score(self):
        self.plot_results(self.manager_score_file_name, "manager's average score",
                          "epochs", "average total reward in epochs")


