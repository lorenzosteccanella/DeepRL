import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os import listdir
from os.path import isfile, join
import numpy as np

def open_file_ms(path):
	list_results_x = []
	list_results_y = []
	onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
	for files in onlyfiles:
		x, y = list(), list()
		with open(path+files) as f:
			for line in f:
				x.append(float(line.split()[0]))
				y.append(float(line.split()[2]))
		list_results_x.append(x)
		list_results_y.append(y)
	
	return list_results_x, list_results_y

def open_file_2_ms(path):
	list_results_x = []
	list_results_y = []
	onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
	for files in onlyfiles:
		x, y = list(), list()
		with open(path+files) as f:
			for line in f:
				x.append(float(line.split()[1]))
				y.append(float(line.split()[2]))
		list_results_x.append(x)
		list_results_y.append(y)
	
	return list_results_x, list_results_y

def open_file_3_ms(path):
	list_results_x = []
	list_results_y = []
	onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
	for files in onlyfiles:
		x, y = list(), list()
		with open(path+files) as f:
			for line in f:
				x.append(float(line.split()[1]))
				y.append(float(line.split()[3]))
		list_results_x.append(x)
		list_results_y.append(y)
	
	return list_results_x, list_results_y

def adjust_data(datax, datay):
	x = []
	for i in range(len(datax)):
		x.append(list(range(1, 400001)))
	
	y = []
	tmp_y = []
	for i, result_x, result_y in zip(range(len(datax)), datax, datay):
		y.append([])	
		for j in range(len(result_x) -1):
			steps = int(result_x[j+1] - result_x[j])
			for k in range(steps):
				y[i].append(result_y[j])
		if len(y[i]) != 400000:
			val = y[i][len(y[i])-1]
			for m in range(400000 - len(y[i])):
				y[i].append(val)
			
	return x[0], y

img_env_1 = mpimg.imread('/home/lorenzo/Documenti/UPF/DeepRL/Plots/images/env16x16Treasure.png')

List_of_path = ["/home/lorenzo/Documenti/UPF/DeepRL/Plots/data/Experiment4/HRL/", "/home/lorenzo/Documenti/UPF/DeepRL/Plots/data/Experiment4/HRL-CO/"]

List_of_color= [('#CC4F1B', '#FF9848') , ('#4B6034', '#7FE90B') , ('#0A305D', '#0471F1'), ('#581C58', '#EB13EB'), ('#0A305D', '#0471F1'), ('#581C58', '#EB13EB') ]
List_of_title= ["HRL-SIL", "HRL-SIL-CO"]


#fig, ax = plt.subplots(2, 4, figsize=(15,15))


for i, path in zip(range(len(List_of_path)), List_of_path):
	
	list_results_x, list_results_y = open_file_3_ms(path)
	x, list_results_y = adjust_data(list_results_x, list_results_y)

	#min_length = min([len(result) for result in list_results_x])
	#x = list_results_x[0][:min_length]

	#list_results_y = [a[:min_length] for a in list_results_y]
	st = np.vstack(list_results_y)

	y_mean = np.mean(st, axis=0)
	y_max = np.max(st, axis=0)
	y_min = np.min(st, axis=0)
	
	#ax[0,i].set_title(List_of_title[i])
	#ax[0,i].imshow(img_env_1)
	#ax[1,i].set_title("rewards_over_nsteps")
	#ax[1,i].plot(x, y_mean, color=List_of_color[i][0], label='SIL')
	#ax[1,i].fill_between(x, y_min, y_max, alpha=0.5, edgecolor=List_of_color[i][0], facecolor=List_of_color[i][1])
	
	plt.plot(x, y_mean, color=List_of_color[i][0], label= List_of_title[i])
	plt.fill_between(x, y_min, y_max, alpha=0.5, edgecolor=List_of_color[i][0], facecolor=List_of_color[i][1])
	plt.legend(loc="best", prop={'size': 12})
	plt.xlabel("number of steps")
	plt.ylabel("total reward")

path_2_save= "/home/lorenzo/Documenti/UPF/DeepRL/Plots/images/"
file_name= "Montezuma"
plt.savefig(path_2_save + file_name, bbox_inches='tight')
plt.show()




