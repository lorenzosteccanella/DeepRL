import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def open_file(file_name):
	x, y = list(), list()
	with open(file_name) as f:
		for line in f:
			x.append(float(line.split()[0]))
			y.append(float(line.split()[2]))
	
	return x,y

def open_file_2(file_name):
	x, y = list(), list()
	with open(file_name) as f:
		for line in f:
			x.append(float(line.split()[1]))
			y.append(float(line.split()[2]))
	
	return x,y

img1 = mpimg.imread('/home/lorenzo/Documenti/UPF/DeepRL/Plots/images/env0.png')
img2 = mpimg.imread('/home/lorenzo/Documenti/UPF/DeepRL/Plots/images/Transfer-totR_3_env0_v2.png')
img4 = mpimg.imread('/home/lorenzo/Documenti/UPF/DeepRL/Plots/images/env1.png')
img5 = mpimg.imread('/home/lorenzo/Documenti/UPF/DeepRL/Plots/images/Transfer-totR_3_env1.png')
img7 = mpimg.imread('/home/lorenzo/Documenti/UPF/DeepRL/Plots/images/env2.png')
img8 = mpimg.imread('/home/lorenzo/Documenti/UPF/DeepRL/Plots/images/Transfer-totR_3_env2.png')


fig, ax = plt.subplots(2, 3, figsize=(15,10))
ax[0,0].set_title("env0")
ax[0,0].imshow(img1)
ax[1,0].set_title("Transfer Learning Performance")
ax[1,0].imshow(img2)

ax[0,1].set_title("env1")
ax[0,1].imshow(img4)
ax[1,1].set_title("Transfer Learning Performance")
ax[1,1].imshow(img5)

ax[0,2].set_title("env2")
ax[0,2].imshow(img7)
ax[1,2].set_title("Transfer Learning Performance")
ax[1,2].imshow(img8)

ax[0,0].set_aspect('auto')
ax[1,0].set_aspect('auto')
ax[0,1].set_aspect('auto')
ax[1,1].set_aspect('auto')
ax[0,2].set_aspect('auto')
ax[1,2].set_aspect('auto')

ax[0,0].set_axis_off()
ax[1,0].set_axis_off()
ax[0,1].set_axis_off()
ax[1,1].set_axis_off()
ax[0,2].set_axis_off()
ax[1,2].set_axis_off()


path_2_save= "/home/lorenzo/Documenti/UPF/DeepRL/Plots/images/"
file_name= "TRANSFER-totr3"
plt.savefig(path_2_save + file_name, bbox_inches='tight')
plt.show()
