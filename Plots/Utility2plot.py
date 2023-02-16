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

img_env_1 = mpimg.imread('/home/lorenzo/Documenti/UPF/DeepRL/Plots/images/env8x8_1.png')
img_env_2 = mpimg.imread('/home/lorenzo/Documenti/UPF/DeepRL/Plots/images/env8x8_2.png')
img_env_3 = mpimg.imread('/home/lorenzo/Documenti/UPF/DeepRL/Plots/images/env8x8_3.png')

x_sil_reward_ep_env1, y_sil_reward_ep_env1 = open_file('/home/lorenzo/Documenti/UPF/DeepRL/results/TEST  -  TEST_SIL_POSITION_TRANSFER_1/Fri_May_22_01:24:18_2020/seed_0/Position_Transfer_SIL')

x_hrlsil_reward_ep_env1, y_hrlsil_reward_ep_env1 = open_file('/home/lorenzo/Documenti/UPF/DeepRL/results/TEST  -  heuristic_count_TEST_SIL_POSITION_TRANSFER_1/Fri_May_22_01:23:48_2020/seed_0/Position_Transfer_SIL_HRL_E_GREEDY')

x_sil_reward_n_steps_env1, y_sil_reward_n_steps_env1 = open_file_2('/home/lorenzo/Documenti/UPF/DeepRL/results/TEST  -  TEST_SIL_POSITION_TRANSFER_1/Fri_May_22_01:24:18_2020/seed_0/Position_Transfer_SIL')

x_hrlsil_reward_n_steps_env1, y_hrlsil_reward_n_steps_env1 = open_file_2('/home/lorenzo/Documenti/UPF/DeepRL/results/TEST  -  heuristic_count_TEST_SIL_POSITION_TRANSFER_1/Fri_May_22_01:23:48_2020/seed_0/Position_Transfer_SIL_HRL_E_GREEDY')



x_sil_reward_ep_env2, y_sil_reward_ep_env2 = open_file('/home/lorenzo/Documenti/UPF/DeepRL/results/TEST  -  TEST_SIL_POSITION_TRANSFER_1/Fri_May_22_01:24:18_2020/seed_0/Position_Transfer_SIL - GE_MazeKeyDoor8keyDoor2-v0 (copia)')

x_hrlsil_reward_ep_env2, y_hrlsil_reward_ep_env2 = open_file('/home/lorenzo/Documenti/UPF/DeepRL/results/TEST  -  heuristic_count_TEST_SIL_POSITION_TRANSFER_1/Fri_May_22_01:23:48_2020/seed_0/Position_Transfer_SIL_HRL_E_GREEDY - GE_MazeKeyDoor8keyDoor2-v0')

x_sil_reward_n_steps_env2, y_sil_reward_n_steps_env2 = open_file_2('/home/lorenzo/Documenti/UPF/DeepRL/results/TEST  -  TEST_SIL_POSITION_TRANSFER_1/Fri_May_22_01:24:18_2020/seed_0/Position_Transfer_SIL - GE_MazeKeyDoor8keyDoor2-v0 (copia)')

x_hrlsil_reward_n_steps_env2, y_hrlsil_reward_n_steps_env2 = open_file_2('/home/lorenzo/Documenti/UPF/DeepRL/results/TEST  -  heuristic_count_TEST_SIL_POSITION_TRANSFER_1/Fri_May_22_01:23:48_2020/seed_0/Position_Transfer_SIL_HRL_E_GREEDY - GE_MazeKeyDoor8keyDoor2-v0')



x_sil_reward_ep_env3, y_sil_reward_ep_env3 = open_file('/home/lorenzo/Documenti/UPF/DeepRL/results/TEST  -  TEST_SIL_POSITION_TRANSFER_1/Fri_May_22_01:24:18_2020/seed_0/Position_Transfer_SIL - GE_MazeKeyDoor8keyDoor3-v0 (copia)')

x_hrlsil_reward_ep_env3, y_hrlsil_reward_ep_env3 = open_file('/home/lorenzo/Documenti/UPF/DeepRL/results/TEST  -  heuristic_count_TEST_SIL_POSITION_TRANSFER_1/Fri_May_22_01:23:48_2020/seed_0/Position_Transfer_SIL_HRL_E_GREEDY - GE_MazeKeyDoor8keyDoor3-v0')

x_sil_reward_n_steps_env3, y_sil_reward_n_steps_env3 = open_file_2('/home/lorenzo/Documenti/UPF/DeepRL/results/TEST  -  TEST_SIL_POSITION_TRANSFER_1/Fri_May_22_01:24:18_2020/seed_0/Position_Transfer_SIL - GE_MazeKeyDoor8keyDoor3-v0 (copia)')

x_hrlsil_reward_n_steps_env3, y_hrlsil_reward_n_steps_env3 = open_file_2('/home/lorenzo/Documenti/UPF/DeepRL/results/TEST  -  heuristic_count_TEST_SIL_POSITION_TRANSFER_1/Fri_May_22_01:23:48_2020/seed_0/Position_Transfer_SIL_HRL_E_GREEDY - GE_MazeKeyDoor8keyDoor3-v0')



x_sil_reward_ep_env4, y_sil_reward_ep_env4 = open_file('/home/lorenzo/Documenti/UPF/DeepRL/results/TEST  -  TEST_SIL_POSITION_TRANSFER_1/Fri_May_22_01:24:18_2020/seed_0/Position_Transfer_SIL - GE_MazeKeyDoor8keyDoor1-v0')

x_hrlsil_reward_ep_env4, y_hrlsil_reward_ep_env4 = open_file('/home/lorenzo/Documenti/UPF/DeepRL/results/TEST  -  heuristic_count_TEST_SIL_POSITION_TRANSFER_1/Fri_May_22_01:23:48_2020/seed_0/Position_Transfer_SIL_HRL_E_GREEDY - GE_MazeKeyDoor8keyDoor1-v0')

x_sil_reward_n_steps_env4, y_sil_reward_n_steps_env4 = open_file_2('/home/lorenzo/Documenti/UPF/DeepRL/results/TEST  -  TEST_SIL_POSITION_TRANSFER_1/Fri_May_22_01:24:18_2020/seed_0/Position_Transfer_SIL - GE_MazeKeyDoor8keyDoor1-v0')

x_hrlsil_reward_n_steps_env4, y_hrlsil_reward_n_steps_env4 = open_file_2('/home/lorenzo/Documenti/UPF/DeepRL/results/TEST  -  heuristic_count_TEST_SIL_POSITION_TRANSFER_1/Fri_May_22_01:23:48_2020/seed_0/Position_Transfer_SIL_HRL_E_GREEDY - GE_MazeKeyDoor8keyDoor1-v0')
























fig, ax = plt.subplots(3, 4, figsize=(15,15))
ax[0,0].set_title("KeyDoor8x8_1")
ax[0,0].imshow(img_env_1)
ax[1,0].set_title("rewards_over_episodes")
ax[1,0].plot(x_sil_reward_ep_env1, y_sil_reward_ep_env1, '-b', label='SIL')
ax[1,0].plot(x_hrlsil_reward_ep_env1, y_hrlsil_reward_ep_env1, '--r', label='HRL_SIL')
ax[2,0].set_title("rewards_over_nsteps")
ax[2,0].plot(x_sil_reward_n_steps_env1, y_sil_reward_n_steps_env1, '-b', label='SIL')
ax[2,0].plot(x_hrlsil_reward_n_steps_env1, y_hrlsil_reward_n_steps_env1, '--r', label='HRL_SIL')

ax[0,1].set_title("KeyDoor8x8_2")
ax[0,1].imshow(img_env_2)
ax[1,1].set_title("rewards_over_episodes")
ax[1,1].plot(x_sil_reward_ep_env2, y_sil_reward_ep_env2, '-b', label='SIL')
ax[1,1].plot(x_hrlsil_reward_ep_env2, y_hrlsil_reward_ep_env2, '--r', label='HRL_SIL')
ax[2,1].set_title("rewards_over_nsteps")
ax[2,1].plot(x_sil_reward_n_steps_env2, y_sil_reward_n_steps_env2, '-b', label='SIL')
ax[2,1].plot(x_hrlsil_reward_n_steps_env2, y_hrlsil_reward_n_steps_env2, '--r', label='HRL_SIL')

ax[0,2].set_title("KeyDoor8x8_3")
ax[0,2].imshow(img_env_3)
ax[1,2].set_title("rewards_over_episodes")
ax[1,2].plot(x_sil_reward_ep_env3, y_sil_reward_ep_env3, '-b', label='SIL')
ax[1,2].plot(x_hrlsil_reward_ep_env3, y_hrlsil_reward_ep_env3, '--r', label='HRL_SIL')
ax[2,2].set_title("rewards_over_nsteps")
ax[2,2].plot(x_sil_reward_n_steps_env3, y_sil_reward_n_steps_env3, '-b', label='SIL')
ax[2,2].plot(x_hrlsil_reward_n_steps_env3, y_hrlsil_reward_n_steps_env3, '--r', label='HRL_SIL')

ax[0,3].set_title("KeyDoor8x8_1")
ax[0,3].imshow(img_env_1)
ax[1,3].set_title("rewards_over_episodes")
ax[1,3].plot(x_sil_reward_ep_env4, y_sil_reward_ep_env4, '-b', label='SIL')
ax[1,3].plot(x_hrlsil_reward_ep_env4, y_hrlsil_reward_ep_env4, '--r', label='HRL_SIL')
ax[2,3].set_title("rewards_over_nsteps")
ax[2,3].plot(x_sil_reward_n_steps_env4, y_sil_reward_n_steps_env4, '-b', label='SIL')
ax[2,3].plot(x_hrlsil_reward_n_steps_env4, y_hrlsil_reward_n_steps_env4, '--r', label='HRL_SIL')

leg = ax[1,0].legend();
leg = ax[2,0].legend();
leg = ax[1,1].legend();
leg = ax[2,1].legend();
leg = ax[1,2].legend();
leg = ax[2,2].legend();
leg = ax[1,3].legend();
leg = ax[2,3].legend();

plt.show()
