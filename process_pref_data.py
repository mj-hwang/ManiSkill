import h5py
import numpy as np
import random

PATH_1 = "./demos/PickLargerCube-v1/motionplanning/20250308_220733.h5"
PATH_2 = "./demos/PickSmallerCube-v1/motionplanning/20250308_215535.h5"

DATASET_SIZE = 5000
TRAJ_LEN = 64

f1 = h5py.File(PATH_1, "r")
f2 = h5py.File(PATH_1, "r")

obs_1 = []
actions_1 = []

obs_2 = []
actions_2 = []

# for key in f1.keys():
#     obs_1.append(f1[key]["obs"]["sensor_data"]["base_camera"]["rgb"][:-1])
#     actions_1.append(f1[key]["actions"])

# for key in f2.keys():
#     obs_2.append(f1[key]["obs"]["sensor_data"]["base_camera"]["rgb"][:-1])
#     actions_2.append(f1[key]["actions"])

keys_1 = random.choices(list(f1.keys()), k=DATASET_SIZE)
keys_2 = random.choices(list(f2.keys()), k=DATASET_SIZE)

for i in range(DATASET_SIZE):
    print(i)
    traj_len_1 = len(f1[keys_1[i]]["actions"])
    start_idx_1 = random.randint(0, traj_len_1 - TRAJ_LEN)
    obs_1.append(f1[keys_1[i]]["obs"]["sensor_data"]["base_camera"]["rgb"][start_idx_1:start_idx_1 + TRAJ_LEN])
    actions_1.append(f1[keys_1[i]]["actions"][start_idx_1:start_idx_1 + TRAJ_LEN])

    traj_len_2 = len(f2[keys_2[i]]["actions"])
    start_idx_2 = random.randint(0, traj_len_2 - TRAJ_LEN)
    obs_2.append(f2[keys_2[i]]["obs"]["sensor_data"]["base_camera"]["rgb"][start_idx_2:start_idx_2 + TRAJ_LEN])
    actions_2.append(f2[keys_2[i]]["actions"][start_idx_2:start_idx_2 + TRAJ_LEN])

obs_1 = np.array(obs_1)
actions_1 = np.array(actions_1)

print("obs_1 shape:", obs_1.shape)
print("actions_1 shape:", actions_1.shape)

obs_2 = np.array(obs_2)
actions_2 = np.array(actions_2)

print("obs_2 shape:", obs_2.shape)
print("actions_2 shape:", actions_2.shape)

labels = np.zeros(len(obs_1))

np.savez_compressed("data.npz", obs_1=obs_1, actions_1=actions_1, obs_2=obs_2, actions_2=actions_2, labels=labels)

f1.close()
f2.close()


