import h5py
import numpy as np
import random

# NOTE: Trajectory A referes to preferred trajectory.
TASKS = [
    # "pick_larger",
    "push_larger",
    # "place_larger",
    # "pull_larger",
]

NUM_TRAJ_PER_TASK = 500

DATA_PATHS = {
    "pick_larger": {
        "A": "./demos/PickLargerCube-v1/motionplanning/pick_larger_train_1000.rgb.pd_ee_delta_pose.physx_cpu.h5",
        "B": "./demos/PickSmallerCube-v1/motionplanning/pick_smaller_train_1000.rgb.pd_ee_delta_pose.physx_cpu.h5",
    },
    "push_larger": {
        "A": "./demos/PushLargerCube-v1/motionplanning/push_larger_train_1000.rgb.pd_ee_delta_pose.physx_cpu.h5",
        "B": "./demos/PushSmallerCube-v1/motionplanning/push_smaller_train_1000.rgb.pd_ee_delta_pose.physx_cpu.h5",
    },
    "place_larger": {
        "A": "./demos/PlaceLargerCube-v1/motionplanning/place_larger_train_1000.rgb.pd_ee_delta_pose.physx_cpu.h5",
        "B": "./demos/PlaceSmallerCube-v1/motionplanning/place_smaller_train_1000.rgb.pd_ee_delta_pose.physx_cpu.h5",
    },
    "pull_larger": {
        "A": "./demos/PullLargerCube-v1/motionplanning/pull_larger_train_1000.rgb.pd_ee_delta_pose.physx_cpu.h5",
        "B": "./demos/PullSmallerCube-v1/motionplanning/pull_smaller_train_1000.rgb.pd_ee_delta_pose.physx_cpu.h5",
    },
}

TRAJ_LEN = 64
LANG_MODEL_NAME = "google-t5/t5-small"

def main():
    print("###### PHASE 1: processing trajectory data ######")
    obs_1 = []
    actions_1 = []
    obs_2 = []
    actions_2 = []

    for task in TASKS:
        print(task)
        f1 = h5py.File(DATA_PATHS[task]["A"], "r")
        f2 = h5py.File(DATA_PATHS[task]["B"], "r")

        keys_1 = list(f1.keys())
        keys_2 = list(f2.keys())

        i = 0
        num_traj = 0
        while num_traj < NUM_TRAJ_PER_TASK:
            print(f"Trajectory Index: {i}")
            print(f"Num Trajectories so far: {num_traj}")
            traj_len_1 = len(f1[keys_1[i]]["actions"])
            traj_len_2 = len(f2[keys_2[i]]["actions"])
            if traj_len_1 < TRAJ_LEN or traj_len_2 < TRAJ_LEN:
                print(f"detected a trajectory shorter than {TRAJ_LEN}; len_1: {traj_len_1}, len_2: {traj_len_2}")
                i += 1
                continue

            start_idx_1 = random.randint(0, traj_len_1 - TRAJ_LEN)
            ob1 = f1[keys_1[i]]["obs"]["sensor_data"]["base_camera"]["rgb"][start_idx_1:start_idx_1 + TRAJ_LEN]
            ob1 = np.transpose(ob1, (0, 3, 1, 2))
            obs_1.append(ob1)
            actions_1.append(f1[keys_1[i]]["actions"][start_idx_1:start_idx_1 + TRAJ_LEN])

            start_idx_2 = random.randint(0, traj_len_2 - TRAJ_LEN)
            ob2 = f2[keys_2[i]]["obs"]["sensor_data"]["base_camera"]["rgb"][start_idx_2:start_idx_2 + TRAJ_LEN]
            ob2 = np.transpose(ob2, (0, 3, 1, 2))
            obs_2.append(ob2)
            actions_2.append(f2[keys_2[i]]["actions"][start_idx_2:start_idx_2 + TRAJ_LEN])
            i += 1
            num_traj += 1

        f1.close()
        f2.close()

    obs_1 = np.array(obs_1)
    actions_1 = np.array(actions_1)

    print("obs_1 shape:", obs_1.shape)
    print("actions_1 shape:", actions_1.shape)

    obs_2 = np.array(obs_2)
    actions_2 = np.array(actions_2)

    print("obs_2 shape:", obs_2.shape)
    print("actions_2 shape:", actions_2.shape)

    labels = np.zeros(len(obs_1))

    print("labels shape:", labels.shape)

    print("###### PHASE 3: saving data ######")
    np.savez_compressed(
        f"data_{'_'.join(TASKS)}_train_{NUM_TRAJ_PER_TASK * len(TASKS)}_updated.npz", 
        obs_1=obs_1, 
        action_1=actions_1, 
        obs_2=obs_2, 
        action_2=actions_2, 
        label=labels,
    )


if __name__ == '__main__':
    main()
