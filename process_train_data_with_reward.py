import h5py
import numpy as np
import random

# NOTE: Trajectory A referes to preferred trajectory.
TASKS = [
    # "pick_larger",
    # "push_larger",
    # "place_larger",
    "pull_larger",
]

NUM_TRAJ_PER_TASK = 500

DATA_PATHS = {
    "pick_larger": {
        # "A": "./demos/PickLargerCube-v1/motionplanning/pick_larger_train_1000.rgb.pd_ee_delta_pose.physx_cpu.h5",
        "A": "./demos/PickLargerCube-v1/motionplanning/pick_larger_train_reward_1000.rgb.pd_ee_delta_pose.physx_cpu.h5",
    },
    "push_larger": {
        # "A": "./demos/PushLargerCube-v1/motionplanning/push_larger_train_1000.rgb.pd_ee_delta_pose.physx_cpu.h5",
        "A": "./demos/PushLargerCube-v1/motionplanning/push_larger_train_reward_1000.rgb.pd_ee_delta_pose.physx_cpu.h5",
    },
    "place_larger": {
        # "A": "./demos/PlaceLargerCube-v1/motionplanning/place_larger_train_1000.rgb.pd_ee_delta_pose.physx_cpu.h5",
        "A": "./demos/PlaceLargerCube-v1/motionplanning/place_larger_train_reward_1000.rgb.pd_ee_delta_pose.physx_cpu.h5",
    },
    "pull_larger": {
        # "A": "./demos/PullLargerCube-v1/motionplanning/pull_larger_train_1000.rgb.pd_ee_delta_pose.physx_cpu.h5",
        "A": "./demos/PullLargerCube-v1/motionplanning/pull_larger_train_reward_1000.rgb.pd_ee_delta_pose.physx_cpu.h5",
    },
}

TRAJ_LEN = 64

def main():
    print("###### PHASE 1: processing trajectory data ######")
    obs = []
    next_obs = []
    actions = []
    rewards = []

    for task in TASKS:
        print(task)
        f1 = h5py.File(DATA_PATHS[task]["A"], "r")
        keys_1 = list(f1.keys())

        i = 0
        num_traj = 0
        while num_traj < NUM_TRAJ_PER_TASK:
            print(f"Trajectory Index: {i}")
            print(f"Num Trajectories so far: {num_traj}")
            traj_len = len(f1[keys_1[i]]["actions"])
            # print(traj_len, TRAJ_LEN)
            if traj_len < TRAJ_LEN + 1:
                print(f"detected a trajectory shorter than {TRAJ_LEN}; len_1: {traj_len}")
                i += 1
                continue

            start_idx = random.randint(0, traj_len - TRAJ_LEN - 1)
            ob = f1[keys_1[i]]["obs"]["sensor_data"]["base_camera"]["rgb"][start_idx:start_idx + TRAJ_LEN]
            ob = np.transpose(ob, (0, 3, 1, 2))
            obs.append(ob)
            
            next_ob = f1[keys_1[i]]["obs"]["sensor_data"]["base_camera"]["rgb"][start_idx + 1:start_idx + TRAJ_LEN + 1]
            next_ob = np.transpose(next_ob, (0, 3, 1, 2))
            next_obs.append(next_ob)
            
            actions.append(f1[keys_1[i]]["actions"][start_idx:start_idx + TRAJ_LEN])
            rewards.append(f1[keys_1[i]]["rewards"][start_idx:start_idx + TRAJ_LEN])

            i += 1
            num_traj += 1

        f1.close()

    obs = np.array(obs)
    next_obs = np.array(next_obs)
    actions = np.array(actions)
    rewards = np.array(rewards)

    print("obs shape:", obs.shape)
    print("next_obs shape:", next_obs.shape)
    print("actions shape:", actions.shape)
    print("rewards shape:", rewards.shape)


    print("###### PHASE 3: saving data ######")
    np.savez_compressed(
        f"data_{'_'.join(TASKS)}_train_{NUM_TRAJ_PER_TASK * len(TASKS)}_with_reward.npz", 
        obs=obs,
        next_obs=next_obs,
        action=actions,
        rewards=rewards,
    )


if __name__ == '__main__':
    main()
