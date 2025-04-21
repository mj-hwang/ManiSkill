import h5py
import numpy as np
import random
from transformers import AutoTokenizer

# NOTE: Trajectory A referes to preferred trajectory.
TASKS = [
    "pick_larger", "pick_larger_swapped",
    "push_larger", "push_larger_swapped",
    "place_larger", "place_larger_swapped",
    "pull_larger", "pull_larger_swapped",
    # "pick_smaller", "pick_smaller_swapped", 
    # "push_smaller", "push_smaller_swapped",
]

NUM_TRAJ_PER_TASK = 50

DATA_PATHS = {
    "pick_larger": {
        "A": "./demos/PickLargerCube-v1/motionplanning/pick_larger_validation_100.rgb.pd_ee_delta_pose.physx_cpu.h5",
        "B": "./demos/PickSmallerCube-v1/motionplanning/pick_smaller_validation_100.rgb.pd_ee_delta_pose.physx_cpu.h5",
    },
    "pick_larger_swapped": {
        "A": "./demos/PickLargerCubeSwapped-v1/motionplanning/pick_larger_validation_100.rgb.pd_ee_delta_pose.physx_cpu.h5",
        "B": "./demos/PickSmallerCubeSwapped-v1/motionplanning/pick_smaller_validation_100.rgb.pd_ee_delta_pose.physx_cpu.h5",
    },
    "push_larger": {
        "A": "./demos/PushLargerCube-v1/motionplanning/push_larger_validation_100.rgb.pd_ee_delta_pose.physx_cpu.h5",
        "B": "./demos/PushSmallerCube-v1/motionplanning/push_smaller_validation_100.rgb.pd_ee_delta_pose.physx_cpu.h5",
    }, 
    "push_larger_swapped": {
        "A": "./demos/PushLargerCubeSwapped-v1/motionplanning/push_larger_validation_100.rgb.pd_ee_delta_pose.physx_cpu.h5",
        "B": "./demos/PushSmallerCubeSwapped-v1/motionplanning/push_smaller_validation_100.rgb.pd_ee_delta_pose.physx_cpu.h5",
    },
    "place_larger": {
        "A": "./demos/PlaceLargerCube-v1/motionplanning/place_larger_validation_100.rgb.pd_ee_delta_pose.physx_cpu.h5",
        "B": "./demos/PlaceSmallerCube-v1/motionplanning/place_smaller_validation_100.rgb.pd_ee_delta_pose.physx_cpu.h5",
    }, 
    "place_larger_swapped": {
        "A": "./demos/PlaceLargerCubeSwapped-v1/motionplanning/place_larger_validation_100.rgb.pd_ee_delta_pose.physx_cpu.h5",
        "B": "./demos/PlaceSmallerCubeSwapped-v1/motionplanning/place_smaller_validation_100.rgb.pd_ee_delta_pose.physx_cpu.h5",
    },
    "pull_larger": {
        "A": "./demos/PullLargerCube-v1/motionplanning/pull_larger_validation_100.rgb.pd_ee_delta_pose.physx_cpu.h5",
        "B": "./demos/PullSmallerCube-v1/motionplanning/pull_smaller_validation_100.rgb.pd_ee_delta_pose.physx_cpu.h5",
    }, 
    "pull_larger_swapped": {
        "A": "./demos/PullLargerCubeSwapped-v1/motionplanning/pull_larger_validation_100.rgb.pd_ee_delta_pose.physx_cpu.h5",
        "B": "./demos/PullSmallerCubeSwapped-v1/motionplanning/pull_smaller_validation_100.rgb.pd_ee_delta_pose.physx_cpu.h5",
    },
}

TASK_NLS = {
    "pick_larger": "Pick up a larger cube to the green target sphere",
    "pick_larger_swapped": "Pick up a larger cube to the green target sphere",
    "push_larger": "Push a larger cube toward the green line",
    "push_larger_swapped": "Push a larger cube toward the green line",
    "place_larger": "Place a larger cube in the green bin",
    "place_larger_swapped": "Place a larger cube in the green bin",
    "pull_larger": "Pull a larger cube toward the green line",
    "pull_larger_swapped": "Pull a larger cube toward the green line",
}


# IMPORTANT: followings are task labels
# 1: pick_larger
# 2: pick_larger_swapped
# 3: push_larger
# 4: push_larger_swapped
# 5: place_larger
# 6: place_larger_swapped
# 7: pull_larger
# 8: pull_larger_swapped

TRAJ_LEN = 64
LANG_MODEL_NAME = "google-t5/t5-small"

def main():
    print("###### PHASE 1: processing trajectory data ######")
    obs_1 = []
    actions_1 = []
    obs_2 = []
    actions_2 = []

    for task in TASKS:
        f1 = h5py.File(DATA_PATHS[task]["A"], "r")
        f2 = h5py.File(DATA_PATHS[task]["B"], "r")

        keys_1 = list(f1.keys())
        keys_2 = list(f2.keys())

        for i in range(NUM_TRAJ_PER_TASK):
            print(i)
            traj_len_1 = len(f1[keys_1[i]]["actions"])
            start_idx_1 = random.randint(0, traj_len_1 - TRAJ_LEN)
            ob1 = f1[keys_1[i]]["obs"]["sensor_data"]["base_camera"]["rgb"][start_idx_1:start_idx_1 + TRAJ_LEN]
            ob1 = np.transpose(ob1, (0, 3, 1, 2))
            obs_1.append(ob1)
            actions_1.append(f1[keys_1[i]]["actions"][start_idx_1:start_idx_1 + TRAJ_LEN])

            traj_len_2 = len(f2[keys_2[i]]["actions"])
            start_idx_2 = random.randint(0, traj_len_2 - TRAJ_LEN)
            ob2 = f2[keys_2[i]]["obs"]["sensor_data"]["base_camera"]["rgb"][start_idx_2:start_idx_2 + TRAJ_LEN]
            ob2 = np.transpose(ob2, (0, 3, 1, 2))
            obs_2.append(ob2)
            actions_2.append(f2[keys_2[i]]["actions"][start_idx_2:start_idx_2 + TRAJ_LEN])
        
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

    task_labels = np.repeat([1, 2, 3, 4, 5, 6, 7, 8], NUM_TRAJ_PER_TASK)

    print("task_labels shape:", task_labels.shape)

    print("###### PHASE 2: processing language (reason & task) data ######")
    tokenizer = AutoTokenizer.from_pretrained(LANG_MODEL_NAME)

    tokenized_task = tokenizer(
        [TASK_NLS[task] for task in TASKS],
        padding=True,
        add_special_tokens=True,
        return_tensors="np",
    )
    task_tokens = np.repeat(
        tokenized_task["input_ids"],
        repeats=NUM_TRAJ_PER_TASK,
        axis=0,
    )
    task_masks = np.repeat(
        tokenized_task["attention_mask"],
        repeats=NUM_TRAJ_PER_TASK,
        axis=0,
    )

    print("task_tokens shape:", task_tokens.shape)
    print("task_masks shape:", task_masks.shape)

    print("###### PHASE 3: saving data ######")
    np.savez_compressed(
        f"data_all_valid_{NUM_TRAJ_PER_TASK * len(TASKS)}_updated.npz", 
        obs_1=obs_1, 
        action_1=actions_1, 
        obs_2=obs_2, 
        action_2=actions_2, 
        label=labels,
        task_tokens=task_tokens,
        task_masks=task_masks,
        task_labels=task_labels,
    )

if __name__ == '__main__':
    main()
