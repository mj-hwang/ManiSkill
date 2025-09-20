import h5py
import numpy as np
import torch
import random
from transformers import AutoTokenizer, T5Config, T5EncoderModel

# NOTE: Trajectory A referes to preferred trajectory.
TASKS = [
    "pick_larger",
    "push_larger",
    "place_larger",
    "pull_larger",
]

NUM_TRAJ_PER_TASK = 1000

# DATA_PATHS = {
#     "pick_larger": {
#         "A": "./demos/PickLargerCube-v1/motionplanning/pick_larger_train_1000.rgb.pd_ee_delta_pose.physx_cpu.h5",
#         "B": "./demos/PickSmallerCube-v1/motionplanning/pick_smaller_train_1000.rgb.pd_ee_delta_pose.physx_cpu.h5",
#     },
#     "push_larger": {
#         "A": "./demos/PushLargerCube-v1/motionplanning/push_larger_train_1000.rgb.pd_ee_delta_pose.physx_cpu.h5",
#         "B": "./demos/PushSmallerCube-v1/motionplanning/push_smaller_train_1000.rgb.pd_ee_delta_pose.physx_cpu.h5",
#     },
#     "place_larger": {
#         "A": "./demos/PlaceLargerCube-v1/motionplanning/place_larger_train_1000.rgb.pd_ee_delta_pose.physx_cpu.h5",
#         "B": "./demos/PlaceSmallerCube-v1/motionplanning/place_smaller_train_1000.rgb.pd_ee_delta_pose.physx_cpu.h5",
#     },
#     "pull_larger": {
#         "A": "./demos/PullLargerCube-v1/motionplanning/pull_larger_train_1000.rgb.pd_ee_delta_pose.physx_cpu.h5",
#         "B": "./demos/PullSmallerCube-v1/motionplanning/pull_smaller_train_1000.rgb.pd_ee_delta_pose.physx_cpu.h5",
#     },
# }

DATA_PATHS = {
    "pick_larger": {
        "A": "./demos/PickLargerCube-v1/motionplanning/pick_larger_train_1000_state.state.pd_ee_delta_pose.physx_cpu.h5",
        "B": "./demos/PickSmallerCube-v1/motionplanning/pick_smaller_train_1000_state.state.pd_ee_delta_pose.physx_cpu.h5",
    },
    "push_larger": {
        "A": "./demos/PushLargerCube-v1/motionplanning/push_larger_train_1000_state.state.pd_ee_delta_pose.physx_cpu.h5",
        "B": "./demos/PushSmallerCube-v1/motionplanning/push_smaller_train_1000_state.state.pd_ee_delta_pose.physx_cpu.h5",
    },
    "place_larger": {
        "A": "./demos/PlaceLargerCube-v1/motionplanning/place_larger_train_1000_state.state.pd_ee_delta_pose.physx_cpu.h5",
        "B": "./demos/PlaceSmallerCube-v1/motionplanning/place_smaller_train_1000_state.state.pd_ee_delta_pose.physx_cpu.h5",
    },
    "pull_larger": {
        "A": "./demos/PullLargerCube-v1/motionplanning/pull_larger_train_1000_state.state.pd_ee_delta_pose.physx_cpu.h5",
        "B": "./demos/PullSmallerCube-v1/motionplanning/pull_smaller_train_1000_state.state.pd_ee_delta_pose.physx_cpu.h5",
    },
}

TASK_NLS = {
    "pick_larger": "pick up larger cube to green sphere",
    "push_larger": "push larger cube toward green line",
    "place_larger": "place larger cube in green bin",
    "pull_larger": "pull larger cube toward green line",
}

REASON_NLS = {
    "pick_larger": "cube is larger",
    "push_larger": "cube is larger",
    "place_larger": "cube is larger",
    "pull_larger": "cube is larger",
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
            traj_len_1 = len(f1[keys_1[i]]["actions"])
            traj_len_2 = len(f2[keys_2[i]]["actions"])
            if traj_len_1 < TRAJ_LEN or traj_len_2 < TRAJ_LEN:
                print(f"Trajectory Index: {i}")
                print(f"Num Trajectories so far: {num_traj}")
                print(f"detected a trajectory shorter than {TRAJ_LEN}; len_1: {traj_len_1}, len_2: {traj_len_2}")
                i += 1
                continue

            start_idx_1 = random.randint(0, traj_len_1 - TRAJ_LEN)
            obs_1.append(f1[keys_1[i]]["obs"][start_idx_1:start_idx_1 + TRAJ_LEN])
            actions_1.append(f1[keys_1[i]]["actions"][start_idx_1:start_idx_1 + TRAJ_LEN])

            start_idx_2 = random.randint(0, traj_len_2 - TRAJ_LEN)
            obs_2.append(f2[keys_2[i]]["obs"][start_idx_2:start_idx_2 + TRAJ_LEN])
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

    print("###### PHASE 2: processing language (reason & task) data ######")
    tokenizer = AutoTokenizer.from_pretrained(LANG_MODEL_NAME)
    lang_encoder = T5EncoderModel.from_pretrained(LANG_MODEL_NAME)
    lang_encoder.config.dropout_rate = 0.0
    lang_encoder.eval()

    for param in lang_encoder.parameters():
        param.requires_grad = False

    tokenized_reasons = tokenizer(
        [REASON_NLS[task] for task in TASKS],
        padding=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    reason_tokens = tokenized_reasons["input_ids"]
    reason_masks = tokenized_reasons["attention_mask"]
    reason_lhs = lang_encoder(reason_tokens, attention_mask=reason_masks).last_hidden_state
    float_reason_mask = reason_masks.unsqueeze(-1).type_as(reason_lhs)
    masked_reason_lhs = reason_lhs * float_reason_mask
    reason_embeddings = masked_reason_lhs.sum(dim=1) / float_reason_mask.sum(dim=1)
    reason_embeddings = torch.repeat_interleave(reason_embeddings, NUM_TRAJ_PER_TASK, dim=0)
    reason_embeddings = reason_embeddings.cpu().numpy()

    tokenized_tasks = tokenizer(
        [TASK_NLS[task] for task in TASKS],
        padding=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    task_tokens = tokenized_tasks["input_ids"]
    task_masks = tokenized_tasks["attention_mask"]
    task_lhs = lang_encoder(task_tokens, attention_mask=task_masks).last_hidden_state
    float_task_mask = task_masks.unsqueeze(-1).type_as(task_lhs)
    masked_task_lhs = task_lhs * float_task_mask
    task_embeddings = masked_task_lhs.sum(dim=1) / float_task_mask.sum(dim=1)
    task_embeddings = torch.repeat_interleave(task_embeddings, NUM_TRAJ_PER_TASK, dim=0)
    task_embeddings = task_embeddings.cpu().numpy()

    task_labels = np.repeat(np.arange(len(TASKS)), NUM_TRAJ_PER_TASK)

    print("reason_embeddings shape:", reason_embeddings.shape)
    print("task_embeddings shape:", task_embeddings.shape)
    print("task_labels shape:", task_labels.shape)

    print("###### PHASE 3: saving data ######")
    np.savez_compressed(
        f"data_{'_'.join(TASKS)}_train_{NUM_TRAJ_PER_TASK * len(TASKS)}_state.npz", 
        obs_1=obs_1, 
        action_1=actions_1, 
        obs_2=obs_2, 
        action_2=actions_2, 
        label=labels,
        reason_embeddings=reason_embeddings,
        task_embeddings=task_embeddings,
        task_labels=task_labels,
    )


if __name__ == '__main__':
    main()
