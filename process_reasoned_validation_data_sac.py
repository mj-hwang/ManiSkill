import h5py
import numpy as np
import torch
import random
from transformers import AutoTokenizer, T5Config, T5EncoderModel

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
        "A": "./examples/baselines/sac/datasets/PickLargerCube-v1/sac_policy/valid.h5",
        "B": "./examples/baselines/sac/datasets/PickSmallerCube-v1/sac_policy/valid.h5",
    },
    "pick_larger_swapped": {
        "A": "./examples/baselines/sac/datasets/PickLargerCubeSwapped-v1/sac_policy/valid.h5",
        "B": "./examples/baselines/sac/datasets/PickSmallerCubeSwapped-v1/sac_policy/valid.h5",
    },
    "place_larger": {
        "A": "./examples/baselines/sac/datasets/PlaceLargerCube-v1/sac_policy/valid.h5",
        "B": "./examples/baselines/sac/datasets/PlaceSmallerCube-v1/sac_policy/valid.h5",
    },
    "place_larger_swapped": {
        "A": "./examples/baselines/sac/datasets/PlaceLargerCubeSwapped-v1/sac_policy/valid.h5",
        "B": "./examples/baselines/sac/datasets/PlaceSmallerCubeSwapped-v1/sac_policy/valid.h5",
    },
    "push_larger": {
        "A": "./examples/baselines/sac/datasets/PushLargerCube-v1/sac_policy/valid.h5",
        "B": "./examples/baselines/sac/datasets/PushSmallerCube-v1/sac_policy/valid.h5",
    },
    "push_larger_swapped": {
        "A": "./examples/baselines/sac/datasets/PushLargerCubeSwapped-v1/sac_policy/valid.h5",
        "B": "./examples/baselines/sac/datasets/PushSmallerCubeSwapped-v1/sac_policy/valid.h5",
    },
    "pull_larger": {
        "A": "./examples/baselines/sac/datasets/PullLargerCube-v1/sac_policy/valid.h5",
        "B": "./examples/baselines/sac/datasets/PullSmallerCube-v1/sac_policy/valid.h5",
    },
    "pull_larger_swapped": {
        "A": "./examples/baselines/sac/datasets/PullLargerCubeSwapped-v1/sac_policy/valid.h5",
        "B": "./examples/baselines/sac/datasets/PullSmallerCubeSwapped-v1/sac_policy/valid.h5",
    },
}



# TASK_NLS = {
#     "pick_larger": "pick up larger cube to green sphere",
#     "push_larger": "push larger cube toward green line",
#     "place_larger": "place larger cube in green bin",
#     "pull_larger": "pull larger cube toward green line",
# }

TASK_NLS = {
    "pick_larger": "pick up larger cube to green sphere",
    "pick_larger_swapped": "pick up larger cube to green sphere",
    "push_larger": "push larger cube toward green line",
    "push_larger_swapped": "push larger cube toward green line",
    "place_larger": "place larger cube in green bin",
    "place_larger_swapped": "place larger cube in green bin",
    "pull_larger": "pull larger cube toward green line",
    "pull_larger_swapped": "pull larger cube toward green line",
}

def set_seed(seed_value):
    """Sets a global seed for reproducibility across all libraries."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    
    # These lines are crucial for CUDA operations to be deterministic.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# IMPORTANT: followings are task labels
# 1: pick_larger
# 2: pick_larger_swapped
# 3: push_larger
# 4: push_larger_swapped
# 5: place_larger
# 6: place_larger_swapped
# 7: pull_larger
# 8: pull_larger_swapped

TRAJ_LEN = 4
LANG_MODEL_NAME = "google-t5/t5-small"

def main():
    set_seed(42)
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
            obs_1.append(f1[keys_1[i]]["obs"][start_idx_1:start_idx_1 + TRAJ_LEN])
            actions_1.append(f1[keys_1[i]]["actions"][start_idx_1:start_idx_1 + TRAJ_LEN])

            traj_len_2 = len(f2[keys_2[i]]["actions"])
            start_idx_2 = random.randint(0, traj_len_2 - TRAJ_LEN)
            obs_2.append(f2[keys_2[i]]["obs"][start_idx_2:start_idx_2 + TRAJ_LEN])
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
    print("###### PHASE 2: processing language (reason & task) data ######")
    tokenizer = AutoTokenizer.from_pretrained(LANG_MODEL_NAME)
    lang_encoder = T5EncoderModel.from_pretrained(LANG_MODEL_NAME)
    lang_encoder.config.dropout_rate = 0.0
    lang_encoder.eval()

    for param in lang_encoder.parameters():
        param.requires_grad = False

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
    print(task_embeddings.mean(1))
    task_embeddings = torch.repeat_interleave(task_embeddings, NUM_TRAJ_PER_TASK, dim=0)
    task_embeddings = task_embeddings.cpu().numpy()

    task_labels = np.repeat(np.arange(len(TASKS)), NUM_TRAJ_PER_TASK)

    print("task_embeddings shape:", task_embeddings.shape)
    print("task_labels shape:", task_labels.shape)

    print("###### PHASE 3: saving data ######")
    np.savez_compressed(
        f"data_all_valid_{NUM_TRAJ_PER_TASK * len(TASKS)}_sac.npz", 
        obs_1=obs_1, 
        action_1=actions_1, 
        obs_2=obs_2, 
        action_2=actions_2, 
        label=labels,
        task_embeddings=task_embeddings,
        task_labels=task_labels,
    )

if __name__ == '__main__':
    main()
