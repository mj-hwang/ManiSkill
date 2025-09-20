import h5py
import numpy as np
import torch
import random
from transformers import AutoTokenizer, T5Config, T5EncoderModel

# NOTE: Trajectory A referes to preferred trajectory.


# TASKS = [
#     "pick_larger_swapped",
#     "push_larger_swapped",
#     "place_larger_swapped",
#     "pull_larger_swapped",
# ]

# TASK = "pick_larger"
TASK = "pull_larger"

NUM_TRAJ_PER_TASK = 1000

DATA_PATHS = {
    "pick_larger": {
        "A": "./examples/baselines/sac/datasets/PickLargerCube-v1/sac_policy/offlinerl.h5",
        "B": "./examples/baselines/sac/datasets/PickSmallerCube-v1/sac_policy/offlinerl.h5",
    },
    "pick_larger_swapped": {
        "A": "./examples/baselines/sac/datasets/PickLargerCubeSwapped-v1/sac_policy/offlinerl.h5",
        "B": "./examples/baselines/sac/datasets/PickSmallerCubeSwapped-v1/sac_policy/offlinerl.h5",
    },
    "place_larger": {
        "A": "./examples/baselines/sac/datasets/PlaceLargerCube-v1/sac_policy/offlinerl.h5",
        "B": "./examples/baselines/sac/datasets/PlaceSmallerCube-v1/sac_policy/offlinerl.h5",
    },
    "place_larger_swapped": {
        "A": "./examples/baselines/sac/datasets/PlaceLargerCubeSwapped-v1/sac_policy/offlinerl.h5",
        "B": "./examples/baselines/sac/datasets/PlaceSmallerCubeSwapped-v1/sac_policy/offlinerl.h5",
    },
    "push_larger": {
        "A": "./examples/baselines/sac/datasets/PushLargerCube-v1/sac_policy/offlinerl.h5",
        "B": "./examples/baselines/sac/datasets/PushSmallerCube-v1/sac_policy/offlinerl.h5",
    },
    "push_larger_swapped": {
        "A": "./examples/baselines/sac/datasets/PushLargerCubeSwapped-v1/sac_policy/offlinerl.h5",
        "B": "./examples/baselines/sac/datasets/PushSmallerCubeSwapped-v1/sac_policy/offlinerl.h5",
    },
    "pull_larger": {
        "A": "./examples/baselines/sac/datasets/PullLargerCube-v1/sac_policy/offlinerl.h5",
        "B": "./examples/baselines/sac/datasets/PullSmallerCube-v1/sac_policy/offlinerl.h5",
    },
    "pull_larger_swapped": {
        "A": "./examples/baselines/sac/datasets/PullLargerCubeSwapped-v1/sac_policy/offlinerl.h5",
        "B": "./examples/baselines/sac/datasets/PullSmallerCubeSwapped-v1/sac_policy/offlinerl.h5",
    },
}

TASK_NLS = {
    "pick_larger": "pick up larger cube to green sphere",
    "push_larger": "push larger cube toward green line",
    "place_larger": "place larger cube in green bin",
    "pull_larger": "pull larger cube toward green line",
    "pick_larger_swapped": "pick up larger cube to green sphere",
    "push_larger_swapped": "push larger cube toward green line",
    "place_larger_swapped": "place larger cube in green bin",
    "pull_larger_swapped": "pull larger cube toward green line",
}

TRAJ_LEN = 4
LANG_MODEL_NAME = "google-t5/t5-small"

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
    
    print(f"Global seed set to {seed_value} for full reproducibility.")

def main():
    set_seed(42)
    print("###### PHASE 1: processing trajectory data ######")
    obs_1 = []
    next_obs_1 = []
    actions_1 = []
    # rewards_1 = []

    print(TASK)

    f1 = h5py.File(DATA_PATHS[TASK]["A"], "r")

    keys_1 = list(f1.keys())

    i = 0
    num_traj = 0
    while num_traj < NUM_TRAJ_PER_TASK:
        print(f"Trajectory Index: {i}")
        print(f"Num Trajectories so far: {num_traj}")
        traj_len_1 = len(f1[keys_1[i]]["actions"])
        if traj_len_1 < TRAJ_LEN:
            print(f"detected a trajectory shorter than {TRAJ_LEN}; len_1: {traj_len_1}")
            i += 1
            continue

        # start_idx_1 = random.randint(0, traj_len_1 - TRAJ_LEN - 1)
        # # ob1 = f1[keys_1[i]]["obs"]["sensor_data"]["base_camera"]["rgb"][start_idx_1:start_idx_1 + TRAJ_LEN]
        # # ob1 = np.transpose(ob1, (0, 3, 1, 2))
        # # obs_1.append(ob1)
        # obs_1.append(f1[keys_1[i]]["obs"][start_idx_1:start_idx_1 + TRAJ_LEN])
        # next_obs_1.append(f1[keys_1[i]]["obs"][start_idx_1 + 1:start_idx_1 + TRAJ_LEN + 1])
        # actions_1.append(f1[keys_1[i]]["actions"][start_idx_1:start_idx_1 + TRAJ_LEN])
        obs_1.extend(f1[keys_1[i]]["obs"][:-1])
        next_obs_1.extend(f1[keys_1[i]]["obs"][1:])
        actions_1.extend(f1[keys_1[i]]["actions"][:])
        
        i += 1
        num_traj += 1

    f1.close()

    obs_1 = np.array(obs_1)
    next_obs_1 = np.array(next_obs_1)
    actions_1 = np.array(actions_1)

    total_num_transitions = actions_1.shape[0]

    print("###### PHASE 2: processing language (reason & task) data ######")
    tokenizer = AutoTokenizer.from_pretrained(LANG_MODEL_NAME)
    lang_encoder = T5EncoderModel.from_pretrained(LANG_MODEL_NAME)
    lang_encoder.config.dropout_rate = 0.0
    lang_encoder.eval()

    for param in lang_encoder.parameters():
        param.requires_grad = False

    tokenized_tasks = tokenizer(
        [TASK_NLS[TASK]],
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
    task_embeddings = torch.repeat_interleave(task_embeddings, total_num_transitions, dim=0)
    task_embeddings = task_embeddings.cpu().numpy()

    print("task_embeddings shape:", task_embeddings.shape)

    print("###### PHASE 3: saving data ######")
    np.savez_compressed(
        f"data_offlinerl_{TASK}_{NUM_TRAJ_PER_TASK}_only_preferred_sac.npz", 
        obs=obs_1, 
        next_obs=next_obs_1, 
        action=actions_1,
        task_embeddings=task_embeddings,
    )


if __name__ == '__main__':
    main()
