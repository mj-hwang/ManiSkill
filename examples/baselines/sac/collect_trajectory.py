#!/usr/bin/env python3
"""
Minimal trajectory collector for ManiSkill using a pretrained SAC policy.

- Loads an Actor network (same structure as the original SAC training script).
- Wraps the env with RecordEpisode to save trajectories to HDF5 (and optional MP4).
- No training, replay buffers, tensorboard, or extras.

Example:
    python collect_trajectory.py \
        --env-id PickCube-v1 \
        --checkpoint runs/PickCube-v1/best_ckpt.pt \
        --record-dir ./datasets \
        --num-traj 50 \
        --traj-name sac_rollouts

The script assumes your checkpoint contains either the whole state_dict for the actor,
or a dict with key "actor".
"""

import argparse
import os
import os.path as osp
import time
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from mani_skill.utils.wrappers.record import RecordEpisode

# ----------------------------
# Policy
# ----------------------------

# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        return self.net(x)


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        h, l = env.single_action_space.high, env.single_action_space.low
        self.register_buffer("action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32))
        # will be saved in the state_dict

    def forward(self, x):
        x = self.backbone(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_eval_action(self, x):
        x = self.backbone(x)
        mean = self.fc_mean(x)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)

# ----------------------------
# Utilities
# ----------------------------

def _as_float_np(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)

def load_actor(checkpoint: str, env, device: torch.device) -> Actor:
    actor = Actor(env).to(device)
    ckpt = torch.load(checkpoint)
    actor.load_state_dict(ckpt['actor'])
    return actor

def parse_args():
    parser = argparse.ArgumentParser("Collect trajectories from a pretrained SAC policy")
    # Environment
    parser.add_argument("--env-id", type=str, required=True, help="ManiSkill environment ID (e.g., PickCube-v1)")
    parser.add_argument("--obs-mode", type=str, default="state", help="Observation mode; use 'state' for this script")
    parser.add_argument("--control-mode", type=str, default=None, help="Optional control mode override (e.g., pd_joint_delta_pos)")
    parser.add_argument("--seed", type=int, default=0, help="Starting seed for env.reset")
    parser.add_argument("--cuda", type=bool, default=True, help="Use CUDA if available")

    # Policy
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pretrained checkpoint (.pt)")

    # Recording
    parser.add_argument("--record-dir", type=str, default="datasets", help="Directory to store trajectories")
    parser.add_argument("--traj-name", type=str, default=None, help="Name prefix for trajectories (default: timestamp)")
    parser.add_argument("--num-traj", type=int, default=100, help="Number of trajectories to save")
    parser.add_argument("--save-video", action="store_true", help="Also save MP4 video for each trajectory")
    parser.add_argument("--only-count-success", action="store_true",
                        help="If set, only count successful episodes toward --num-traj (unsuccessful episodes are not saved)")
    parser.add_argument("--video-fps", type=int, default=30)

    return parser.parse_args()

def make_env(env_id: str, obs_mode: str, control_mode: Optional[str]):
    # Minimal non-vector env. We rely on RecordEpisode for recording.
    env_kwargs = dict(obs_mode=obs_mode, render_mode="rgb_array", sim_backend="gpu")
    if control_mode is not None:
        env_kwargs["control_mode"] = control_mode
    env = gym.make(env_id, **env_kwargs)

    # Flatten action dict spaces if any (e.g., for some ManiSkill envs)
    if isinstance(env.action_space, gym.spaces.Dict):
        from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
        env = FlattenActionSpaceWrapper(env)

    # For this minimal collector we require vector observations
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise RuntimeError("This script expects obs_mode='state' resulting in a Box observation. "
                           "Got: {}".format(type(env.observation_space)))

    return env

def main():
    args = parse_args()
    os.makedirs(args.record_dir, exist_ok=True)

    # Create base env (not wrapped) for sizing the policy correctly
    base_env = make_env(args.env_id, args.obs_mode, args.control_mode)

    # Wrap with RecordEpisode for logging/trajectory saving
    traj_name = args.traj_name or time.strftime("%Y%m%d_%H%M%S")
    output_dir = osp.join(args.record_dir, args.env_id, "sac_policy")
    env = RecordEpisode(
        base_env,
        output_dir=output_dir,
        trajectory_name=traj_name,
        source_type="sac_policy",
        source_desc="pretrained SAC policy rollout",
        video_fps=args.video_fps,
        record_reward=True,
        save_on_reset=False,
    )
    output_h5_path = env._h5_file.filename  # future files follow same prefix

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    actor = load_actor(args.checkpoint, env, device=device)

    seed = int(args.seed)
    saved = 0
    pbar = tqdm(total=args.num_traj, desc="Collecting trajectories")

    while saved < args.num_traj:
        obs, _ = env.reset(seed=seed)
        seed += 1
        done = False
        success = False
        latest_info = {}
        i = 0

        while not done:
            # obs_tensor = torch.from_numpy(_as_float_np(obs)).to(device).unsqueeze(0)
            # with torch.no_grad():
            #     action = actor.get_eval_action(obs).cpu().numpy().reshape(env.action_space.shape)
            # action = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
            # obs, reward, terminated, truncated, info = env.step(action)
            with torch.no_grad():
                obs, reward, terminated, truncated, info = env.step(actor.get_eval_action(obs))
            # breakpoint()
            done = bool(terminated or truncated)
            latest_info = info
            i += 1
        
        print(i)
        i = 0

        # Determine success flag if provided by env
        success = bool(latest_info.get("success", False))

        if args.only_count_success and not success:
            # Discard trajectory (and video) and do not progress the bar
            env.flush_trajectory(save=False)
            if args.save_video:
                env.flush_video(save=False)
            continue

        # Save episode
        env.flush_trajectory()
        if args.save_video:
            env.flush_video()
        saved += 1
        pbar.update(1)
        pbar.set_postfix(dict(success=success))

    env.close()
    print(f"Example HDF5 path prefix: {output_h5_path}")
    print(f"Done. Saved {saved} trajectories to: {output_dir}")

if __name__ == "__main__":
    main()
