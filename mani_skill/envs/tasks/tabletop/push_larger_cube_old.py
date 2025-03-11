from typing import Any, Dict, Union

import numpy as np
import sapien
import torch
import random

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("PushLargerCube-v1", max_episode_steps=50)
class PushLargerCubeEnv(BaseEnv):
    """
    **Task Description:**
    The goal is to push a larger cube out of two cubes (red & blue) over the green line.
    - if the blue_is_larger flag is set to True, the blue cube will be larger than the red cube 

    **Randomizations:**
    - both cubes have their xy positions on top of the table scene randomized. The positions are sampled such that the cubes do not collide with each other
    - if the blue_is_larger flag is set to True, colors are randomized.

    **Success Conditions:**
    - the cube's x position is over 0.2
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]
    x_pushed_thresh = 0.15
    larger_cube_half_size = 0.025
    smaller_cube_half_size = 0.015

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, blue_is_larger=True, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.blue_is_larger = blue_is_larger
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        colors = ([0, 0, 1, 1], [1, 0, 0, 1])
        if not self.blue_is_larger:
            random.shuffle(colors)
        larger_cube_color, smaller_cube_color = colors
            
        self.larger_cube = actors.build_cube(
            self.scene,
            half_size=self.larger_cube_half_size,
            color=larger_cube_color,
            name="larger_cube",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )
        self.smaller_cube = actors.build_cube(
            self.scene,
            half_size=self.smaller_cube_half_size,
            color=smaller_cube_color,
            name="smaller_cube",
            initial_pose=sapien.Pose(p=[1, 0, 0.1]),
        )
        self.goal_site = actors.build_box(
            self.scene,
            half_sizes=[0.005, 1, 0.005],
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        # self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02
            # xy = torch.rand((b, 2)) * 0.2 - 0.1
            xy = torch.rand((b, 2))
            xy[:, 0] = xy[:, 0] * 0.2 - 0.1  # Scale x to [-0.1, 0.1]
            xy[:, 1] = xy[:, 1] * (-0.1)     # Scale y to [-0.1, 0.0]
            
            region = [[-0.015, -0.2], [0.015, 0.2]]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001
            larger_cube_xy = xy + sampler.sample(radius, 100)
            smaller_cube_xy = xy + sampler.sample(radius, 100, verbose=False)
            q = [1, 0, 0, 0]

            # make sure cubes do not 

            xyz[:, :2] = larger_cube_xy
            self.larger_cube.set_pose(Pose.create_from_pq(p=xyz.clone(), q=q))

            xyz[:, :2] = smaller_cube_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.smaller_cube.set_pose(Pose.create_from_pq(p=xyz, q=q))

            # set a little bit above 0 so the target is sitting on the table
            target_region_xyz = torch.tensor([self.x_pushed_thresh, 0, 0.005])
            self.goal_site.set_pose(
                Pose.create_from_pq(
                    p=target_region_xyz,
                    q=q,
                )
            )

    def evaluate(self):
        is_obj_pushed = self.larger_cube.pose.p[..., 0] >= self.x_pushed_thresh

        return {
            "success": is_obj_pushed,
        }

    # def _get_obs_extra(self, info: Dict):
    #     obs = dict(
    #         tcp_pose=self.agent.tcp.pose.raw_pose,
    #     )
    #     return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.larger_cube.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.larger_cube.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        qvel_without_gripper = self.agent.robot.get_qvel()
        if self.robot_uids == "xarm6_robotiq":
            qvel_without_gripper = qvel_without_gripper[..., :-6]
        elif self.robot_uids == "panda":
            qvel_without_gripper = qvel_without_gripper[..., :-2]
        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(qvel_without_gripper, axis=1)
        )
        reward += static_reward * info["is_obj_placed"]

        reward[info["success"]] = 5
        return reward
    
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # We also create a pose marking where the robot should push the cube from that is easiest (pushing from behind the cube)
        tcp_push_pose = Pose.create_from_pq(
            p=self.larger_cube.pose.p
            + torch.tensor([-self.larger_cube_half_size - 0.005, 0, 0], device=self.device)
        )
        tcp_to_push_pose = tcp_push_pose.p - self.agent.tcp.pose.p
        tcp_to_push_pose_dist = torch.linalg.norm(tcp_to_push_pose, axis=1)
        reaching_reward = 1 - torch.tanh(5 * tcp_to_push_pose_dist)
        reward = reaching_reward

        # compute a placement reward to encourage robot to move the cube to the center of the goal region
        # we further multiply the place_reward by a mask reached so we only add the place reward if the robot has reached the desired push pose
        # This reward design helps train RL agents faster by staging the reward out.
        reached = tcp_to_push_pose_dist < 0.01
        obj_to_goal_x_dist = self.x_pushed_thresh - self.larger_cube.pose.p[..., 0]
        push_reward = 1 - torch.tanh(5 * obj_to_goal_x_dist)
        reward += push_reward * reached

        # assign rewards to parallel environments that achieved success to the maximum of 3.
        reward[info["success"]] = 3
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 3.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
