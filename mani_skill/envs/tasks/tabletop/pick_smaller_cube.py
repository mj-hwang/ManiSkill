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


@register_env("PickSmallerCube-v1", max_episode_steps=50)
class PickSmallerCubeEnv(BaseEnv):
    """
    **Task Description:**
    The goal is to pick up a smaller cube out of two cubes (red & blue).
    - if the red_is_larger flag is set to True, the red cube will be larger than the blue cube 

    **Randomizations:**
    - both cubes have their z-axis rotation randomized
    - both cubes have their xy positions on top of the table scene randomized. The positions are sampled such that the cubes do not collide with each other
    - the target goal position (marked by a green sphere) of the cube has its xy position randomized in the region [0.1, 0.1] x [-0.1, -0.1] and z randomized in [0, 0.3]
    - if the red_is_larger flag is set to False, colors are randomized.

    **Success Conditions:**
    - the smaller cube position is within `goal_thresh` (default 0.025m) euclidean distance of the goal position
    - the robot is static (q velocity < 0.2)
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]
    goal_thresh = 0.03
    larger_cube_half_size = 0.025
    smaller_cube_half_size = 0.015

    def __init__(
        self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, red_is_larger=True, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.red_is_larger = red_is_larger
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
        colors = ([1, 0, 0, 1], [0, 0, 1, 1])
        if not self.red_is_larger:
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
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
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
            xy = torch.rand((b, 2)) * 0.2 - 0.1
            region = [[-0.1, -0.2], [0.1, 0.2]]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.003
            larger_cube_xy = xy + sampler.sample(radius, 100)
            smaller_cube_xy = xy + sampler.sample(radius, 100, verbose=False)

            xyz[:, :2] = larger_cube_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.larger_cube.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            xyz[:, :2] = smaller_cube_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.smaller_cube.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            goal_xyz[:, 2] = torch.rand((b)) * 0.05 + 0.25
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.smaller_cube.pose.p, axis=1)
            <= self.goal_thresh
        )
        is_grasped = self.agent.is_grasping(self.larger_cube)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    # def _get_obs_extra(self, info: Dict):
    #     obs = dict(
    #         tcp_pose=self.agent.tcp.pose.raw_pose,
    #         goal_pos=self.goal_site.pose.p,
    #     )
    #     return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.smaller_cube.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.smaller_cube.pose.p, axis=1
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

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8
