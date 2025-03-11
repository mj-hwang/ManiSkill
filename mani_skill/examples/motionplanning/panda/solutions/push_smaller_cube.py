import numpy as np
import sapien

from mani_skill.envs.tasks import PushSmallerCubeEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver

def solve(env: PushSmallerCubeEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    FINGER_LENGTH = 0.025
    env = env.unwrapped
    planner.close_gripper()
    p_reach = env.smaller_cube.pose.sp.p + np.array([-0.045, 0, 0])
    reach_pose = sapien.Pose(p=p_reach, q=env.agent.tcp.pose.sp.q)
    planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    p_goal = env.goal_site.pose.sp.p + np.array([-0.035, 0, 0])
    p_goal[..., 1] = p_reach[..., 1]
    goal_pose = sapien.Pose(p=p_goal,q=env.agent.tcp.pose.sp.q)
    res = planner.move_to_pose_with_screw(goal_pose)

    planner.close()
    return res
