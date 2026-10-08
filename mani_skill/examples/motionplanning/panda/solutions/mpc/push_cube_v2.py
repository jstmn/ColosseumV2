"""Successful PushCube-v2 motion plan.

Close the gripper, approach the cube from -x, then push it into the goal region.
"""

import numpy as np
import sapien

from mani_skill.envs.mpcm.push_cube_envs import PushCubeEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver


def solve(
    env: PushCubeEnv,
    seed=None,
    debug=False,
    vis=False,
    slow_down: bool = False,
    add_sinusoidal_noise: bool = False,
):
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        slow_down=slow_down,
        add_sinusoidal_noise=add_sinusoidal_noise,
    )
    env = env.unwrapped

    planner.close_gripper()
    reach_pose = sapien.Pose(p=env.obj.pose.sp.p + np.array([-0.1, 0, 0]), q=env.agent.tcp.pose.sp.q)
    res = planner.move_to_pose_with_screw(reach_pose)
    if res == -1:
        return res

    # Closed gripper meets the cube's back face, so the cube center leads the TCP by cube_half_size.
    goal_pose = sapien.Pose(
        p=env.goal_region.pose.sp.p + np.array([-env.cube_half_size, 0, 0]),
        q=env.agent.tcp.pose.sp.q,
    )
    res = planner.move_to_pose_with_screw(goal_pose)
    planner.close()
    return res
