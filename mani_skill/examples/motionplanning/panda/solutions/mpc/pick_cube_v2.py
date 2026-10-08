"""Successful PickCube-v2-wrist motion plan.

Top-down grasp, then lift the cube above ``goal_height``. No side grasps, pokes,
drops, or other non-task behaviors.
"""

import numpy as np
import sapien

from mani_skill.envs.mpcm.pick_cube_envs import PickCubeWristMountedEnv
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

FINGER_LENGTH = 0.025
CUBE_FINAL_Z_OFFSET = 0.05


def solve(
    env: PickCubeWristMountedEnv,
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

    obb = get_actor_obb(env.cube)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    grasp_pose = env.agent.build_grasp_pose(approaching, grasp_info["closing"], env.cube.pose.sp.p)

    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1:
        return res
    planner.close_gripper()

    goal_pose = sapien.Pose(
        p=np.array(
            [
                env.cube_spawn_center[0],
                env.cube_spawn_center[1],
                env.goal_height + CUBE_FINAL_Z_OFFSET,
            ]
        ),
        q=grasp_pose.q,
    )
    res = planner.move_to_pose_with_screw(goal_pose)
    planner.close()
    return res
