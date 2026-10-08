"""Successful PullCubeTool-v2 motion plan.

Grasp the L-hook, place it behind the cube, then pull the cube toward the robot.
"""

import numpy as np
import sapien

from mani_skill.envs.mpcm.pull_cube_tool_envs import PullCubeToolWristMountedEnv
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

TOOL_GRASP_DEPTH = 0.03
# Along the handle toward the corner of the L. Tool origin is the free end of the handle.
TOOL_GRASP_HANDLE_OFFSET = 0.04
# Handle center is 2.5 cm above the table. Shift the pinch toward the table.
TOOL_GRASP_Z_OFFSET = -0.02
LIFT_HEIGHT = 0.20
# Inner hook face sits this far past the cube's far +x face, before the pull toward the robot.
HOOK_PAST_CUBE = 0.08
HOOK_Y_OFFSET = -0.08
APPROACH_STANDOFF = 0.02


def _tool_grasp_pose(env: PullCubeToolWristMountedEnv) -> sapien.Pose:
    assert 0 < TOOL_GRASP_HANDLE_OFFSET < env.handle_length, (
        f"TOOL_GRASP_HANDLE_OFFSET={TOOL_GRASP_HANDLE_OFFSET} must be on the handle (0, {env.handle_length})"
    )
    tool_obb = get_actor_obb(env.l_shape_tool)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        tool_obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=TOOL_GRASP_DEPTH,
    )
    closing = grasp_info["closing"]
    # Wrist-cam j7 is −π/4 vs panda's +π/4, which can flip closing. Keep grasp +x along the handle.
    tool_T = env.l_shape_tool.pose.to_transformation_matrix()[0].cpu().numpy()
    tool_x = tool_T[:3, 0]
    if np.dot(np.cross(closing, approaching), tool_x) < 0:
        closing = -closing
    grasp_x = np.cross(closing, approaching)
    assert np.dot(grasp_x, tool_x) > 0, f"grasp +x is not along the handle: grasp_x={grasp_x}, tool_x={tool_x}"
    tool_p = env.l_shape_tool.pose.sp.p.copy()
    grasp_center = tool_p + tool_x * TOOL_GRASP_HANDLE_OFFSET
    grasp_center[2] = tool_p[2] + TOOL_GRASP_Z_OFFSET
    return env.agent.build_grasp_pose(approaching, closing, grasp_center)


def _lift_pose_halfway_to_approach(grasp_pose: sapien.Pose, approach_pose: sapien.Pose) -> sapien.Pose:
    grasp_p = np.asarray(grasp_pose.p, dtype=float)
    approach_p = np.asarray(approach_pose.p, dtype=float)
    assert grasp_p.shape == (3,), f"grasp_pose.p shape={grasp_p.shape}"
    assert approach_p.shape == (3,), f"approach_pose.p shape={approach_p.shape}"
    lift_p = 0.5 * (grasp_p + approach_p)
    lift_p[2] = approach_p[2]
    return sapien.Pose(p=lift_p, q=grasp_pose.q)


def solve(
    env: PullCubeToolWristMountedEnv,
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

    grasp_pose = _tool_grasp_pose(env)
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    cube_pos = env.cube.pose.sp.p
    tcp_to_hook_inner = env.handle_length - env.hook_length - TOOL_GRASP_HANDLE_OFFSET
    assert tcp_to_hook_inner > 0, (
        f"grasp is at or past the hook: TOOL_GRASP_HANDLE_OFFSET={TOOL_GRASP_HANDLE_OFFSET}, "
        f"handle_length={env.handle_length}, hook_length={env.hook_length}"
    )
    hook_x = env.cube_half_size + HOOK_PAST_CUBE - tcp_to_hook_inner
    approach_pose = sapien.Pose(cube_pos) * sapien.Pose([hook_x - APPROACH_STANDOFF, HOOK_Y_OFFSET, LIFT_HEIGHT - 0.05])
    approach_pose.set_q(grasp_pose.q)
    hook_pose = sapien.Pose(cube_pos) * sapien.Pose([hook_x, HOOK_Y_OFFSET, 0])
    hook_pose.set_q(grasp_pose.q)
    lift_pose = _lift_pose_halfway_to_approach(grasp_pose, approach_pose)

    res = planner.move_to_pose_with_screw(reach_pose)
    if res == -1:
        return res
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1:
        return res
    planner.close_gripper(t=9)

    res = planner.move_to_pose_with_screw(lift_pose)
    if res == -1:
        return res
    res = planner.move_to_pose_with_screw(approach_pose)
    if res == -1:
        return res
    res = planner.move_to_pose_with_screw(hook_pose)
    if res == -1:
        return res

    pull_pose = hook_pose * sapien.Pose([-0.35, 0, 0])
    res = planner.move_to_pose_with_screw(pull_pose)
    planner.close()
    return res
