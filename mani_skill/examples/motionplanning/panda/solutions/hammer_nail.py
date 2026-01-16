import numpy as np
import sapien
import sapien.render

from mani_skill.envs.tasks.tabletop.hammer_nail import HammerNailEnv
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)


def solve(env: HammerNailEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    env_sim = env.unwrapped
    nail_center = env_sim.nails[0].pose.p[0].cpu().numpy()
    hammer_pos = env_sim.hammer.pose.p[0].cpu().numpy()
    block_pos = env_sim.block.pose.p[0].cpu().numpy()

    # Always print starting positions
    print(f"Starting hammer position: {hammer_pos}")
    print(f"Starting nail position: {nail_center}")
    print(f"Starting block position: {block_pos}")

    # Grasp hammer from above near the head using its world AABB.
    approaching = np.array([0.0, 0.0, -1.0])
    render_comp = env_sim.hammer._objs[0].find_component_by_type(
        sapien.render.RenderBodyComponent
    )
    aabb = render_comp.compute_global_aabb_tight()
    center = (aabb[0] + aabb[1]) * 0.5
    extents = aabb[1] - aabb[0]
    length_axis = 0 if extents[0] >= extents[1] else 1
    head_sign = 1.0 if nail_center[length_axis] >= center[length_axis] else -1.0
    head_pos = center.copy()
    head_pos[length_axis] = (
        aabb[1, length_axis] if head_sign > 0 else aabb[0, length_axis]
    )
    grasp_point = center.copy()
    grasp_point[length_axis] = head_pos[length_axis] - head_sign * 0.15  # 15cm from head toward handle
    if length_axis == 0:
        closing = np.array([0.0, 1.0, 0.0])
    else:
        closing = np.array([1.0, 0.0, 0.0])
    grasp_pose = env_sim.agent.build_grasp_pose(approaching, closing, grasp_point)
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.12])
    # Grasp hammer
    planner.open_gripper()
    result = planner.move_to_pose_with_RRTConnect(reach_pose)
    if result == -1:
        result = planner.move_to_pose_with_screw(reach_pose)
        if result == -1:
            planner.close()
            return -1
    result = planner.move_to_pose_with_screw(grasp_pose)
    if result == -1:
        result = planner.move_to_pose_with_RRTConnect(grasp_pose)
        if result == -1:
            planner.close()
            return -1

    planner.close_gripper(t=60, gripper_state=-1.0)

    # Lift hammer to nail height
    lift_pose = sapien.Pose(
        [grasp_pose.p[0], grasp_pose.p[1], nail_center[2]],
        grasp_pose.q,
    )
    result = planner.move_to_pose_with_RRTConnect(lift_pose)
    if result == -1:
        result = planner.move_to_pose_with_screw(lift_pose)
        if result == -1:
            planner.close()
            return -1

    # Move to striking position to the left of the nail (in -Y direction)
    # Offset along world +Y to align the hammer head with the nail.
    head_from_grip = abs(head_pos[1] - grasp_point[1])
    ready_center = nail_center.copy()
    ready_center[0] = nail_center[0] - 0.14  # Same X as lifted pose
    ready_center[1] = nail_center[1] - head_from_grip + 0.2  # Position to left with clearance
    ready_center[2] = nail_center[2] + 0.005  # Same height as nail
    ready_pose = sapien.Pose(ready_center, grasp_pose.q)
    result = planner.move_to_pose_with_RRTConnect(ready_pose)
    if result == -1:
        planner.close()
        return -1

    # Strike horizontally on nail once to drive it in.
    for strike_num in range(1):
        # Move forward incrementally in smaller steps
        strike_center = ready_center.copy()
        strike_center[1] = nail_center[1] - head_from_grip + 0.05  # Move closer
        strike_center[2] = nail_center[2] + 0.015
        mid_strike_pose = sapien.Pose(strike_center, grasp_pose.q)
        result = planner.move_to_pose_with_RRTConnect(mid_strike_pose)
        if result == -1:
            planner.close()
            return -1

        # Pull back for next strike
        if strike_num < 0:  # Don't pull back on last strike
            result = planner.move_to_pose_with_RRTConnect(ready_pose)
            if result == -1:
                planner.close()
                return -1

    # Final pull back after all strikes
    result = planner.move_to_pose_with_RRTConnect(ready_pose)
    if result == -1:
        planner.close()
        return -1

    # Move to safe drop position
    drop_pose = lift_pose
    result = planner.move_to_pose_with_RRTConnect(drop_pose)
    if result == -1:
        planner.close()
        return -1

    # Release hammer
    planner.open_gripper()

    planner.close()
    return result
