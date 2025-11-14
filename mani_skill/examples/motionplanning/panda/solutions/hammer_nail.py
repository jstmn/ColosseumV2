import numpy as np
import sapien

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
    hammer_center = env_sim.hammer.pose.p[0].cpu().numpy()
    nail_center = env_sim.nails[0].pose.p[0].cpu().numpy()
    head_offset = env_sim._hammer_head_offset
    com_offset = env_sim._hammer_com_offset.cpu().numpy()

    # Grasp hammer from above - after 270° rotation:
    # Head is at +X (right side), handle extends in -X (left side)
    approaching = np.array([0.0, 0.0, -1.0])
    closing = np.array([0.0, 1.0, 0.0])  # Close gripper along Y axis

    # Grip at center of mass for stable grasp
    grasp_point = hammer_center + com_offset  # COM is at ~[0.093, 0, 0] relative to center
    grasp_pose = env_sim.agent.build_grasp_pose(approaching, closing, grasp_point)
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.12])

    # Grasp hammer
    planner.open_gripper()
    result = planner.move_to_pose_with_RRTConnect(reach_pose)
    if result == -1:
        planner.close()
        return -1

    result = planner.move_to_pose_with_screw(grasp_pose)
    if result == -1:
        planner.close()
        return -1

    planner.close_gripper()

    # Lift hammer slightly
    lift_pose = grasp_pose * sapien.Pose([0, 0, 0.10])
    result = planner.move_to_pose_with_RRTConnect(lift_pose)
    if result == -1:
        planner.close()
        return -1

    # Move to striking position to the left of the nail (in -X direction)
    # After 270° rotation: head is at +X from where we grip (COM)
    # Head is (head_offset - com_offset) distance from grip point in +X direction
    head_from_grip = head_offset - com_offset[0]
    ready_center = nail_center.copy()
    ready_center[0] = nail_center[0] - head_from_grip - 0.15  # Position to left with clearance
    ready_center[2] = nail_center[2]  # Same height as nail
    ready_pose = sapien.Pose(ready_center, grasp_pose.q)
    result = planner.move_to_pose_with_RRTConnect(ready_pose)
    if result == -1:
        planner.close()
        return -1

    # Strike horizontally on nail multiple times to drive it deeper (in +X direction)
    # Use controlled strikes to prevent penetration
    for strike_num in range(5):
        # Move forward incrementally in smaller steps
        strike_center = ready_center.copy()
        strike_center[0] = nail_center[0] - head_from_grip - 0.05  # Move closer
        mid_strike_pose = sapien.Pose(strike_center, grasp_pose.q)
        result = planner.move_to_pose_with_RRTConnect(mid_strike_pose)
        if result == -1:
            planner.close()
            return -1

        # Final strike position - slower motion with screw, strike through the nail position
        strike_center[0] = nail_center[0] - head_from_grip + 0.02  # Strike past nail back
        strike_pose = sapien.Pose(strike_center, grasp_pose.q)
        result = planner.move_to_pose_with_screw(strike_pose)
        if result == -1:
            planner.close()
            return -1

        # Pull back for next strike
        if strike_num < 4:  # Don't pull back on last strike
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
