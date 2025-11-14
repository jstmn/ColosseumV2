import numpy as np
import sapien

from mani_skill.envs.tasks.tabletop.pick_dish_from_rack import PickDishFromRackEnv
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)


def solve(env: PickDishFromRackEnv, seed=None, debug=False, vis=False):
    """Pick plate from rack and place it flat on the table."""
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode

    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    env_sim = env.unwrapped

    # Get plate position and orientation (plate is vertical in rack)
    plate_pose = env_sim.plate.pose
    plate_pos = plate_pose.p[0].cpu().numpy()

    if debug:
        print(f"\n=== PLATE POSITION IN RACK ===")
        print(f"Plate position: {plate_pos}")

    # Plate is vertical standing on its edge next to the rack
    # Normal points in -Y direction
    # Approach from top-down to grasp the top rim

    # Approach from above
    approaching = np.array([0, 0, -1])

    # Closing direction left-right (X direction) to pinch across the rim
    closing = np.array([1, 0, 0])

    # Position the grasp point on the TOP of the vertical plate
    center = plate_pos.copy()
    # Plate geometry parameters
    plate_outer_radius = env_sim._plate_outer_radius
    plate_inner_radius = env_sim._plate_inner_radius
    plate_rim_height = env_sim._plate_rim_height

    # Grasp the top rim - the plate is standing on edge, so top is at plate_pos[2] + rim_height/2
    rim_grasp_radius = (plate_outer_radius + plate_inner_radius) / 2.0
    center[2] = plate_pos[2] + plate_rim_height * 0.3  # Slightly below top of rim

    # Build grasp pose
    grasp_pose = env_sim.agent.build_grasp_pose(approaching, closing, center)

    if debug:
        print(f"\n=== GRASP INFO ===")
        print(f"Plate center: {plate_pos}")
        print(f"Grasp center: {center}")
        print(f"Approaching: {approaching}")
        print(f"Closing: {closing}")

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    if debug:
        print(f"\n=== STEP 1: REACH ===")

    # Back away 10cm before approaching
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.10])
    result = planner.move_to_pose_with_RRTConnect(reach_pose)
    if result == -1:
        if debug:
            print("❌ Failed to reach")
        planner.close()
        return result

    if debug:
        print("✓ Reached approach position")

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    if debug:
        print(f"\n=== STEP 2: GRASP ===")

    result = planner.move_to_pose_with_screw(grasp_pose)
    if result == -1:
        if debug:
            print("❌ Failed to grasp position")
        planner.close()
        return result

    # Close gripper with maximum force
    planner.close_gripper(t=40, gripper_state=-1.0)

    # Let physics settle after grasping
    qpos = env_sim.agent.robot.get_qpos()[0, : len(planner.planner.joint_vel_limits)].cpu().numpy()
    for i in range(20):
        if planner.control_mode == "pd_joint_pos":
            action = np.hstack([qpos, -1.0])
        else:
            action = np.hstack([qpos, qpos * 0, -1.0])
        env_sim.step(action)

    is_grasped = env_sim.agent.is_grasping(env_sim.plate)[0].item()

    if debug:
        gripper_qpos = env_sim.agent.robot.get_qpos()[0, 7:9].cpu().numpy()
        print(f"Gripper closed: {gripper_qpos}")
        print(f"Is grasped: {is_grasped}")

    # -------------------------------------------------------------------------- #
    # Pull out of rack
    # -------------------------------------------------------------------------- #
    if debug:
        print(f"\n=== STEP 3: PULL OUT ===")

    # Pull forward (positive Y) and up
    pullout_pose = grasp_pose * sapien.Pose([0, 0, -0.15])
    res = planner.move_to_pose_with_screw(pullout_pose)

    if debug:
        plate_after_pull = env_sim.plate.pose.p[0].cpu().numpy()
        print(f"✓ Pulled plate out to: {plate_after_pull}")

    # -------------------------------------------------------------------------- #
    # Rotate to horizontal
    # -------------------------------------------------------------------------- #
    if debug:
        print(f"\n=== STEP 4: ROTATE TO HORIZONTAL ===")

    # Rotate -90 degrees around Y to make plate horizontal
    rotation = sapien.Pose(q=[0.7071068, 0, -0.7071068, 0])  # -90 deg around Y
    horizontal_pose = pullout_pose * rotation

    res = planner.move_to_pose_with_screw(horizontal_pose)

    if debug:
        plate_horizontal = env_sim.plate.pose.p[0].cpu().numpy()
        print(f"✓ Rotated to horizontal: {plate_horizontal}")

    # -------------------------------------------------------------------------- #
    # Move to goal position
    # -------------------------------------------------------------------------- #
    if debug:
        print(f"\n=== STEP 5: MOVE TO GOAL ===")

    # Get goal position
    goal_pos = env_sim._plate_goal_position.copy()

    # Calculate proper height for plate to rest on table
    table_p_arr = np.asarray(env_sim.table_scene.table.pose.p).ravel()
    table_z = float(table_p_arr[-1])
    table_top_z = table_z + float(env_sim.table_scene.table_height)
    goal_pos[2] = table_top_z + 0.15  # 15cm above table for approach

    # Move to goal with horizontal orientation
    goal_pose = sapien.Pose(p=goal_pos, q=horizontal_pose.q)

    res = planner.move_to_pose_with_RRTConnect(goal_pose)
    if res == -1:
        if debug:
            print("❌ Failed to move to goal")
        planner.close()
        return res

    if debug:
        plate_at_goal = env_sim.plate.pose.p[0].cpu().numpy()
        print(f"✓ Moved to goal: {plate_at_goal}")

    # -------------------------------------------------------------------------- #
    # Lower to table
    # -------------------------------------------------------------------------- #
    if debug:
        print(f"\n=== STEP 6: LOWER TO TABLE ===")

    # Lower to just above table surface
    lower_pose = goal_pose * sapien.Pose([0, 0, -0.12])  # Lower 12cm

    res = planner.move_to_pose_with_screw(lower_pose)

    if debug:
        plate_lowered = env_sim.plate.pose.p[0].cpu().numpy()
        print(f"✓ Lowered plate: {plate_lowered}")

    # Settle before release
    qpos = env_sim.agent.robot.get_qpos()[0, : len(planner.planner.joint_vel_limits)].cpu().numpy()
    for i in range(30):
        if planner.control_mode == "pd_joint_pos":
            action = np.hstack([qpos, -1.0])
        else:
            action = np.hstack([qpos, qpos * 0, -1.0])
        env_sim.step(action)

    # -------------------------------------------------------------------------- #
    # Release plate
    # -------------------------------------------------------------------------- #
    if debug:
        print(f"\n=== STEP 7: RELEASE ===")

    planner.open_gripper()

    if debug:
        final_plate_pos = env_sim.plate.pose.p[0].cpu().numpy()
        print(f"✓ Released plate at: {final_plate_pos}")

    # -------------------------------------------------------------------------- #
    # Move back to safe position
    # -------------------------------------------------------------------------- #
    if debug:
        print(f"\n=== STEP 8: RETURN TO SAFE POSITION ===")

    # Move up and away
    retreat_pose = lower_pose * sapien.Pose([0, 0, 0.15])
    res = planner.move_to_pose_with_RRTConnect(retreat_pose)

    if debug:
        if res == -1:
            print("❌ Failed to retreat")
        else:
            print("✓ Moved to safe retreat position")
        print(f"✓ Task complete!")

    planner.close()
    return res
