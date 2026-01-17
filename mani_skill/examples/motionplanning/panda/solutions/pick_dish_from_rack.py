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

    # Plate is vertical standing in the rack (rotated 90° around X)
    # Circular face is perpendicular to Y direction
    # Grasp from the TOP and lift straight up (simplest approach)

    # Approach from above (down in -Z direction)
    approaching = np.array([0, 0, -1])

    # Closing direction front-back (Y direction) to pinch across the plate's diameter
    # Since plate face is perpendicular to Y, we close in Y direction
    closing = np.array([0, 1, 0])

    # Position the grasp point at the TOP of the vertical plate
    center = plate_pos.copy()
    # Plate geometry parameters
    plate_outer_radius = env_sim._plate_outer_radius
    plate_inner_radius = env_sim._plate_inner_radius
    plate_rim_height = env_sim._plate_rim_height

    # Grasp at the top of the vertical plate
    center[1] = plate_pos[1] # Center Y
    center[2] = plate_pos[2] + plate_outer_radius * 0.8  # Near the top rim

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

    # Back away 10cm before approaching (move up in gripper's local Z frame)
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

    result = planner.move_to_pose_with_RRTConnect(grasp_pose)
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

    # Lift straight up out of the rack
    pullout_pose = grasp_pose * sapien.Pose([0, 0, -0.20])
    res = planner.move_to_pose_with_RRTConnect(pullout_pose)

    if debug:
        plate_after_pull = env_sim.plate.pose.p[0].cpu().numpy()
        print(f"✓ Pulled plate out to: {plate_after_pull}")


    if debug:
        print(f"\n=== STEP 4: MOVE AWAY FROM RACK ===")

    # Get current plate position after pullout
    current_plate_pos = env_sim.plate.pose.p[0].cpu().numpy()

    # Move plate toward center (Y=0) and away from rack
    # If plate is on left (Y>0), move right (negative Y direction)
    # If plate is on right (Y<0), move left (positive Y direction)
    target_y = 0.0  # Move toward center
    y_offset = target_y - current_plate_pos[1]

    # Build target pose in world coordinates
    target_pos = current_plate_pos.copy()
    target_pos[1] = target_y  # Move to center Y
    target_pos[0] = -0.2  # Move forward (toward robot)

    target_pose = env_sim.agent.build_grasp_pose(approaching, closing, target_pos)
    res = planner.move_to_pose_with_RRTConnect(target_pose)

    if debug:
        plate_after_move = env_sim.plate.pose.p[0].cpu().numpy()
        print(f"✓ Moved plate to: {plate_after_move}")

    # -------------------------------------------------------------------------- #
    # Wait for physics to settle
    # -------------------------------------------------------------------------- #
    arm_dof = len(env_sim.agent.arm_joint_names)
    home_qpos = env_sim.agent.robot.get_qpos()[0, :arm_dof].cpu().numpy()
    for _ in range(60):
        if env_sim.control_mode == "pd_joint_pos":
            action = np.hstack([home_qpos, planner.gripper_state])
        else:
            action = np.hstack([home_qpos, np.zeros(arm_dof), planner.gripper_state])
        obs, reward, terminated, truncated, info = env.step(action)
        if vis:
            env_sim.render_human()

    planner.close()
    return obs, reward, terminated, truncated, info
