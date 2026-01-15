import numpy as np
import sapien

from mani_skill.envs.tasks.tabletop.place_dish_in_rack import PlaceDishInRackEnv
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)


def solve(env: PlaceDishInRackEnv, seed=None, debug=False, vis=False):
    """Grasp flat plate from the rim and lift it up."""
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

    def move_or_abort(target_pose, use_rrt_first=False):
        # Choose planner order based on context
        if use_rrt_first:
            res = planner.move_to_pose_with_RRTConnect(target_pose)
            if res == -1:
                res = planner.move_to_pose_with_screw(target_pose)
        else:
            res = planner.move_to_pose_with_screw(target_pose)
            if res == -1:
                res = planner.move_to_pose_with_RRTConnect(target_pose)
        if res == -1:
            planner.close()
        return res

    # Get plate and rack positions
    plate_pose = env_sim.plate.pose
    plate_pos = plate_pose.p[0].cpu().numpy()
    rack_pos_initial = env_sim.dish_rack.pose.p[0].cpu().numpy()

    # Always print starting positions
    print(f"Starting plate position: {plate_pos}")
    print(f"Starting rack position: {rack_pos_initial}")

    if debug:
        print(f"\n=== PLATE POSITION ===")
        print(f"Plate position: {plate_pos}")

    # Plate is flat on table (outer radius=0.085m, rim height=0.018m)
    plate_outer_radius = env_sim._plate_outer_radius
    plate_inner_radius = env_sim._plate_inner_radius
    plate_rim_height = env_sim._plate_rim_height
    plate_base_thickness = env_sim._plate_base_thickness

    # Approach from top-down to grasp the rim
    approaching = np.array([0, 0, -1])  # Approach from above

    # Closing direction horizontal to pinch opposite sides of the rim
    closing = np.array([1, 0, 0])

    # Position the grasp point on the RIGHT SIDE of the plate
    # Approach from top-down, gripper fingers will pinch left-right across the rim
    center = plate_pos.copy()

    # Offset to the RIGHT side (positive X direction) - position over the rim
    # rim_grasp_radius = (plate_outer_radius + plate_inner_radius) / 2.0
    rim_grasp_radius = 1.25*plate_outer_radius # ( + plate_inner_radius) / 2.0
    
    center[0] = plate_pos[0] + rim_grasp_radius - 0.03 # Move to right side in X direction

    # Height should be lower on the rim for a more secure grip
    # center[2] = plate_pos[2] + plate_base_thickness + plate_rim_height * 0.3  # Low-mid on rim
    center[2] = plate_pos[2] + plate_base_thickness + plate_rim_height/2 - 0.01  # Low-mid on rim


    # Build grasp pose
    grasp_pose = env_sim.agent.build_grasp_pose(approaching, closing, center)

    if debug:
        print(f"\n=== GRASP INFO ===")
        print(f"Plate center: {plate_pos}")
        print(f"Grasp center (at rim): {center}")
        print(f"Plate outer radius: {plate_outer_radius}")
        print(f"Approaching: {approaching}")
        print(f"Closing: {closing}")

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    if debug:
        print(f"\n=== STEP 1: REACH ===")

    # Back away 6cm before approaching for better motion planning success
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.065])
    result = move_or_abort(reach_pose, use_rrt_first=True)
    if result == -1:
        if debug:
            print("❌ Failed to reach")
        return result

    if debug:
        print("✓ Reached approach position")

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    if debug:
        print(f"\n=== STEP 2: GRASP ===")

    result = move_or_abort(grasp_pose, use_rrt_first=True)
    if result == -1:
        if debug:
            print("❌ Failed to grasp position")
        return result

    # Close gripper with maximum force and longer duration to prevent slipping
    planner.close_gripper(t=25, gripper_state=-1.0)  # Close for 25 steps with full force

    # Let physics settle after grasping to ensure firm grip
    qpos = env_sim.agent.robot.get_qpos()[0, : len(planner.planner.joint_vel_limits)].cpu().numpy()
    for i in range(10):  # Hold position longer
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

    # Abort early if grasp failed
    if not is_grasped:
        if debug:
            print("❌ Grasp failed - aborting")
        planner.close()
        return -1

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    if debug:
        print(f"\n=== STEP 3: LIFT ===")

    # Smaller lift height to reduce motion magnitude
    lift_pose = sapien.Pose([0, 0, 0.06]) * grasp_pose
    res = move_or_abort(lift_pose)

    if debug:
        plate_after_lift = env_sim.plate.pose.p[0].cpu().numpy()
        print(f"✓ Lifted plate to: {plate_after_lift}")

    # -------------------------------------------------------------------------- #
    # Move to rack FIRST (keep plate horizontal for stability during transport)
    # -------------------------------------------------------------------------- #
    if debug:
        print(f"\n=== STEP 4: MOVE TO RACK (HORIZONTAL) ===")

    # Get rack position and dimensions
    rack_pos = env_sim.dish_rack.pose.p[0].cpu().numpy()
    rack_height = env_sim._rack_extent[2]  # 0.085m

    # Target the center of slot 2 (middle slot)
    # Slot dividers in rack local Y: [-0.105, -0.047, 0.015, 0.075]
    # Slot 2 center is at Y = (-0.047 + 0.015) / 2 = -0.016 in local coords
    # Compensate for grasp offset: we grasp at rim, ~8cm from plate center in X
    grasp_offset_x = rim_grasp_radius - 0.03  # ~0.0825m offset from plate center
    target_pos = rack_pos.copy()
    target_pos[0] += grasp_offset_x  # Compensate so plate center lands at rack center
    target_pos[1] -= 0.02  # Target slot 2 center in Y
    target_pos[2] += rack_height + 0.1  # clearance above rack

    # Keep horizontal orientation (same as lift pose)
    transport_pose = sapien.Pose(p=target_pos, q=lift_pose.q)

    # Use RRT for transport as it handles large motions better
    res = move_or_abort(transport_pose, use_rrt_first=True)
    if res == -1:
        if debug:
            print("❌ Failed to move to rack")
        return res

    if debug:
        plate_at_rack = env_sim.plate.pose.p[0].cpu().numpy()
        print(f"✓ Moved to rack center (horizontal): {target_pos}")
        print(f"  Plate is at: {plate_at_rack}")

    # Check if plate is still grasped after transport
    still_grasped = env_sim.agent.is_grasping(env_sim.plate)[0].item()
    if not still_grasped:
        if debug:
            print("❌ Plate slipped during transport - aborting")
        planner.close()
        return -1

    # -------------------------------------------------------------------------- #
    # Rotate to vertical above rack center
    # -------------------------------------------------------------------------- #
    if debug:
        print(f"\n=== STEP 5: ROTATE TO VERTICAL ABOVE RACK ===")

    # Rotate 90 degrees to make plate vertical
    rotation = sapien.Pose(q=[0.7071068, 0, 0.7071068, 0])  # 90 deg around Y
    vertical_pose = transport_pose * rotation

    res = move_or_abort(vertical_pose)
    if res == -1:
        return res

    if debug:
        plate_vertical = env_sim.plate.pose.p[0].cpu().numpy()
        print(f"✓ Rotated to vertical: {plate_vertical}")



    # -------------------------------------------------------------------------- #
    # Lower plate into slot
    # -------------------------------------------------------------------------- #
    if debug:
        print(f"\n=== STEP 6: LOWER INTO SLOT ===")

    # Lower plate straight down into the slot (no horizontal motion)
    lower_pos = vertical_pose.p.copy()
    lower_pos[1] -= 0.02  # Adjust Y slightly to center in slot
    lower_pos[2] -= 0.07  # Lower into slot
    lower_pose = sapien.Pose(lower_pos, vertical_pose.q)
    res = move_or_abort(lower_pose)

    if debug:
        plate_lowered = env_sim.plate.pose.p[0].cpu().numpy()
        print(f"✓ Lowered plate to: {plate_lowered}")

    # -------------------------------------------------------------------------- #
    # Release plate
    # -------------------------------------------------------------------------- #
    if debug:
        print(f"\n=== STEP 7: RELEASE ===")

    # Hold position briefly before release to let physics settle
    qpos = env_sim.agent.robot.get_qpos()[0, : len(planner.planner.joint_vel_limits)].cpu().numpy()
    for _ in range(5):
        if planner.control_mode == "pd_joint_pos":
            action = np.hstack([qpos, -1.0])
        else:
            action = np.hstack([qpos, qpos * 0, -1.0])
        env_sim.step(action)

    planner.open_gripper()

    # Let plate settle after release
    for _ in range(20):
        if planner.control_mode == "pd_joint_pos":
            action = np.hstack([qpos, 1.0])  # Keep gripper open
        else:
            action = np.hstack([qpos, qpos * 0, 1.0])
        env_sim.step(action)

    if debug:
        final_plate_pos = env_sim.plate.pose.p[0].cpu().numpy()
        print(f"✓ Released plate at: {final_plate_pos}")

    # -------------------------------------------------------------------------- #
    # Raise EE after placing
    # -------------------------------------------------------------------------- #
    if debug:
        print(f"\n=== STEP 8: RAISE EE ===")

    # Raise end effector up after releasing the plate
    raise_pos = lower_pose.p.copy()
    raise_pos[1] -= 0.1  # Back away slightly in Y
    raise_pos[2] += 0.02  # Raise 2cm
    raise_pose = sapien.Pose(raise_pos, lower_pose.q)
    res = move_or_abort(raise_pose)

    raise_pos[2] += 0.12  # Raise 12cm to clear the rack
    raise_pose = sapien.Pose(raise_pos, lower_pose.q)
    res = move_or_abort(raise_pose)

    if debug:
        print(f"✓ Raised EE to: {raise_pos}")

    planner.close()
    return res
