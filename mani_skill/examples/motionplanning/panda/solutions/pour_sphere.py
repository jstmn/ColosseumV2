import numpy as np
import sapien

from mani_skill.envs.tasks.tabletop.pour_sphere import PourSphereEnv
from transforms3d.quaternions import axangle2quat, qmult

from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)


def solve(env: PourSphereEnv, seed=None, debug=False, vis=False):
    """Grasp cup1 from above (rim grasp), lift, move toward cup2, tilt to pour.

    All movements are computed relative to cup positions.
    """
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in ["pd_joint_pos", "pd_joint_pos_vel"]

    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    env_sim = env.unwrapped
    cup1_pos = env_sim.cup1.pose.p[0].cpu().numpy()
    cup2_pos = env_sim.cup2.pose.p[0].cpu().numpy()

    # Get cup dimensions
    cup_height = env_sim._cup_height
    cup_radius = env_sim._cup_radius

    # Compute distance between cups
    cup_distance = np.linalg.norm(cup2_pos[:2] - cup1_pos[:2])

    if debug:
        print(f"cup1_pos: {cup1_pos}")
        print(f"cup2_pos: {cup2_pos}")
        print(f"cup_distance: {cup_distance:.3f}m")

    # Get current TCP pose (robot starts in pregrasp position)
    current_tcp = env_sim.agent.tcp_pose.sp
    current_pos = np.array(current_tcp.p)
    current_quat = np.array(current_tcp.q)

    if debug:
        print(f"Initial TCP: pos={current_pos}, quat={current_quat}")

    # Open gripper for grasping
    planner.open_gripper()

    # Grasp at the rim of the cup - gripper grips rim edge from outside
    # The rim is at cup1_pos[2] + cup_height
    rim_height = cup1_pos[2] + cup_height
    grasp_height = rim_height  # At rim level so fingers grip rim edge from outside

    if debug:
        print(f"Cup rim height: {rim_height}")
        print(f"Grasp height: {grasp_height}")

    # -------------------------------------------------------------------------- #
    # Step 1: Move laterally to be above cup1 (keep current height and orientation)
    # -------------------------------------------------------------------------- #
    if debug:
        print(f"\n=== STEP 1: MOVE ABOVE CUP1 ===")

    above_cup1 = np.array([cup1_pos[0], cup1_pos[1], current_pos[2]])
    above_pose = sapien.Pose(above_cup1, current_quat)

    if debug:
        print(f"Target above cup1: {above_cup1}")

    result = planner.move_to_pose_with_screw(above_pose)
    if result == -1:
        if debug:
            print("Screw move above failed, trying RRT")
        result = planner.move_to_pose_with_RRTConnect(above_pose)
    if result == -1:
        if debug:
            print("Failed to move above cup1")
        planner.close()
        return result

    if debug:
        tcp = env_sim.agent.tcp_pose.sp.p
        print(f"Moved above cup1, TCP at: {tcp}")

    # -------------------------------------------------------------------------- #
    # Step 2: Descend to grasp position (grasp rim from above)
    # -------------------------------------------------------------------------- #
    if debug:
        print(f"\n=== STEP 2: DESCEND TO RIM ===")

    # Update current position and orientation
    current_tcp = env_sim.agent.tcp_pose.sp
    current_quat = np.array(current_tcp.q)

    grasp_pos = np.array([cup1_pos[0], cup1_pos[1], grasp_height])
    grasp_pose = sapien.Pose(grasp_pos, current_quat)

    if debug:
        print(f"Grasp position: {grasp_pos}")

    result = planner.move_to_pose_with_screw(grasp_pose)
    if result == -1:
        if debug:
            print("Screw descend failed")
        planner.close()
        return result

    if debug:
        tcp = env_sim.agent.tcp_pose.sp.p
        print(f"At grasp position, TCP at: {tcp}")

    # -------------------------------------------------------------------------- #
    # Step 3: Close gripper
    # -------------------------------------------------------------------------- #
    if debug:
        print(f"\n=== STEP 3: CLOSE GRIPPER ===")

    # Close gripper firmly to grasp cup rim
    planner.close_gripper(t=150, gripper_state=-1.0)

    # Let physics settle after grasping
    qpos = env_sim.agent.robot.get_qpos()[0, : len(planner.planner.joint_vel_limits)].cpu().numpy()
    for _ in range(30):
        if planner.control_mode == "pd_joint_pos":
            action = np.hstack([qpos, -1.0])
        else:
            action = np.hstack([qpos, qpos * 0, -1.0])
        env_sim.step(action)

    is_grasped = env_sim.agent.is_grasping(env_sim.cup1)[0].item()
    if debug:
        gripper_qpos = env_sim.agent.robot.get_qpos()[0, 7:9].cpu().numpy()
        print(f"Gripper qpos: {gripper_qpos}")
        print(f"Is grasped: {is_grasped}")

    # -------------------------------------------------------------------------- #
    # Step 4: Lift up
    # -------------------------------------------------------------------------- #
    if debug:
        print(f"\n=== STEP 4: LIFT UP ===")

    current_tcp = env_sim.agent.tcp_pose.sp
    current_pos = np.array(current_tcp.p)
    current_quat = np.array(current_tcp.q)

    # Lift height: modest height to stay within IK limits
    # Keep it lower for better IK success
    pour_height = cup2_pos[2] + cup_height + 0.08  # 8cm above cup2's top
    lift_pos = np.array([current_pos[0], current_pos[1], pour_height])
    lift_pose = sapien.Pose(lift_pos, current_quat)

    if debug:
        print(f"Lift target: {lift_pos}")

    result = planner.move_to_pose_with_screw(lift_pose)
    if result == -1:
        if debug:
            print("Screw lift failed, trying RRT")
        result = planner.move_to_pose_with_RRTConnect(lift_pose)
    if result == -1:
        if debug:
            print(f"Failed to lift")
        planner.close()
        return result

    if debug:
        sphere_pos = env_sim.sphere.pose.p[0].cpu().numpy()
        print(f"After lift: sphere at {sphere_pos}")

    # -------------------------------------------------------------------------- #
    # Step 5: Move toward cup2 position
    # -------------------------------------------------------------------------- #
    if debug:
        print(f"\n=== STEP 5: MOVE TOWARD CUP2 ===")

    current_tcp = env_sim.agent.tcp_pose.sp
    tcp_p = np.array(current_tcp.p)
    tcp_q = np.array(current_tcp.q)

    # Compute direction from cup1 to cup2
    cup1_to_cup2 = cup2_pos - cup1_pos
    cup1_to_cup2[2] = 0  # Only XY
    cup1_to_cup2 = cup1_to_cup2 / (np.linalg.norm(cup1_to_cup2) + 1e-6)

    # Move partway toward cup2 (move 50% of the distance)
    move_distance = cup_distance * 0.85
    target_xy = cup1_pos[:2] + cup1_to_cup2[:2] * move_distance
    move_pos = np.array([target_xy[0], target_xy[1], tcp_p[2] + 0.05])
    move_pose = sapien.Pose(move_pos, tcp_q)

    if debug:
        print(f"Moving {move_distance:.3f}m toward cup2")
        print(f"Target position: {move_pos}")

    result = planner.move_to_pose_with_screw(move_pose)
    if result == -1:
        if debug:
            print("Screw move failed, trying RRT")
        result = planner.move_to_pose_with_RRTConnect(move_pose)

    if debug:
        sphere_pos = env_sim.sphere.pose.p[0].cpu().numpy()
        print(f"After move: sphere at {sphere_pos}")

    # -------------------------------------------------------------------------- #
    # Step 5b: Rotate gripper 90 degrees counter-clockwise before pouring
    # -------------------------------------------------------------------------- #
    if debug:
        print(f"\n=== STEP 5b: ROTATE 90° CCW ===")

    current_tcp = env_sim.agent.tcp_pose.sp
    tcp_p = np.array(current_tcp.p)
    tcp_q = np.array(current_tcp.q)

    # Rotate 90 degrees counter-clockwise around Z axis
    z_rot_angle = np.pi / 2  # 90 degrees
    z_rot_quat = axangle2quat([0, 0, 1], z_rot_angle)
    rotated_quat = qmult(z_rot_quat, tcp_q)
    rotate_pose = sapien.Pose(tcp_p, rotated_quat)

    result = planner.move_to_pose_with_screw(rotate_pose)
    if result == -1:
        if debug:
            print("Screw rotation failed, trying RRT")
        result = planner.move_to_pose_with_RRTConnect(rotate_pose)

    if debug:
        sphere_pos = env_sim.sphere.pose.p[0].cpu().numpy()
        print(f"After rotation: sphere at {sphere_pos}")

    # -------------------------------------------------------------------------- #
    # Step 6: Tilt to pour toward cup2
    # -------------------------------------------------------------------------- #
    if debug:
        print(f"\n=== STEP 6: TILT TO POUR ===")

    current_tcp = env_sim.agent.tcp_pose.sp
    tcp_p = np.array(current_tcp.p)
    tcp_q = np.array(current_tcp.q)

    # Compute pour direction: from current position toward cup2
    pour_dir = cup2_pos - tcp_p
    pour_dir[2] = 0  # Only consider XY
    pour_dir = pour_dir / (np.linalg.norm(pour_dir) + 1e-6)

    # Compute rotation axis perpendicular to pour direction
    # Rotation axis is cross product of up (+Z) and pour direction
    rot_axis = np.cross(np.array([0, 0, 1]), pour_dir)
    rot_axis = rot_axis / (np.linalg.norm(rot_axis) + 1e-6)

    if debug:
        print(f"Pour direction: {pour_dir}")
        print(f"Rotation axis: {rot_axis}")

    # Tilt in steps to pour toward cup2
    total_tilt = 0
    for step in range(4):
        tilt_angle = 35 * np.pi / 180  # 35 degrees per step
        total_tilt += tilt_angle

        # World frame rotation
        tilt_quat = qmult(axangle2quat(rot_axis, tilt_angle), tcp_q)
        tilt_pose = sapien.Pose(p=tcp_p, q=tilt_quat)

        result = planner.move_to_pose_with_screw(tilt_pose)
        if result == -1:
            if debug:
                print(f"Tilt step {step+1} failed at {np.degrees(total_tilt):.0f}°")
            # Wait for sphere to settle
            planner.close_gripper(t=50)
            break

        # Update current pose for next iteration
        current_tcp = env_sim.agent.tcp_pose.sp
        tcp_p = np.array(current_tcp.p)
        tcp_q = np.array(current_tcp.q)

        sphere_pos = env_sim.sphere.pose.p[0].cpu().numpy()
        if debug:
            print(f"After tilt step {step+1} ({np.degrees(total_tilt):.0f}°): sphere at {sphere_pos}")

        # Check if sphere has fallen (Z near ground level)
        if sphere_pos[2] < cup2_pos[2] + 0.05:
            if debug:
                print(f"Sphere has landed")
            break

    # Hold to let sphere fall and settle
    res = planner.close_gripper(t=100)

    # Wait longer for sphere to settle on the table/in cup
    qpos = env_sim.agent.robot.get_qpos()[0, : len(planner.planner.joint_vel_limits)].cpu().numpy()
    for _ in range(200):
        if planner.control_mode == "pd_joint_pos":
            action = np.hstack([qpos, -1.0])
        else:
            action = np.hstack([qpos, qpos * 0, -1.0])
        obs, reward, terminated, truncated, info = env.step(action)
    res = (obs, reward, terminated, truncated, info)

    if debug:
        sphere_pos = env_sim.sphere.pose.p[0].cpu().numpy()
        print(f"After pour: sphere at {sphere_pos}")
        xy_dist = np.linalg.norm(sphere_pos[:2] - cup2_pos[:2])
        print(f"XY distance to cup2: {xy_dist:.4f}m (need < {cup_radius:.4f}m)")

    planner.close()
    return res
