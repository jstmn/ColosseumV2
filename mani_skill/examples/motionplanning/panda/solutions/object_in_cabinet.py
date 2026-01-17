import os
import numpy as np
import sapien

from mani_skill.envs.tasks.tabletop.object_in_cabinet import ObjectInCabinetEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

from mani_skill.examples.motionplanning.panda.solutions.open_cabinet import (
    _get_joint_limits,
)


def solve(env: ObjectInCabinetEnv, seed=None, debug=False, vis=False):
    """
    Solution for ObjectInCabinet task.
    Phase 1: Open cabinet door
    Phase 2: Move arm back and over door to hover above apple
    """
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode

    env_sim = env.unwrapped

    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env_sim.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    qmin, qmax = _get_joint_limits(env_sim.handle_link.joint)

    def check_door():
        qpos = env_sim.handle_link.joint.qpos[0].item()
        pct = (qpos - qmin) / (qmax - qmin) * 100
        return pct

    # Get apple position before we start
    apple_pos = env_sim.apple.pose.p[0].cpu().numpy()
    print(f"Apple at: {apple_pos}")

    # ===== Phase 1: Open the door using pre-recorded trajectory =====
    # Use hardcoded trajectory since robot/cabinet positions are fixed
    traj_path = os.path.join(os.path.dirname(__file__), "open_cabinet_trajectory.npy")
    door_trajectory = np.load(traj_path)

    for action in door_trajectory:
        env.step(action)
        planner.elapsed_steps += 1
        if vis:
            env_sim.render_human()

    door_pct = check_door()
    print(f"Phase 1 complete: Door at {door_pct:.1f}%")

    if door_pct < 40:
        print(f"Failed to open door sufficiently ({door_pct:.1f}%)")
        planner.close()
        return -1

    # ===== Phase 2: Bring arm back, go HIGH over door, hover above apple =====

    # For grasp, try multiple orientations - some may be more reachable than others
    grasp_quats = [
        np.array([0.0, 1.0, 0.0, 0.0]),  # 180° around X
        np.array([0.0, 0.7071, 0.7071, 0.0]),  # 180° around (X+Y)/sqrt(2)
        np.array([0.0, 0.0, 1.0, 0.0]),  # 180° around Y
        np.array([0.0, 0.7071, 0.0, 0.7071]),  # 180° around (X+Z)/sqrt(2)
    ]

    def get_current_quat():
        """Get current TCP quaternion."""
        q = env_sim.agent.tcp_pose.sp.q
        return np.array([q[0], q[1], q[2], q[3]])

    def nav_pose(pos):
        """Create a navigation pose using current orientation."""
        return sapien.Pose(pos, get_current_quat())

    # Get current TCP position after Phase 1
    tcp_pos = env_sim.agent.tcp_pose.sp.p
    print(f"TCP after Phase 1: {tcp_pos}")

    # Step 1: Pull back from handle (relative to TCP)
    tcp_pose = env_sim.agent.tcp_pose.sp
    retreat1 = tcp_pose * sapien.Pose([0, 0, -0.10])
    planner.move_to_pose_with_screw(retreat1)
    print(f"Step 1 - retreat back: Door at {check_door():.1f}%")

    # Step 2: Go straight UP to clear door - use current X,Y
    tcp_pos = env_sim.agent.tcp_pose.sp.p
    high_pos = np.array([tcp_pos[0], tcp_pos[1], 0.75])
    planner.move_to_pose_with_screw(nav_pose(high_pos))
    print(f"Step 2 - go high (z=0.75): Door at {check_door():.1f}%")

    # Step 3: Move to negative Y first (avoid door swing area) while staying high
    # Door swings to positive Y, apple is at negative Y
    waypoint1 = np.array([-0.35, -0.30, 0.75])  # Far back, negative Y, high
    planner.move_to_pose_with_screw(nav_pose(waypoint1))
    print(f"Step 3 - waypoint1: Door at {check_door():.1f}%")

    # Step 4: Move above apple position at high altitude
    above_apple_high = np.array([apple_pos[0], apple_pos[1], 0.65])
    planner.move_to_pose_with_screw(nav_pose(above_apple_high))
    tcp_actual = env_sim.agent.tcp_pose.sp.p
    print(f"Step 4 - above apple (z=0.65): Door at {check_door():.1f}%, TCP at {tcp_actual}")

    # ===== Phase 3: Grasp apple at an ANGLE (45 degrees from horizontal) =====
    planner.open_gripper()

    # Refresh apple position
    apple_pos_now = env_sim.apple.pose.p[0].cpu().numpy()
    print(f"Apple position now: {apple_pos_now}")

    # Angled grasp: approach at 45 degrees from above
    # Direction is normalized [0, 1, -1] / sqrt(2) = approach from -Y, -Z
    approaching = np.array([0.0, 0.7071, -0.7071])  # 45 degree angle
    closing = np.array([1.0, 0.0, 0.0])  # Fingers close along X

    # Step 5: Position behind and above apple
    grasp_height = apple_pos_now[2]  # Apple center height
    offset_y = -0.10  # Behind apple (negative Y)
    pre_grasp_pos = np.array([apple_pos_now[0], apple_pos_now[1] + offset_y, grasp_height + 0.10])
    pre_grasp_pose = env_sim.agent.build_grasp_pose(approaching, closing, pre_grasp_pos)
    res = planner.move_to_pose_with_screw(pre_grasp_pose)
    if res == -1:
        res = planner.move_to_pose_with_RRTConnect(pre_grasp_pose)
    tcp_actual = env_sim.agent.tcp_pose.sp.p
    print(f"Step 5 - angled approach: Door at {check_door():.1f}%, TCP at {tcp_actual}")

    # Step 6: Move in at angle to grasp apple at its center
    grasp_pos = np.array([apple_pos_now[0], apple_pos_now[1], grasp_height])
    grasp_pose = env_sim.agent.build_grasp_pose(approaching, closing, grasp_pos)
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1:
        res = planner.move_to_pose_with_RRTConnect(grasp_pose)

    tcp_actual = env_sim.agent.tcp_pose.sp.p
    apple_actual = env_sim.apple.pose.p[0].cpu().numpy()
    pos_error = np.linalg.norm(tcp_actual - grasp_pos)
    print(f"Step 6 - angled grasp: TCP={tcp_actual}, apple={apple_actual}, error={pos_error:.4f}")

    # Step 7: Close gripper to grasp apple
    planner.close_gripper()
    is_grasped = env_sim.agent.is_grasping(env_sim.apple)
    print(f"Step 7 - close gripper: is_grasped={is_grasped}, Door at {check_door():.1f}%")

    # Step 8: Lift apple (use current gripper orientation)
    tcp_pos = env_sim.agent.tcp_pose.sp.p
    lift_pos = np.array([tcp_pos[0], tcp_pos[1], 0.30])
    tcp_quat = get_current_quat()
    lift_pose = sapien.Pose(lift_pos, tcp_quat)
    planner.move_to_pose_with_screw(lift_pose)
    print(f"Step 8 - lifted apple: Door at {check_door():.1f}%")

    # Step 8b: Reorient gripper by rotating 90 degrees clockwise around Z axis
    # Clockwise from above = negative Z rotation
    # Quaternion for -90 deg around Z: [cos(-45°), 0, 0, sin(-45°)] = [0.7071, 0, 0, -0.7071]
    z_rot_quat = np.array([0.7071, 0.0, 0.0, -0.7071])  # -90 deg around Z

    # Get current TCP quaternion and compose with Z rotation
    current_quat = get_current_quat()  # wxyz format

    # Quaternion multiplication: q_new = q_z_rot * q_current
    # This applies the Z rotation in global frame
    def quat_multiply(q1, q2):
        """Multiply two quaternions in wxyz format."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    forward_quat = quat_multiply(z_rot_quat, current_quat)
    forward_quat = forward_quat / np.linalg.norm(forward_quat)  # Normalize

    tcp_pos = env_sim.agent.tcp_pose.sp.p
    reorient_pose = sapien.Pose(tcp_pos, forward_quat)
    res = planner.move_to_pose_with_screw(reorient_pose)
    if res == -1:
        res = planner.move_to_pose_with_RRTConnect(reorient_pose)
    print(f"Step 8b - reoriented 90° CW around Z: Door at {check_door():.1f}%")

    # Transport: keep angled grasp orientation and move towards cabinet
    # Cabinet dimensions: base at z=0.42, door handle at z~0.37
    # Bottom shelf surface at z=0.42, apple center should be at z=0.46 (0.42 + 0.04 radius)
    shelf_z = 0.46  # Apple center height on bottom shelf
    transport_z = 0.35  # High enough to safely clear door during transport

    # Step 9: Lift to transport height
    tcp_pos = env_sim.agent.tcp_pose.sp.p
    transport_pos = np.array([tcp_pos[0], tcp_pos[1], transport_z])
    res = planner.move_to_pose_with_screw(sapien.Pose(transport_pos, forward_quat))
    tcp_actual = env_sim.agent.tcp_pose.sp.p
    print(f"Step 9 - transport height (z={transport_z}): Door at {check_door():.1f}%, TCP at {tcp_actual}")

    # Step 9b: Move towards cabinet at transport height
    # X=-0.05 is reachable (about 0.25 from cabinet at X=0.20)
    place_x = 0.25
    towards_cabinet = np.array([place_x, 0.0, transport_z])
    res = planner.move_to_pose_with_screw(sapien.Pose(towards_cabinet, forward_quat))
    if res == -1:
        res = planner.move_to_pose_with_RRTConnect(sapien.Pose(towards_cabinet, forward_quat))
    tcp_actual = env_sim.agent.tcp_pose.sp.p
    is_grasped = env_sim.agent.is_grasping(env_sim.apple)
    print(f"Step 9b - towards cabinet: Door at {check_door():.1f}%, TCP at {tcp_actual}, grasped={is_grasped}")

    # Step 10: Lower to shelf level (z=0.46 for apple center on bottom shelf)
    place_pos = np.array([place_x, 0.0, shelf_z])
    res = planner.move_to_pose_with_screw(sapien.Pose(place_pos, forward_quat))
    if res == -1:
        res = planner.move_to_pose_with_RRTConnect(sapien.Pose(place_pos, forward_quat))
    tcp_actual = env_sim.agent.tcp_pose.sp.p
    apple_actual = env_sim.apple.pose.p[0].cpu().numpy()
    is_grasped = env_sim.agent.is_grasping(env_sim.apple)
    print(f"Step 10 - at shelf height (z={shelf_z}): Door at {check_door():.1f}%, TCP at {tcp_actual}, apple at {apple_actual}, grasped={is_grasped}")

    # Step 11: Release apple inside cabinet
    planner.open_gripper()
    apple_actual = env_sim.apple.pose.p[0].cpu().numpy()
    print(f"Step 11 - released apple: Door at {check_door():.1f}%, apple at {apple_actual}")

    # Step 12: Back away from shelf (stay at same height to avoid hitting apple)
    retreat1_pos = np.array([place_x - 0.10, 0.0, shelf_z])
    res = planner.move_to_pose_with_screw(sapien.Pose(retreat1_pos, forward_quat))
    if res == -1:
        res = planner.move_to_pose_with_RRTConnect(sapien.Pose(retreat1_pos, forward_quat))
    print(f"Step 12 - backed away: Door at {check_door():.1f}%")

    # Step 13: Move away from cabinet area
    retreat2_pos = np.array([-0.20, -0.20, 0.50])
    res = planner.move_to_pose_with_screw(sapien.Pose(retreat2_pos, forward_quat))
    if res == -1:
        res = planner.move_to_pose_with_RRTConnect(sapien.Pose(retreat2_pos, forward_quat))
    print(f"Step 13 - retreated: Door at {check_door():.1f}%")

    # Wait for apple to settle
    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape))

    print(f"Final door: {check_door():.1f}%")
    print(f"Apple final pos: {env_sim.apple.pose.p[0].cpu().numpy()}")

    planner.close()
    return obs, reward, terminated, truncated, info
