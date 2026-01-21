import numpy as np
import sapien

from mani_skill.envs.tasks.tabletop.object_in_cabinet import ObjectInCabinetEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

from mani_skill.examples.motionplanning.panda.solutions.open_cabinet import (
    _get_joint_limits,
    _open_cabinet_with_planner,
)


def solve(env: ObjectInCabinetEnv, seed=None, debug=False, vis=False):
    """
    Solution for ObjectInCabinet task using screw motion only.
    Phase 1: Open cabinet door
    Phase 2: Move arm back and over door to hover above obj
    Phase 3: Grasp and transport object to cabinet
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
        joint_vel_limits=1.5,
        joint_acc_limits=1.5,
    )

    qmin, qmax = _get_joint_limits(env_sim.handle_link.joint)

    def check_door():
        qpos = env_sim.handle_link.joint.qpos[0].item()
        pct = (qpos - qmin) / (qmax - qmin) * 100
        return pct

    # Get obj position before we start
    obj_pos = env_sim.obj.pose.p[0].cpu().numpy()
    if debug:
        print(f"Object at: {obj_pos}")

    # Print initial EE-to-handle distance
    handle_pos = env_sim.handle_link_positions()[0].cpu().numpy()
    ee_pos = env_sim.agent.tcp_pose.p[0].cpu().numpy()
    ee_handle_dist = np.linalg.norm(ee_pos - handle_pos)
    if debug:
        print(f"Initial EE-to-handle distance: {ee_handle_dist:.3f}m")

    # ===== Phase 1: Open the door using live motion planning =====
    res = _open_cabinet_with_planner(env, planner, debug=debug, target_frac=0.95)
    if res == -1:
        if debug:
            print("Failed to open cabinet door")
        planner.close()
        return -1

    door_pct = check_door()
    if debug:
        print(f"Phase 1 complete: Door at {door_pct:.1f}%")

    if door_pct < 40:
        if debug:
            print(f"Failed to open door sufficiently ({door_pct:.1f}%)")
        planner.close()
        return -1

    # ===== Phase 2: Bring arm back, go HIGH over door, hover above obj =====

    def get_current_quat():
        """Get current TCP quaternion."""
        q = env_sim.agent.tcp_pose.sp.q
        return np.array([q[0], q[1], q[2], q[3]])

    def nav_pose(pos):
        """Create a navigation pose using current orientation."""
        return sapien.Pose(pos, get_current_quat())

    # Get current TCP position after Phase 1
    tcp_pos = env_sim.agent.tcp_pose.sp.p
    if debug:
        print(f"TCP after Phase 1: {tcp_pos}")

    # Step 1: Pull back from handle (relative to TCP)
    tcp_pose = env_sim.agent.tcp_pose.sp
    retreat1 = tcp_pose * sapien.Pose([0, 0, -0.01])
    res = planner.move_to_pose_with_screw(retreat1)
    if res == -1:
        if debug:
            print("Failed at Step 1 - retreat back")
        planner.close()
        return -1
    if debug:
        print(f"Step 1 - retreat back: Door at {check_door():.1f}%")

    # Step 2: Go straight UP to clear door - use current X,Y
    tcp_pos = env_sim.agent.tcp_pose.sp.p
    high_pos = np.array([tcp_pos[0], tcp_pos[1], 0.75])
    res = planner.move_to_pose_with_screw(nav_pose(high_pos))
    if res == -1:
        if debug:
            print("Failed at Step 2 - go high")
        planner.close()
        return -1
    if debug:
        print(f"Step 2 - go high (z=0.75): Door at {check_door():.1f}%")

    # Step 3: Move to negative Y first (avoid door swing area) while staying high
    # Door swings to positive Y, obj is at negative Y
    waypoint1 = np.array([-0.35, -0.30, 0.75])  # Far back, negative Y, high
    res = planner.move_to_pose_with_screw(nav_pose(waypoint1))
    if res == -1:
        if debug:
            print("Failed at Step 3 - waypoint1")
        planner.close()
        return -1
    if debug:
        print(f"Step 3 - waypoint1: Door at {check_door():.1f}%")

    # Step 4: Move above obj position at high altitude
    above_obj_high = np.array([obj_pos[0], obj_pos[1], 0.65])
    res = planner.move_to_pose_with_screw(nav_pose(above_obj_high))
    if res == -1:
        if debug:
            print("Failed at Step 4 - above obj")
        planner.close()
        return -1
    tcp_actual = env_sim.agent.tcp_pose.sp.p
    if debug:
        print(f"Step 4 - above obj (z=0.65): Door at {check_door():.1f}%, TCP at {tcp_actual}")

    # ===== Phase 3: Grasp object at an angle (45 degrees from horizontal) =====
    planner.open_gripper()

    # Angled grasp: approach at 45 degrees from above
    approaching = np.array([0.0, 0.7071, -0.7071])  # 45 degree angle
    closing = np.array([1.0, 0.0, 0.0])  # Fingers close along X

    # Grasp height - banana is low, use 0.04 to clear table
    grasp_height = 0.01

    # Step 5: Position behind and above object
    offset_y = -0.12  # Behind obj (negative Y)
    pre_grasp_pos = np.array([obj_pos[0], obj_pos[1] + offset_y, grasp_height + 0.08])
    pre_grasp_pose = env_sim.agent.build_grasp_pose(approaching, closing, pre_grasp_pos)
    res = planner.move_to_pose_with_screw(pre_grasp_pose)
    if res == -1:
        if debug:
            print("Failed at Step 5 - angled approach")
        planner.close()
        return -1
    tcp_actual = env_sim.agent.tcp_pose.sp.p
    if debug:
        print(f"Step 5 - angled approach: Door at {check_door():.1f}%, TCP at {tcp_actual}")

    # Step 6: Move in at angle to grasp object
    grasp_pos = np.array([obj_pos[0], obj_pos[1], grasp_height])
    grasp_pose = env_sim.agent.build_grasp_pose(approaching, closing, grasp_pos)
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1:
        if debug:
            print("Failed at Step 6 - angled grasp")
        planner.close()
        return -1

    tcp_actual = env_sim.agent.tcp_pose.sp.p
    obj_actual = env_sim.obj.pose.p[0].cpu().numpy()
    pos_error = np.linalg.norm(tcp_actual - grasp_pos)
    if debug:
        print(f"Step 6 - angled grasp: TCP={tcp_actual}, obj={obj_actual}, error={pos_error:.4f}")

    # Step 7: Close gripper to grasp obj
    planner.close_gripper()
    is_grasped = env_sim.agent.is_grasping(env_sim.obj)
    if debug:
        print(f"Step 7 - close gripper: is_grasped={is_grasped}, Door at {check_door():.1f}%")

    # Step 8: Lift obj (use current gripper orientation) - keep same orientation for stability
    tcp_pos = env_sim.agent.tcp_pose.sp.p
    lift_pos = np.array([tcp_pos[0], tcp_pos[1], 0.30])
    grasp_quat = get_current_quat()  # Save the grasp orientation
    lift_pose = sapien.Pose(lift_pos, grasp_quat)
    res = planner.move_to_pose_with_screw(lift_pose)
    if res == -1:
        if debug:
            print("Failed at Step 8 - lift obj")
        planner.close()
        return -1
    is_grasped = env_sim.agent.is_grasping(env_sim.obj)
    if debug:
        print(f"Step 8 - lifted obj: Door at {check_door():.1f}%, grasped={is_grasped}")

    # Step 8b: Rotate 90 degrees around Z axis to reorient banana for cabinet
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

    z_rot_quat = np.array([0.7071, 0.0, 0.0, -0.7071])  # -90 deg around Z
    transport_quat = quat_multiply(z_rot_quat, grasp_quat)
    transport_quat = transport_quat / np.linalg.norm(transport_quat)

    tcp_pos = env_sim.agent.tcp_pose.sp.p
    rotate_pose = sapien.Pose(tcp_pos, transport_quat)
    res = planner.move_to_pose_with_screw(rotate_pose)
    if res == -1:
        if debug:
            print("Failed at Step 8b - rotate 90°")
        planner.close()
        return -1
    is_grasped = env_sim.agent.is_grasping(env_sim.obj)
    if debug:
        print(f"Step 8b - rotated 90°: Door at {check_door():.1f}%, grasped={is_grasped}")

    # Transport: use rotated orientation
    # Cabinet dimensions: base at z=0.42, door handle at z~0.37
    # Bottom shelf surface at z=0.42, obj center should be at z=0.46 (0.42 + 0.04 radius)
    shelf_z = 0.46  # Object center height on bottom shelf
    transport_z = 0.35  # High enough to safely clear door during transport

    # Step 9: Lift to transport height
    tcp_pos = env_sim.agent.tcp_pose.sp.p
    transport_pos = np.array([tcp_pos[0], tcp_pos[1]+0.05, transport_z + 0.1])
    res = planner.move_to_pose_with_screw(sapien.Pose(transport_pos, transport_quat))
    if res == -1:
        if debug:
            print("Failed at Step 9 - transport height")
        planner.close()
        return -1
    tcp_actual = env_sim.agent.tcp_pose.sp.p
    is_grasped = env_sim.agent.is_grasping(env_sim.obj)
    if debug:
        print(f"Step 9 - transport height (z={transport_z}): Door at {check_door():.1f}%, TCP at {tcp_actual}, grasped={is_grasped}")

    # Step 9b: Move towards cabinet
    place_x = 0.20
    towards_cabinet = np.array([place_x, 0.0, transport_z])
    res = planner.move_to_pose_with_screw(sapien.Pose(towards_cabinet, transport_quat))
    if res == -1:
        if debug:
            print("Failed at Step 9c - towards cabinet")
        planner.close()
        return -1
    tcp_actual = env_sim.agent.tcp_pose.sp.p
    is_grasped = env_sim.agent.is_grasping(env_sim.obj)
    if debug:
        print(f"Step 9c - towards cabinet: Door at {check_door():.1f}%, TCP at {tcp_actual}, grasped={is_grasped}")

    # Step 10: Lower to shelf level (z=0.46 for obj center on bottom shelf)
    place_pos = np.array([place_x, 0.0, shelf_z+0.05])
    res = planner.move_to_pose_with_screw(sapien.Pose(place_pos, transport_quat))
    if res == -1:
        if debug:
            print("Failed at Step 10 - shelf height")
        planner.close()
        return -1
    tcp_actual = env_sim.agent.tcp_pose.sp.p
    obj_actual = env_sim.obj.pose.p[0].cpu().numpy()
    is_grasped = env_sim.agent.is_grasping(env_sim.obj)
    if debug:
        print(f"Step 10 - at shelf height (z={shelf_z}): Door at {check_door():.1f}%, TCP at {tcp_actual}, obj at {obj_actual}, grasped={is_grasped}")

    # Step 11: Release obj inside cabinet
    planner.open_gripper()
    if debug:
        obj_actual = env_sim.obj.pose.p[0].cpu().numpy()
        print(f"Step 11 - released obj: Door at {check_door():.1f}%, obj at {obj_actual}")

    # Step 12: Retreat 10cm
    tcp_pose = env_sim.agent.tcp_pose.sp
    retreat_pose = tcp_pose * sapien.Pose([-0.10, 0, 0])
    planner.move_to_pose_with_screw(retreat_pose)

    # Wait for obj to settle
    for _ in range(30):
        obs, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape))

    planner.close()
    return obs, reward, terminated, truncated, info
