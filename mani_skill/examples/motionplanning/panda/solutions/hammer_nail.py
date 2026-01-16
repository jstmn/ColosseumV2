import numpy as np
import sapien
import sapien.render
from scipy.spatial.transform import Rotation

from mani_skill.envs.tasks.tabletop.hammer_nail import HammerNailEnv
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)
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

    print(f"Starting hammer position: {hammer_pos}")
    print(f"Starting nail position: {nail_center}")
    print(f"Starting block position: {block_pos}")

    # Grasp hammer from above
    approaching = np.array([0.0, 0.0, -1.0])

    obb = get_actor_obb(env_sim.hammer)
    obb_T = np.array(obb.primitive.transform)
    obb_center = obb_T[:3, 3]
    obb_axes = obb_T[:3, :3]
    obb_extents = np.array(obb.primitive.extents)

    # Find handle axis (longest extent)
    long_idx = int(np.argmax(obb_extents))
    handle_axis = obb_axes[:, long_idx]
    handle_axis = handle_axis / (np.linalg.norm(handle_axis) + 1e-6)
    handle_axis[2] = 0.0
    handle_axis_norm = np.linalg.norm(handle_axis)
    if handle_axis_norm < 1e-3:
        handle_axis = np.array([1.0, 0.0, 0.0])
    else:
        handle_axis = handle_axis / handle_axis_norm

    # Direction toward head (CoM is near the heavy head)
    com_world = (env_sim.hammer.pose * env_sim.hammer.cmass_local_pose).p[0].cpu().numpy()
    if np.dot(com_world - obb_center, handle_axis) < 0.0:
        handle_axis = -handle_axis

    # Closing direction perpendicular to handle_axis
    closing = np.cross(approaching, handle_axis)
    closing = closing / (np.linalg.norm(closing) + 1e-6)

    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=closing,
        depth=0.0,
    )
    base_center = grasp_info["center"]

    handle_length = float(obb_extents[long_idx])
    # Grip near the center of mass for stability (CoM is closer to the head)
    # Compute grasp point closer to CoM
    com_to_center = np.dot(com_world - obb_center, handle_axis)
    grasp_offset = com_to_center * 0.7  # Grip 70% toward CoM from center
    grasp_point = base_center + handle_axis * grasp_offset
    grasp_point[2] = base_center[2] - 0.015

    grasp_to_head = handle_length * 0.5 - grasp_offset

    grasp_pose = env_sim.agent.build_grasp_pose(approaching, closing, grasp_point)
    reach_pose = grasp_pose * sapien.Pose([0.0, 0.0, -0.08])

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

    planner.close_gripper(t=100, gripper_state=-1.0)

    # Lift hammer to a safe height first
    lift_height = nail_center[2] + 0.10  # A bit above nail height
    lift_pose = sapien.Pose(
        [grasp_pose.p[0], grasp_pose.p[1], lift_height],
        grasp_pose.q,
    )
    result = planner.move_to_pose_with_RRTConnect(lift_pose)
    if result == -1:
        result = planner.move_to_pose_with_screw(lift_pose)
        if result == -1:
            planner.close()
            return -1

    # Reorient hammer so the head points along +Y for striking
    # Keep gripper pointing down (Z = [0, 0, -1]) - this is more stable for IK
    # Just rotate around Z so gripper X (hammer head) points along +Y

    # Current gripper rotation
    q_wxyz = lift_pose.q
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    current_rot = Rotation.from_quat(q_xyzw)
    current_x = current_rot.apply(np.array([1.0, 0.0, 0.0]))
    current_x[2] = 0.0
    current_x = current_x / (np.linalg.norm(current_x) + 1e-6)

    # Target: gripper X points along +X (strike along +X, from left to right)
    target_direction = np.array([1.0, 0.0, 0.0])
    current_angle = np.arctan2(current_x[1], current_x[0])
    target_angle = np.arctan2(target_direction[1], target_direction[0])
    yaw_angle = target_angle - current_angle

    # Normalize to [-pi, pi]
    while yaw_angle > np.pi:
        yaw_angle -= 2 * np.pi
    while yaw_angle < -np.pi:
        yaw_angle += 2 * np.pi

    z_rot = Rotation.from_euler('z', yaw_angle, degrees=False)
    reoriented_rot = z_rot * current_rot
    reoriented_q_xyzw = reoriented_rot.as_quat()
    reoriented_q_wxyz = np.array([
        reoriented_q_xyzw[3], reoriented_q_xyzw[0],
        reoriented_q_xyzw[1], reoriented_q_xyzw[2]
    ])

    # Move to reoriented pose at current position
    reoriented_pose = sapien.Pose(lift_pose.p, reoriented_q_wxyz)
    result = planner.move_to_pose_with_RRTConnect(reoriented_pose)
    if result == -1:
        result = planner.move_to_pose_with_screw(reoriented_pose)
        if result == -1:
            planner.close()
            return -1

    # Head offset in world frame
    # With gripper X pointing +Y, head_offset_local=[grasp_to_head, 0, 0] becomes [0, grasp_to_head, 0] in world
    head_offset_world = reoriented_rot.apply(np.array([grasp_to_head, 0.0, 0.0]))
    print(f"Head offset world: {head_offset_world}")
    print(f"Gripper X axis: {reoriented_rot.apply([1, 0, 0])}")

    # The hammer is held from above, so the hammer handle center is roughly at gripper height
    # For striking, we need hammer HEAD at nail X and Z, with Y offset
    # Gripper position = desired head position - head_offset_world

    # Ready position: hammer head to LEFT of nail (same Y and Z, negative X offset)
    ready_head_pos = np.array([
        nail_center[0] + 0.03,   # Left of nail (negative X offset)
        nail_center[1] + 0.1,          # Same Y as nail
        nail_center[2] + 0.02           # Slightly above nail
    ])
    ready_gripper_pos = ready_head_pos - head_offset_world
    ready_pose = sapien.Pose(ready_gripper_pos, reoriented_q_wxyz)

    print(f"Nail center: {nail_center}")
    print(f"Ready head pos: {ready_head_pos}")
    print(f"Ready gripper pos: {ready_gripper_pos}")

    result = planner.move_to_pose_with_RRTConnect(ready_pose)
    if result == -1:
        result = planner.move_to_pose_with_screw(ready_pose)
        if result == -1:
            planner.close()
            return -1

    # Strike: move hammer head to nail and past (along +X)
    strike_head_pos = np.array([
        nail_center[0] + 0.03,   # Past nail (positive X)
        nail_center[1] - 0.12,          # Same Y as nail
        nail_center[2] + 0.02           # Same Z as nail
    ])
    strike_gripper_pos = strike_head_pos - head_offset_world
    strike_pose = sapien.Pose(strike_gripper_pos, reoriented_q_wxyz)

    result = planner.move_to_pose_with_screw(strike_pose)
    if result == -1:
        result = planner.move_to_pose_with_RRTConnect(strike_pose)
        if result == -1:
            planner.close()
            return -1

    # Pull back
    result = planner.move_to_pose_with_screw(ready_pose)
    if result == -1:
        planner.close()
        return -1

    # Drop hammer
    planner.open_gripper()

    planner.close()
    return result
