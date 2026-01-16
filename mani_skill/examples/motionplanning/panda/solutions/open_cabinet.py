import numpy as np
import sapien

from mani_skill.envs.tasks.tabletop.open_cabinet import OpenCabinetEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
)
from mani_skill.utils.geometry.trimesh_utils import merge_meshes


def _normalize(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Normalize a vector, returning fallback if norm is too small."""
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return fallback
    return vec / norm


def _get_joint_axis(joint) -> np.ndarray:
    """Get the global rotation axis for a joint."""
    axis = None
    axis_is_local = True
    raw_joint = joint._objs[0] if hasattr(joint, "_objs") and joint._objs else None
    if raw_joint is not None:
        if hasattr(raw_joint, "get_axis"):
            try:
                axis = np.array(raw_joint.get_axis(), dtype=np.float32)
            except Exception:
                axis = None
        if axis is None and hasattr(raw_joint, "axis"):
            axis = np.array(raw_joint.axis, dtype=np.float32)
    if axis is None and hasattr(joint, "get_axis"):
        try:
            axis = np.array(joint.get_axis(), dtype=np.float32)
        except Exception:
            axis = None
    if axis is None and hasattr(joint, "axis"):
        axis = np.array(joint.axis, dtype=np.float32)
    if axis is None:
        joint_pose = joint.get_global_pose().to_transformation_matrix()
        if hasattr(joint_pose, "ndim") and joint_pose.ndim == 3:
            joint_pose = joint_pose[0]
        if hasattr(joint_pose, "cpu"):
            joint_pose = joint_pose.cpu().numpy()
        axis = np.array(joint_pose[:3, 0], dtype=np.float32)
        axis_is_local = False
    if axis_is_local:
        joint_pose = joint.get_global_pose().to_transformation_matrix()
        if hasattr(joint_pose, "ndim") and joint_pose.ndim == 3:
            joint_pose = joint_pose[0]
        if hasattr(joint_pose, "cpu"):
            joint_pose = joint_pose.cpu().numpy()
        axis = joint_pose[:3, :3] @ axis
    return _normalize(axis, np.array([0.0, 0.0, 1.0], dtype=np.float32))


def _get_joint_pivot(joint) -> np.ndarray:
    """Get the pivot point (position) of a joint in global coordinates."""
    joint_pose = joint.get_global_pose()
    pivot = joint_pose.p
    if hasattr(pivot, "cpu"):
        pivot = pivot.cpu().numpy()
    pivot = np.array(pivot)
    if pivot.ndim == 2:
        pivot = pivot[0]
    return pivot


def _get_joint_limits(joint) -> tuple[float, float]:
    """Get the min and max joint limits."""
    qlimits = joint.limits
    if hasattr(qlimits, "cpu"):
        qlimits = qlimits.cpu().numpy()
    qlimits = np.array(qlimits)
    if qlimits.ndim == 3:
        qmin = float(qlimits[0, 0, 0])
        qmax = float(qlimits[0, 0, 1])
    elif qlimits.ndim == 2:
        qmin = float(qlimits[0, 0])
        qmax = float(qlimits[0, 1])
    elif qlimits.ndim == 1 and qlimits.size >= 2:
        qmin = float(qlimits[0])
        qmax = float(qlimits[1])
    else:
        qmin, qmax = -1.0, 1.0
    if qmin > qmax:
        qmin, qmax = qmax, qmin
    return qmin, qmax


def _get_handle_obb(handle_link):
    """Get the oriented bounding box of the handle mesh."""
    try:
        meshes = handle_link.generate_mesh(
            filter=lambda _, render_shape: "handle" in render_shape.name,
            mesh_name="handle",
        )
    except Exception:
        return None
    if not meshes:
        return None
    merged = merge_meshes([mesh for mesh in meshes if mesh is not None])
    if merged is None:
        return None
    link_pose = handle_link.pose.to_transformation_matrix()
    if hasattr(link_pose, "ndim") and link_pose.ndim == 3:
        link_pose = link_pose[0]
    if hasattr(link_pose, "cpu"):
        link_pose = link_pose.cpu().numpy()
    merged.apply_transform(link_pose)
    return merged.bounding_box_oriented


def _rotate_point_about_axis(
    point: np.ndarray, pivot: np.ndarray, axis: np.ndarray, angle: float
) -> np.ndarray:
    """Rotate a point about an axis (Rodrigues' rotation formula)."""
    axis = _normalize(axis, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    vec = point - pivot
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    rotated = (
        vec * cos_theta
        + np.cross(axis, vec) * sin_theta
        + axis * (axis @ vec) * (1.0 - cos_theta)
    )
    return pivot + rotated


def _rotate_vec_about_axis(vec: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a vector about an axis (Rodrigues' rotation formula)."""
    axis = _normalize(axis, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    return (
        vec * np.cos(angle)
        + np.cross(axis, vec) * np.sin(angle)
        + axis * (axis @ vec) * (1.0 - np.cos(angle))
    )


def _open_cabinet_with_planner(
    env: OpenCabinetEnv, planner: PandaArmMotionPlanningSolver
):
    """Execute motion plan to open the cabinet door in a smooth, single motion."""
    env_sim = env.unwrapped

    # Get handle and joint information
    handle_pos = env_sim.handle_link_positions()[0].cpu().numpy()
    joint = env_sim.handle_link.joint
    axis = _get_joint_axis(joint)
    pivot = _get_joint_pivot(joint)
    robot_pos = env_sim.agent.robot.pose.p[0].cpu().numpy()

    # Get handle oriented bounding box for precise grasp
    handle_obb = _get_handle_obb(env_sim.handle_link)

    # Calculate approach direction (from robot to handle)
    approaching = _normalize(handle_pos - robot_pos, np.array([1.0, 0.0, 0.0], dtype=np.float32))

    # Door normal perpendicular to axis and radial direction
    radial = _normalize(handle_pos - pivot, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    door_normal = np.cross(axis, radial)
    door_normal = _normalize(door_normal, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    if np.dot(door_normal, robot_pos - handle_pos) < 0:
        door_normal = -door_normal

    # Tangent direction for gripper closing
    tangent = _normalize(np.cross(axis, door_normal), np.array([0.0, 1.0, 0.0], dtype=np.float32))

    # Compute grasp pose
    finger_length = 0.025
    grasp_backoff = -0.005
    if handle_obb is not None:
        grasp_info = compute_grasp_info_by_obb(
            handle_obb,
            approaching=approaching,
            target_closing=tangent,
            depth=finger_length,
        )
        closing = grasp_info["closing"]
        center = grasp_info["center"] + approaching * grasp_backoff
    else:
        closing = _normalize(np.cross(axis, approaching), np.array([0.0, 1.0, 0.0], dtype=np.float32))
        center = handle_pos + approaching * grasp_backoff

    grasp_pose = env_sim.agent.build_grasp_pose(approaching, closing, center)

    # Phase 1: Approach - move to pre-grasp position
    planner.open_gripper()
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.10])

    res = planner.move_to_pose_with_RRTConnect(reach_pose)
    if res == -1:
        res = planner.move_to_pose_with_screw(reach_pose)
    if res == -1:
        print("Failed to reach pre-grasp pose")
        planner.close()
        return -1

    # Phase 2: Move to grasp position
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1:
        res = planner.move_to_pose_with_RRTConnect(grasp_pose)
    if res == -1:
        print("Failed to reach grasp pose")
        planner.close()
        return res

    # Phase 3: Grasp the handle
    planner.close_gripper()

    # Get current joint state and limits
    qpos = env_sim.handle_link.joint.qpos
    current_qpos = qpos[0].item() if qpos.ndim > 0 else float(qpos)
    qmin, qmax = _get_joint_limits(env_sim.handle_link.joint)

    # Target: open to 90% of max range for >80% success criterion
    target_qpos = qmin + 0.90 * abs(qmax - qmin)

    # Phase 4: Open the door by following an arc
    # Record initial handle position for arc calculation
    handle_pos0 = env_sim.handle_link_positions()[0].cpu().numpy()

    delta = target_qpos - current_qpos
    if abs(delta) < 1e-3:
        delta = 0.5 * abs(qmax - qmin)

    # Use smooth arc motion with small angle steps for reliable planning
    # Smaller steps = more reliable screw motion planning
    angle_step = 0.02  # ~1.15 degrees per step
    num_steps = max(40, int(abs(delta) / angle_step))
    step_angle = delta / num_steps
    current_angle = 0.0

    # Pull-back offsets to keep gripper behind the door surface
    pull_offsets = [-0.02, -0.04, -0.06, -0.08, -0.10]
    consecutive_failures = 0
    max_failures = 8  # Allow more failures before giving up

    for _ in range(num_steps):
        target_angle = current_angle + step_angle

        # Calculate where the handle should be at the target angle
        target_handle = _rotate_point_about_axis(handle_pos0, pivot, axis, target_angle)

        # Rotate approach and closing directions to match door rotation
        approach_rot = _rotate_vec_about_axis(approaching, axis, target_angle)
        closing_rot = _rotate_vec_about_axis(closing, axis, target_angle)

        # Build the target pose at the new handle position
        target_pose = env_sim.agent.build_grasp_pose(
            approach_rot, closing_rot, target_handle + approach_rot * grasp_backoff
        )

        # Try different pull-back distances with screw motion (preferred for smooth arc)
        res = -1
        for pull_back in pull_offsets:
            pull_pose = target_pose * sapien.Pose([0, 0, pull_back])
            res = planner.move_to_pose_with_screw(pull_pose)
            if res != -1:
                current_angle = target_angle
                consecutive_failures = 0
                break

        # If screw fails, try RRT (less smooth but may find path)
        if res == -1:
            for pull_back in pull_offsets:
                pull_pose = target_pose * sapien.Pose([0, 0, pull_back])
                res = planner.move_to_pose_with_RRTConnect(pull_pose)
                if res != -1:
                    current_angle = target_angle
                    consecutive_failures = 0
                    break

        if res == -1:
            consecutive_failures += 1
            # Try smaller step if having trouble
            if consecutive_failures >= max_failures:
                # Try to continue from actual current position
                handle_pos0 = env_sim.handle_link_positions()[0].cpu().numpy()
                qpos = env_sim.handle_link.joint.qpos
                qpos_val = qpos[0].item() if qpos.ndim > 0 else float(qpos)
                current_angle = qpos_val - current_qpos
                consecutive_failures = 0
                # If we've opened past 85%, we can stop
                if qpos_val >= qmin + 0.85 * abs(qmax - qmin):
                    break

        # Check current door opening
        qpos = env_sim.handle_link.joint.qpos
        qpos_val = qpos[0].item() if qpos.ndim > 0 else float(qpos)

        # Success: door is open enough
        if qpos_val >= target_qpos - 0.01:
            break

    # Phase 5: Release and retreat
    planner.open_gripper()

    # Retreat from current position
    tcp_pose = env_sim.agent.tcp_pose.sp
    retreat_pose = tcp_pose * sapien.Pose([0, 0, -0.10])
    res = planner.move_to_pose_with_RRTConnect(retreat_pose)

    # Final status
    qpos = env_sim.handle_link.joint.qpos
    final_qpos = qpos[0].item() if qpos.ndim > 0 else float(qpos)
    open_frac = (final_qpos - qmin) / (qmax - qmin) if abs(qmax - qmin) > 1e-6 else 0.0
    print(f"Cabinet opened to {open_frac*100:.1f}% (qpos={final_qpos:.3f}, target={target_qpos:.3f})")

    return res


def solve(env: OpenCabinetEnv, seed=None, debug=False, vis=False):
    """Solve the OpenCabinet task using motion planning."""
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
    res = _open_cabinet_with_planner(env, planner)
    planner.close()
    return res
