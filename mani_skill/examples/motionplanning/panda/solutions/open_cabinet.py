import numpy as np
import sapien

from mani_skill.envs.tasks.tabletop.open_cabinet import OpenCabinetEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
)
from mani_skill.utils.geometry.trimesh_utils import merge_meshes


def _normalize(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return fallback
    return vec / norm


def _rotate_vec_about_axis(vec: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _normalize(axis, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    return (
        vec * np.cos(angle)
        + np.cross(axis, vec) * np.sin(angle)
        + axis * (axis @ vec) * (1.0 - np.cos(angle))
    )


def _get_joint_axis(joint) -> np.ndarray:
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
    return _normalize(axis, np.array([1.0, 0.0, 0.0], dtype=np.float32))


def _get_joint_pivot(joint) -> np.ndarray:
    joint_pose = joint.get_global_pose()
    pivot = joint_pose.p
    if hasattr(pivot, "cpu"):
        pivot = pivot.cpu().numpy()
    pivot = np.array(pivot)
    if pivot.ndim == 2:
        pivot = pivot[0]
    return pivot


def _rotate_point_about_axis(
    point: np.ndarray, pivot: np.ndarray, axis: np.ndarray, angle: float
) -> np.ndarray:
    axis = _normalize(axis, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    vec = point - pivot
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    rotated = (
        vec * cos_theta
        + np.cross(axis, vec) * sin_theta
        + axis * (axis @ vec) * (1.0 - cos_theta)
    )
    return pivot + rotated


def _get_joint_limits(joint) -> tuple[float, float]:
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


def _find_reach_pose(planner, poses: list[sapien.Pose]):
    for pose in poses:
        res = planner.move_to_pose_with_RRTConnect(pose, dry_run=True)
        if res == -1:
            res = planner.move_to_pose_with_screw(pose, dry_run=True)
        if res != -1:
            return pose
    return None


def _open_cabinet_with_planner(
    env: OpenCabinetEnv, planner: PandaArmMotionPlanningSolver
):
    env_sim = env.unwrapped
    handle_pos = env_sim.handle_link_positions()[0].cpu().numpy()
    joint = env_sim.handle_link.joint
    axis = _get_joint_axis(joint)
    pivot = _get_joint_pivot(joint)
    robot_pos = env_sim.agent.robot.pose.p[0].cpu().numpy()
    handle_obb = _get_handle_obb(env_sim.handle_link)
    radial = _normalize(handle_pos - pivot, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    door_normal = np.cross(axis, radial)
    door_normal = _normalize(door_normal, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    if np.dot(door_normal, robot_pos - handle_pos) < 0:
        door_normal = -door_normal
    approaching = _normalize(
        handle_pos - robot_pos, np.array([1.0, 0.0, 0.0], dtype=np.float32)
    )
    tangent = _normalize(
        np.cross(axis, door_normal), np.array([0.0, 1.0, 0.0], dtype=np.float32)
    )
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
        closing = _normalize(
            np.cross(axis, approaching), np.array([0.0, 1.0, 0.0], dtype=np.float32)
        )
        center = handle_pos + approaching * grasp_backoff
    grasp_pose = env_sim.agent.build_grasp_pose(approaching, closing, center)
    reach_candidates = [
        grasp_pose * sapien.Pose([0, 0, -0.08]),
        grasp_pose * sapien.Pose([0, 0, -0.12]),
    ]
    reach_pose = _find_reach_pose(planner, reach_candidates)
    if reach_pose is None:
        print("Failed to find path to reach pose")
        planner.close()
        return -1

    planner.open_gripper()
    res = planner.move_to_pose_with_RRTConnect(reach_pose)
    if res == -1:
        res = planner.move_to_pose_with_screw(reach_pose)
    if res == -1:
        print("Failed to reach")
        planner.close()
        return res

    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1:
        res = planner.move_to_pose_with_RRTConnect(grasp_pose)
    if res == -1:
        print("Failed to grasp")
        planner.close()
        return res

    planner.close_gripper()
    handle_pos0 = env_sim.handle_link_positions()[0].cpu().numpy()
    qpos = env_sim.handle_link.joint.qpos
    current_qpos = qpos[0].item() if qpos.ndim > 0 else float(qpos)
    qmin, qmax = _get_joint_limits(env_sim.handle_link.joint)
    target_qpos = qmax - 0.01 * abs(qmax - qmin)
    delta = target_qpos - current_qpos
    if abs(delta) < 1e-3:
        delta = 0.7 * abs(qmax - qmin)

    num_steps = max(36, int(abs(delta) / 0.04))
    step_angle = delta / num_steps
    current_angle = 0.0
    pull_offsets = [0.0, -0.01, -0.02]
    last_pose = env_sim.agent.tcp_pose.sp
    for _ in range(1, num_steps + 1):
        target_angle = current_angle + step_angle
        attempt_angles = [
            target_angle,
            current_angle + 0.5 * step_angle,
            current_angle + 0.25 * step_angle,
        ]
        res = -1
        for angle in attempt_angles:
            target_handle = _rotate_point_about_axis(handle_pos0, pivot, axis, angle)
            approach_rot = _rotate_vec_about_axis(approaching, axis, angle)
            closing_rot = _rotate_vec_about_axis(closing, axis, angle)
            target_pose = env_sim.agent.build_grasp_pose(
                approach_rot,
                closing_rot,
                target_handle + approach_rot * grasp_backoff,
            )
            for pull_back in pull_offsets:
                pull_pose = target_pose * sapien.Pose([0, 0, pull_back])
                res = planner.move_to_pose_with_RRTConnect(pull_pose)
                if res == -1:
                    res = planner.move_to_pose_with_screw(pull_pose)
                if res != -1:
                    current_angle = angle
                    break
            if res != -1:
                break
        if res == -1:
            print("Failed during arc")
            break

        last_pose = env_sim.agent.tcp_pose.sp
        qpos = env_sim.handle_link.joint.qpos
        qpos_val = qpos[0].item() if qpos.ndim > 0 else float(qpos)
        if qpos_val >= target_qpos - 1e-3:
            break

    planner.open_gripper()
    retreat_pose = last_pose * sapien.Pose([0, 0, -0.1])
    res = planner.move_to_pose_with_RRTConnect(retreat_pose)
    return res


def solve(env: OpenCabinetEnv, seed=None, debug=False, vis=False):
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
