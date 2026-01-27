import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks.tabletop.colosseum_v2.open_drawer import OpenDrawerEnv
from mani_skill.utils import common
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb
from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply
import torch

def _yaw_from_quaternion(q: torch.Tensor) -> float:
    """
    Convert a quaternion to a yaw angle in radians.
    """
    return torch.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2]**2 + q[3]**2)).item()

def _rotate_vec_by_yaw(vec: np.ndarray, yaw: float) -> np.ndarray:
    """
    Rotate a vector by a yaw angle in radians.
    """
    return np.array([vec[0] * np.cos(yaw) - vec[1] * np.sin(yaw), vec[0] * np.sin(yaw) + vec[1] * np.cos(yaw), vec[2]])

def solve(env: OpenDrawerEnv, seed=None, debug=False, vis=False):
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
    FINGER_LENGTH = 0.025
    env = env.unwrapped
    handle_obj = env.handle_link_goal
    handle_pose_0 = handle_obj.pose 

    # Add rotational noise to the target grasp pose
    rotation_noise_range = np.deg2rad(1.0)
    rotation_noise = [np.random.uniform(-rotation_noise_range, rotation_noise_range) for _ in range(3)]
    drawer_yaw = _yaw_from_quaternion(env.cabinet.pose.q[0])
    target_rotation = euler2quat(np.pi / 2 + rotation_noise[0], -np.pi + rotation_noise[1], np.pi / 2 + rotation_noise[2] + drawer_yaw) # this was working

    # Configure target poses
    grasp_pose = sapien.Pose(q=target_rotation, p=handle_obj.pose.p.cpu().numpy()[0])
    reach_pose = grasp_pose * sapien.Pose([0, 0.0, -0.05])
    handle_target_pose = sapien.Pose(q=target_rotation, p=handle_pose_0.p.cpu().numpy()[0])
    handle_target_pose.p -= _rotate_vec_by_yaw(np.array([0.2, 0, 0]), drawer_yaw)

    res = planner.move_to_pose_with_screw(reach_pose, dry_run=True)
    if res == -1:
        print("Failed")
        return res


    # Execute
    planner.open_gripper()
    planner.move_to_pose_with_screw(reach_pose)
    planner.open_gripper()
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()
    planner.move_to_pose_with_screw(handle_target_pose)
    res = planner.open_gripper()
    planner.close()
    return res
