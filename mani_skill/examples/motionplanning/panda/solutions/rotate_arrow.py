import numpy as np
import sapien
import gymnasium as gym
import torch
import time
from transforms3d.euler import euler2quat
from mani_skill.envs.tasks.tabletop.colosseum_v2.rotate_arrow import RotateArrowEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb
from copy import deepcopy
from time import sleep

def main():
    env: RotateArrowEnv = gym.make(
        "RotateArrow-v1",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
    )
    for seed in range(100):
        res = solve(env, seed=seed, debug=False, vis=True)
        print(res)
    env.close()

def solve(env: RotateArrowEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        joint_vel_limits=0.75,
        joint_acc_limits=0.75,
    )

    env = env.unwrapped

    # Get tool OBB and compute grasp pose
    obb = get_actor_obb(env.arrow)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=0.03,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, env.arrow.pose.sp.p)
    grasp_pose_init = deepcopy(grasp_pose)
    offset = sapien.Pose([0, 0.05, 0])
    grasp_pose = grasp_pose * (offset)
    reach_pose_1 = grasp_pose

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    res = None
    approach_pose = sapien.Pose(p=grasp_pose_init.p + np.array([-0.025, 0.0, 0.0]), q=grasp_pose_init.q)
    res = planner.move_to_pose_with_screw(approach_pose)
    if res == -1:
        return res

    # print("sleeping for 10 seconds")
    # sleep(10)

    # # Then, approach down to the grasp pose.
    # hover_pose
    # res = planner.move_to_pose_with_screw(reach_pose_1)
    # if res == -1:
    #     return res
    # print(f"Reach pose: {reach_pose_1}")

    arrow_transform = env.arrow.pose.to_transformation_matrix()[0]
    arrow_x_axis = arrow_transform[:3, 0]  # First column is X axis
    arrow_x_axis = arrow_x_axis / np.linalg.norm(arrow_x_axis)
    arrow_init_x_axis = deepcopy(arrow_x_axis)
    count = 0
    while(np.dot(arrow_x_axis, arrow_init_x_axis) > -0.9): # while angle < 170 degrees
        count += 1
        if count > 20:
            return -1
        # Get arrow's current transformation matrix
        arrow_transform = env.arrow.pose.to_transformation_matrix()[0]
        arrow_x_axis = arrow_transform[:3, 0]  # First column is X axis
        
        # Create rotation matrix where X axis matches arrow's X axis
        # and Z axis points down (for the gripper)
        z_axis = np.array([0, 0, -1])  # gripper points down
        y_axis = np.cross(z_axis, arrow_x_axis)  # compute Y axis
        y_axis = y_axis / np.linalg.norm(y_axis)  # normalize
        x_axis = np.cross(y_axis, z_axis)  # recompute X to ensure orthogonality
        
        # Build rotation matrix
        R = np.column_stack([x_axis, y_axis, z_axis])
        
        # Convert to quaternion (you may need to import)
        from transforms3d.quaternions import mat2quat
        q = mat2quat(R)
        
        # Set pose
        reach_pose_1.set_q(q)
        # Translate in local Y (will follow new orientation)
        reach_pose_1 = reach_pose_1 * sapien.Pose([0.003, 0.03, 0])
        res = planner.move_to_pose_with_screw(reach_pose_1)
        if res == -1: return res
    planner.close()
    return res


if __name__ == "__main__":
    main()
