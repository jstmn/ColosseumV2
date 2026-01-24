import gymnasium as gym
import numpy as np
import sapien
import time
from mani_skill.examples.motionplanning.dual_panda.motionplanner import DualPandaMotionPlanningSolver
from mani_skill.envs.tasks import DualPandaPushBoxEnv
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb
import torch

def main():
    """
    Test the dual panda motion planner with various scenarios.
    """
    env:DualPandaPushBoxEnv = gym.make(
        'DualArmPushBox-v1',
        obs_mode='none',
        control_mode="pd_joint_pos",  # Use pd_joint_pos for motion planning
        render_mode='human',  # Use 'human' for visualization
    )
    print("=== Testing Dual Panda Motion Planner ===\n")

    for seed in range(10):  # Test with 3 different seeds
        success = solve(env, seed=seed, debug=True, vis=True)
        
    env.close()

def solve(env:DualPandaPushBoxEnv, seed, debug, vis):
    env.reset(seed=seed)
    
    planner = DualPandaMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        print_env_info=True
    )

    env = env.unwrapped
    
    # Get TCP poses
    tcp_1_pose = env.agent.tcp_1_pose
    tcp_2_pose = env.agent.tcp_2_pose
    
    # Extract positions for both arms
    if hasattr(tcp_1_pose.p, 'cpu'):
        p1 = tcp_1_pose.p.cpu().numpy().flatten()
        p2 = tcp_2_pose.p.cpu().numpy().flatten()
    else:
        p1 = np.array(tcp_1_pose.p).flatten()
        p2 = np.array(tcp_2_pose.p).flatten()
    
    # Get box and goal positions
    box_pos = env.box.pose.sp.p
    goal_pos = env.goal_region.pose.sp.p
    
    if hasattr(box_pos, 'cpu'):
        box_pos = box_pos.cpu().numpy()
    else:
        box_pos = np.array(box_pos)
    
    if hasattr(goal_pos, 'cpu'):
        goal_pos = goal_pos.cpu().numpy()
    else:
        goal_pos = np.array(goal_pos)
    
    FINGER_LENGTH = 0.025
    
    # Get approaching vector along box's z-axis
    obb = get_actor_obb(env.box)
    box_transform = env.box.pose.sp.to_transformation_matrix()
    if hasattr(box_transform, 'cpu'):  # if it's a tensor
        approaching = -box_transform[:3, 2].cpu().numpy()
    else:
        approaching = -np.array(box_transform[:3, 2])
    
    # Get target closing direction from closer arm
    target_closing_1 = env.agent.tcp_1.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    target_closing_2 = env.agent.tcp_2.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    
    # Compute grasp info
    grasp_info_1 = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing_1,
        depth=FINGER_LENGTH,
    )
    
    closing, center = grasp_info_1["closing"], grasp_info_1["center"]
    
    # Build grasp pose - approach from above
    grasp_pose_1 = env.agent.build_grasp_pose(approaching, closing, env.box.pose.sp.p)

    grasp_info_2 = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing_2,
        depth=FINGER_LENGTH,
    )
    
    closing, center = grasp_info_2["closing"], grasp_info_2["center"]
    
    # Build grasp pose - approach from above
    grasp_pose_2 = env.agent.build_grasp_pose(approaching, closing, env.box.pose.sp.p)
    
    grasp_pose_1 = grasp_pose_1 * sapien.Pose(p=[-0.06, 0.155,0])
    grasp_pose_2 = grasp_pose_2 * sapien.Pose(p=[-0.06, -0.155,0])
    grasp_pose_1_approach = grasp_pose_1 * sapien.Pose(p=[0, 0, -0.1])
    grasp_pose_2_approach = grasp_pose_2 * sapien.Pose(p=[0, 0, -0.1])
    
    # Move closer arm to grasp approach pose
    result = planner.move_to_pose_pair_with_RRTConnect(
        grasp_pose_2_approach,
        grasp_pose_1_approach
    )

    if result == -1:
        return result
    
    result = planner.move_to_pose_pair_with_RRTConnect(
        grasp_pose_2,
        grasp_pose_1
    )
    
    if result == -1:
        return result

    final_pose_1 = sapien.Pose(p=[-0.25, -0.06, grasp_pose_1.p[2]],q=[0, 0.707, 0.707, 0])
    final_pose_2 = sapien.Pose(p=[-0.25, 0.06, grasp_pose_2.p[2]], q=[0, 0.707, -0.707, 0])
    
    result = planner.move_to_pose_pair_with_screw(
        final_pose_2,
        final_pose_1
    )
    
    if result == -1:
        return result
        
    return result

if __name__ == "__main__":
    main()