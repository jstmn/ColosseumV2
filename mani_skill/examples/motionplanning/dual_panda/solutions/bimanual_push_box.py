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
        print(f"\n--- Seed {seed} ---")
        success = solve(env, seed=seed, debug=True, vis=True)
        
        # if success:
        #     print(f"✓ Test passed (seed={seed})")
        # else:
        #     print(f"✗ Test failed (seed={seed})")
        
    env.close()
    print("\n=== All tests completed ===")

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
    
    # Determine which arm is closer to the box
    dist_1 = np.linalg.norm(p1 - box_pos)
    dist_2 = np.linalg.norm(p2 - box_pos)
    closer_arm = 1 if dist_1 <= dist_2 else 2
    
    print(f"Arm 1 distance to box: {dist_1:.3f}")
    print(f"Arm 2 distance to box: {dist_2:.3f}")
    print(f"Using arm {closer_arm} (closer to box)")
    
    FINGER_LENGTH = 0.025
    
    # Get approaching vector along box's z-axis
    obb = get_actor_obb(env.box)
    box_transform = env.box.pose.sp.to_transformation_matrix()
    if hasattr(box_transform, 'cpu'):  # if it's a tensor
        approaching = -box_transform[:3, 2].cpu().numpy()
    else:
        approaching = -np.array(box_transform[:3, 2])
    
    # Get target closing direction from closer arm
    if closer_arm == 1:
        target_closing = env.agent.tcp_1.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    else:
        target_closing = env.agent.tcp_2.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    
    # Compute grasp info
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    
    closing, center = grasp_info["closing"], grasp_info["center"]
    
    # Build grasp pose - approach from above
    planner.close_gripper(arm_index=closer_arm)
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, env.box.pose.sp.p)
    if closer_arm == 2:
        grasp_approach_pose = grasp_pose * sapien.Pose(p=[0, 0.1, 0])
    else:
        grasp_approach_pose = grasp_pose * sapien.Pose(p=[0, 0.16, 0])
        
    # Move closer arm to grasp approach pose
    print(f"\n--- Moving arm {closer_arm} to Step 1 ---")
    result = planner.move_to_pose_with_screw(
        grasp_approach_pose,
        arm_index=closer_arm
    )
    
    if result == -1:
        print(f"Failed to move arm {closer_arm} to Step 1")
        return result
    
    move_front = goal_pos[..., 1] - box_pos[..., 1]
    print(f"✓ Arm {closer_arm} reached Step 1")
    if closer_arm==2:
        grasp_pose = grasp_pose * sapien.Pose(p=[0, move_front+0.05, 0])
    else:
        grasp_pose = grasp_pose * sapien.Pose(p=[0, -move_front+0.05, 0])
    grasp_pose = grasp_pose * sapien.Pose(p=[0, 0, 0])
    print("Calculated Grasp Pose", grasp_pose)
    # Move to grasp pose
    print(f"\n--- Moving arm {closer_arm} to Step 2 ---")
    result = planner.move_to_pose_with_screw(
        grasp_pose,
        arm_index=closer_arm
    )
    
    if result == -1:
        print(f"Failed to move arm {closer_arm} to Step 2")
        return result
    
    print(f"✓ Arm {closer_arm} reached Step 2")
    
    # Close gripper
    print(f"\n--- Closing arm {closer_arm} gripper ---")
    print(f"✓ Arm {closer_arm} gripper closed")
    
    # Move back up
    print(f"\n--- Moving arm {closer_arm} back ---")
    
    grasp_approach_pose = grasp_approach_pose * sapien.Pose(p=[0, 0.05, 0])
    result = planner.move_to_pose_with_screw(
        grasp_approach_pose,
        arm_index=closer_arm
    )
    
    if result == -1:
        print(f"Failed to lift arm {closer_arm}")
        return result
    
    print(f"✓ Arm {closer_arm} lifted successfully")
    # planner.render_wait()
    if env.evaluate()["success"] == torch.tensor(True):
        return result
    
    # Create a push pose at the goal location (same approaching direction)
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, env.box.pose.sp.p)
    if closer_arm == 2:
        push_grasp_pose = grasp_pose * sapien.Pose(p=[-0.1, 0, 0])
    else:
        push_grasp_pose = grasp_pose * sapien.Pose(p=[0.1, 0, 0])
        
    result = planner.move_to_pose_with_screw(
        push_grasp_pose,
        arm_index=closer_arm
    )
    
    if result == -1:
        print(f"Failed to lift arm {closer_arm}")
        return result

    # Move to grasp pose
    print(f"\n--- Moving arm {closer_arm} to Step 3 ---")
    move_side = goal_pos[..., 0] - box_pos[..., 0]
    print(move_side)
    if closer_arm == 2:
        grasp_pose = grasp_pose * sapien.Pose(p=[-move_side-0.05, 0, 0])
    else:
        grasp_pose = grasp_pose * sapien.Pose(p=[move_side+0.05, 0, 0])
    
    result = planner.move_to_pose_with_screw(
        grasp_pose,
        arm_index=closer_arm
    )
    
    if result == -1:
        print(f"Failed to move arm {closer_arm} to Step 3")
        return result
    
    return result

if __name__ == "__main__":
    main()