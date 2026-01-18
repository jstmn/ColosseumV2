import gymnasium as gym
import numpy as np
import sapien
import time

import trimesh
from mani_skill.examples.motionplanning.dual_panda.motionplanner import DualPandaMotionPlanningSolver
from mani_skill.envs.tasks import DualArmPickBottleEnv
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb

def main():
    """
    Test the dual panda motion planner with various scenarios.
    """
    env:DualArmPickBottleEnv = gym.make(
        'DualArmPickBottle-v0',
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

def solve(env:DualArmPickBottleEnv, seed, debug, vis):
    env.reset(seed=seed)
    
    planner = DualPandaMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        print_env_info=True
    )
    
        
    # Get initial poses
    tcp_1_pose = env.unwrapped.agent.tcp_1_pose
    tcp_2_pose = env.unwrapped.agent.tcp_2_pose

    # 1. Handle Position (p)
    if hasattr(tcp_1_pose.p, 'cpu'):
        p1 = tcp_1_pose.p.cpu().numpy().flatten()
        p2 = tcp_2_pose.p.cpu().numpy().flatten()
    else:
        p1 = np.array(tcp_1_pose.p).flatten()
        p2 = np.array(tcp_2_pose.p).flatten()

    # 2. Handle Orientation (q)
    if hasattr(tcp_1_pose.q, 'cpu'):
        q1 = tcp_1_pose.q.cpu().numpy().flatten()
        q2 = tcp_2_pose.q.cpu().numpy().flatten()
    else:
        q1 = np.array(tcp_1_pose.q).flatten()
        q2 = np.array(tcp_2_pose.q).flatten()


    FINGER_LENGTH = 0.025
    env = env.unwrapped
    obb = get_actor_obb(env.obj)
    approaching = np.array([0, 1, 0])
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.agent.tcp_1.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # we can build a simple grasp pose using this information for Panda
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, env.obj.pose.sp.p)

    grasp_1_approach_pose = grasp_pose*sapien.Pose(p=[0.15,0,-0.1])
    grasp_1_approach_pose.q = np.array([-0.5,0.5,0.5,0.5])
    res = planner.move_to_pose_with_screw(
        grasp_1_approach_pose,  # left
        arm_index=1
    )

    if res==-1:
        print("Failed grasp_approach")
        return res
    grasp_pose = grasp_pose*sapien.Pose(p=[0.14,0,0.00])
    grasp_pose.q = np.array([-0.5,0.5,0.5,0.5])
    res = planner.move_to_pose_with_screw(
        grasp_pose,  # left
        arm_index=1
    )

    if res==-1:
        print("Failed grasp_approach")
        return res
    
    planner.close_gripper(arm_index=1, t=10)
    
    print("\n5. Lifting...")
    lift_1 = sapien.Pose(
        p=np.array([-0.333, -0.10, 1.5]),
        q=np.array([-0.5,0.5,0.5,0.5])
    )
    
    lift_2 = sapien.Pose(
        p=np.array([-0.333, 0.10, 1.48]),
        q=np.array([0.5,0.5,0.5,-0.5])
    )
    
    res = planner.move_to_pose_pair_with_RRTConnect(
        lift_2,  # left
        lift_1
        # refine_steps=5
    )
    
    if res == -1:
        print("Failed to lift")
        return res
    
    # 5. Lift up
    print("\n5. Lifting...")
    lift_1 = sapien.Pose(
        p=np.array([-0.333, 0.04, 1.5]),
        q=np.array([-0.5,0.5,0.5,0.5])
    )
    
    lift_2 = sapien.Pose(
        p=np.array([-0.333, 0.04, 1.48]),
        q=np.array([0.5,0.5,0.5,-0.5])
    )
    
    res = planner.move_to_pose_pair_with_screw(
        lift_2,  # left
        lift_1,  # right
        # refine_steps=5
    )
    
    if res == -1:
        print("Failed to lift")
        return res
    
            # 6. Open grippers
    print("\n6. Releasing...")
    planner.close_gripper(arm_index=2, t=5)
    planner.open_gripper(arm_index=1, t=5)
    lift_2 = sapien.Pose(
        p=np.array([-0.333, 0.20, 1.48]),
        q=np.array([0.5,0.5,0.5,-0.5])
    )
    res = planner.move_to_pose_with_screw(
        lift_2,
        arm_index=2
    )
    
    if res == -1:
        print("Failed to lift")
        return res
    
    grasp_1 = sapien.Pose(
        p=np.array([-0.2, -0.141, 1]),
        q=grasp_pose.q
    )
    grasp_2 = sapien.Pose(
        p=np.array([0.1, 0.141, 1]),
        q=lift_2.q
    )

    res = planner.move_to_pose_pair_with_screw(
        grasp_2,  # left
        grasp_1,  # right
        # refine_steps=5
    )
    
    return res

if __name__ == "__main__":
    main()