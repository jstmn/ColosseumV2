import gymnasium as gym
import numpy as np
import sapien
import time
from mani_skill.examples.motionplanning.dual_panda.motionplanner import DualPandaMotionPlanningSolver
from mani_skill.envs.tasks import TwoRobotStackCube
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb

def main():
    """
    Test the dual panda motion planner with various scenarios.
    """
    env:TwoRobotStackCube = gym.make(
        "DualArmStackCube-v1',
        obs_mode='none',
        control_mode="pd_joint_pos",  # Use pd_joint_pos for motion planning
        render_mode='human',  # Use 'human' for visualization
    )
    print("=== Testing Dual Panda Motion Planner ===\n")

    for seed in range(10):  # Test with 3 different seeds
        print(f"\n--- Seed {seed} ---")
        success = solve(env, seed=seed, debug=True, vis=True)
        
        if success:
            print(f"✓ Test passed (seed={seed})")
        else:
            print(f"✗ Test failed (seed={seed})")
        
    env.close()
    print("\n=== All tests completed ===")

def solve(env:TwoRobotStackCube, seed, debug, vis):
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

    # retrieves the object oriented bounding box (trimesh box object)
    obb_A = get_actor_obb(env.cubeA)
    obb_B = get_actor_obb(env.cubeB)
    
    approaching = np.array([0, 0, -1])
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.agent.tcp_1.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # we can build a simple grasp pose using this information for Panda
    grasp_info = compute_grasp_info_by_obb(
        obb_A,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_A_pose = env.agent.build_grasp_pose(approaching, closing, env.cubeA.pose.sp.p)

    target_closing = env.agent.tcp_2.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # we can build a simple grasp pose using this information for Panda
    grasp_info = compute_grasp_info_by_obb(
        obb_B,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_B_pose = env.agent.build_grasp_pose(approaching, closing, env.cubeB.pose.sp.p)

    place_A_pose = sapien.Pose(p=np.array(env.goal_region.pose.p[0]), q=grasp_A_pose.get_q())
    place_A_pose = place_A_pose * sapien.Pose(p=[0,0,-0.1])
    
    place_B_pose = sapien.Pose(p=np.array(env.goal_region.pose.p[0]), q=grasp_B_pose.get_q())
    place_B_pose = place_B_pose * sapien.Pose(p=[0,0,-0.1])

    grasp_A_approach_pose = grasp_A_pose*sapien.Pose(p=[0,0,-0.1])
    grasp_B_approach_pose = grasp_B_pose*sapien.Pose(p=[0,0,-0.1])
    
    result = planner.move_to_pose_pair_with_screw(
        grasp_B_approach_pose,
        grasp_A_approach_pose  # left
    )

    if result==-1:
        print("Failed grasp_approach")
        return result
    
    result = planner.move_to_pose_pair_with_screw(
        grasp_B_pose,
        grasp_A_pose  # left
    )

    if result==-1:
        print("Failed grasp_approach")
        return result
    
    planner.close_gripper(arm_index=1, t=10)
    planner.close_gripper(arm_index=2, t=10)
    
    place_A_lift_pose = place_A_pose * sapien.Pose(p=[0,0,-0.05])
    # life A above target
    result = planner.move_to_pose_with_screw(
        place_A_lift_pose,
        arm_index=1
    )
    
    if result == -1:
        print("Failed to lift")
        return result
    
    # Place A
    result = planner.move_to_pose_with_screw(
        place_A_pose,
        arm_index=1
    )
    
    if result == -1:
        print("Failed to lift")
        return result
    
    planner.open_gripper(arm_index=1, t=10)

    final_A_pose = sapien.Pose(p=[-0.4,-0.254,1.018],q=[0.138,0.694,0.694,-0.138])
    result = planner.move_to_pose_with_screw(
        final_A_pose,  # left
        arm_index=1
        # refine_steps=5
    )
    
    if result == -1:
        print("Failed to Reset arm A")
        return result
    
    # Place B
    place_B_pose = place_B_pose*sapien.Pose(p=[0,0,-0.1])
    result = planner.move_to_pose_with_screw(
        place_B_pose,
        arm_index=2
    )
    
    if result == -1:
        print("Failed to lift")
        return result
    
    planner.open_gripper(arm_index=2, t=10)
    
    return result


if __name__ == "__main__":
    main()