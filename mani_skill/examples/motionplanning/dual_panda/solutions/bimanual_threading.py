import gymnasium as gym
import numpy as np
import sapien
import time
from mani_skill.examples.motionplanning.dual_panda.motionplanner import DualPandaMotionPlanningSolver
from mani_skill.envs.tasks import DualPandaThreadingEnv
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb
from mani_skill.utils.structs.pose import Pose

def main():
    """
    Test the dual panda motion planner with various scenarios.
    """
    env:DualPandaThreadingEnv = gym.make(
        'DualArmThreading-v1',
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

def solve(env:DualPandaThreadingEnv, seed, debug, vis):
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
    obb_needle = get_actor_obb(env.needle)
    needle_transform = env.needle.pose.sp.to_transformation_matrix()
    # Along the x axis of the needle
    if hasattr(needle_transform, 'cpu'):  # if it's a tensor
        approaching = -needle_transform[:3, 2].cpu().numpy()
    else:
        approaching = -np.array(needle_transform[:3, 2])

    target_closing = env.agent.tcp_1.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb_needle,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_1_pose = env.agent.build_grasp_pose(approaching, closing, env.needle.pose.sp.p)
    grasp_1_pose = grasp_1_pose*sapien.Pose(p=[0.03,0,0.04])

    # ==== Grasp the peg
    approach_1_pose = grasp_1_pose*sapien.Pose(p=[0, 0, -0.1])
    result = planner.move_to_pose_with_screw(
        approach_1_pose,  # left
        arm_index=1
    )
    if result==-1:
        return result


    result = planner.move_to_pose_with_screw(
        grasp_1_pose,  # left
        arm_index=1
    )
    if result==-1:
        return result
    # ==== Grasp the peg
    
    # ==== Grasp the ring tripod
    obb_tripod = get_actor_obb(env.ring_tripod)
    tripod_transform = env.ring_tripod.pose.sp.to_transformation_matrix()
    # Along the x axis of the tripod
    if hasattr(tripod_transform, 'cpu'):  # if it's a tensor
        approaching = -tripod_transform[:3, 1].cpu().numpy()
    else:
        approaching = -np.array(tripod_transform[:3, 1])

    target_closing = env.agent.tcp_2.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb_tripod,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    
    lift_1_pose = grasp_1_pose*sapien.Pose(p=[0,0,-0.1])
    planner.close_gripper(arm_index=1, t=10)
    grasp_2_pose = env.agent.build_grasp_pose(approaching, closing, env.ring_tripod.pose.sp.p)
    grasp_2_pose = grasp_2_pose*sapien.Pose(p=[0,-0.05,-0.15])
    grasp_2_pose.q = [0,0,0.707,-0.707]
    result = planner.move_to_pose_with_screw(
        lift_1_pose,
        arm_index=1
    )
    if result==-1:
        return result
    
    grasp_2_pose = grasp_2_pose*sapien.Pose(p=[0,0,0.12])
    result = planner.move_to_pose_with_screw(
        grasp_2_pose,
        arm_index=2
    )
    if result==-1:
        return result
    planner.close_gripper(arm_index=2, t=10)
    # ==== Grasp the ring tripod
    
    # lift_2_pose = sapien.Pose(p=[-0.0, -0.011, 1.186-0.17],q=[0,0,0.707,-0.707])
    lift_2_pose = sapien.Pose(p=[-0.228, -0.011, 1.186-0.17],q=[0,0,0.707,-0.707])
    lift_2_pose = lift_2_pose*sapien.Pose(p=[0,0.0,-0.03])
    lift_1_pose = sapien.Pose(p=[-0.24, -0.011, 1.36-0.1], q=[0.707,-0.707,0,0])
    # lift_1_approach_pose = lift_1_pose*sapien.Pose(p=[-0.4, 0, 0])
    lift_1_approach_pose = lift_1_pose*sapien.Pose(p=[-0.1, 0, 0])
    
    result = planner.move_to_pose_with_screw(
        lift_1_approach_pose,
        arm_index=1
    )
    
    if result==-1:
        return result

    result = planner.move_to_pose_with_screw(
        lift_2_pose,
        arm_index=2
    )
    if result==-1:
        return result

    result = planner.move_to_pose_with_screw(
        lift_1_pose,
        arm_index=1
    )
    if result==-1:
        return result

    return result

if __name__ == "__main__":
    main()




