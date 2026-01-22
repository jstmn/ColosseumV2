import gymnasium as gym
import numpy as np
import sapien
import time
from mani_skill.examples.motionplanning.dual_panda.motionplanner import DualPandaMotionPlanningSolver
from mani_skill.envs.tasks import DualArmPenCapEnv
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb

def main():
    """
    Test the dual panda motion planner with various scenarios.
    """
    env:DualArmPenCapEnv = gym.make(
        'DualArmPenCap-v1',
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

def solve(env:DualArmPenCapEnv, seed, debug, vis):
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
    obb_pen = get_actor_obb(env.pen)
    obb_cap = get_actor_obb(env.cap)
    
    approaching = np.array([0, 0, -1])
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.agent.tcp_1.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # we can build a simple grasp pose using this information for Panda
    grasp_info = compute_grasp_info_by_obb(
        obb_pen,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pen_pose = env.agent.build_grasp_pose(approaching, closing, env.pen.pose.sp.p)

    target_closing = env.agent.tcp_2.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # we can build a simple grasp pose using this information for Panda
    grasp_info = compute_grasp_info_by_obb(
        obb_cap,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_cap_pose = env.agent.build_grasp_pose(approaching, closing, env.cap.pose.sp.p)

    grasp_cap_pose = grasp_cap_pose*sapien.Pose(p=[-0.08,0,0.03])
    grasp_pen_pose = grasp_pen_pose*sapien.Pose(p=[0,0,0.03])
    
    grasp_cap_approach_pose = grasp_cap_pose*sapien.Pose(p=[0,0,-0.1])
    grasp_pen_approach_pose = grasp_pen_pose*sapien.Pose(p=[0,0,-0.1])

    result = planner.move_to_pose_pair_with_RRTConnect(
        grasp_pen_approach_pose,  # left
        grasp_cap_approach_pose
    )

    if result == -1:
        print("Failed grasp_approach")
        return result
    
    # viewer = planner.base_env.render_human()
    # while True:
    #     if viewer.window.key_down("c"):
    #         break
    #     planner.base_env.render_human()
    
    result = planner.move_to_pose_pair_with_screw(
        grasp_pen_pose,  # left
        grasp_cap_pose
    )

    if result==-1:
        print("Failed grasp")
        return result
    
    
    planner.close_gripper(arm_index=1, t=10)
    planner.close_gripper(arm_index=2, t=10)
    
    lift_pen_pose = sapien.Pose(p=[-0.171,0.124,1],q=grasp_pen_pose.q)
    lift_cap_pose = sapien.Pose(p=[-0.181,0.07,1.0],q=grasp_cap_pose.q)
    lift_cap_pose = lift_cap_pose * sapien.Pose(p=[0,-0.01,-0.0])
    lift_cap_appoach = lift_cap_pose*sapien.Pose(p=[0.3,0,0])
    
    result = planner.move_to_pose_pair_with_screw(
        lift_pen_pose,
        lift_cap_appoach
    )
    
    if result == -1:
        print("Failed to lift")
        return result
    
    # Put cap
    result = planner.move_to_pose_with_screw(
        lift_cap_pose,
        arm_index=1
    )
    
    if result == -1:
        print("Failed to lift")
        return result
    
    planner.render_wait()
    return result


if __name__ == "__main__":
    main()