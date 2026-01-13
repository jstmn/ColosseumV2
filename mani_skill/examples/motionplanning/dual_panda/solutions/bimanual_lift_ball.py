import gymnasium as gym
import numpy as np
import sapien
import time
from mani_skill.examples.motionplanning.dual_panda.motionplanner import DualPandaMotionPlanningSolver
from mani_skill.envs.tasks import DualArmLiftBallEnv
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb

def main():
    """
    Test the dual panda motion planner with various scenarios.
    """
    env:DualArmLiftBallEnv = gym.make(
        'DualArmLiftBall-v0',
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

def solve(env:DualArmLiftBallEnv, seed, debug, vis):
    env.reset(seed=seed)
    
    planner = DualPandaMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        print_env_info=True
    )
    
        
    try:
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
        obb = get_actor_obb(env.ball)

        grasp_1_pose = env.ball.pose*sapien.Pose(p=[0,0,-0.03],q=[0,1,0,0])
        
        # grasp_2_pose = env.ball.pose*sapien.Pose(p=[0,-0.08,0],q=[0.707,0.707,0,0])
        
        grasp_1_approach_pose = grasp_1_pose*sapien.Pose(p=[0,0,-0.1])
        # grasp_2_approach_pose = grasp_2_pose*sapien.Pose(p=[0, 0, -0.1])
        result = planner.move_arm_to_pose_with_RRTConnect(
            grasp_1_approach_pose,  # left
            arm_index=1
        )

        if result==-1:
            print("Failed grasp_approach")
            return False
        # grasp_1_pose = grasp_1_pose*sapien.Pose(p=[0,0,0.05])
        # grasp_2_pose = grasp_2_pose*sapien.Pose(p=[0,0,0.05])
        result = planner.move_to_pose_with_screw(
            grasp_1_pose,  # left
            arm_index=1
        )

        if result==-1:
            print("Failed grasp_approach")
            return False
        
        planner.close_gripper(arm_index=1, t=10)
        # planner.close_gripper(arm_index=2, t=10)
        
        print("\n5. Lifting...")
        lift_1 = grasp_1_pose*sapien.Pose(p=[0,0,-0.2],q=[0.707, 0.707, 0, 0])
        lift_2 = grasp_1_pose*sapien.Pose(p=[0,0,-0.2],q=[-0.5, 0.5, 0.5, 0.5])
        lift_2 = lift_2*sapien.Pose(p=[0,0,-0.1])
        result = planner.move_to_pose_pair_with_RRTConnect(
            lift_2,  # left
            lift_1
            # refine_steps=5
        )
        lift_2 = lift_2*sapien.Pose(p=[0,0,0.1])
        result = planner.move_to_pose_with_screw(
            lift_2,
            arm_index=2
        )
        if result == -1:
            print("Failed to lift")
            return False
        
        # 5. Move
        print("\n5. Move...")
        
        
        lift_1 = lift_1*sapien.Pose(p=[-0.3,0,0])
        lift_2 = lift_2*sapien.Pose(p=[0,-0.3,0])
        
        result = planner.move_to_pose_pair_with_screw(
            lift_2,  # left
            lift_1,  # right
            # refine_steps=5
        )
        
        if result == -1:
            print("Failed to lift")
            return False
        
        # Place down
        
        lift_1 = lift_1*sapien.Pose(p=[0,0.1,0])
        lift_2 = lift_2*sapien.Pose(p=[0.1,0,0])
        
        result = planner.move_to_pose_pair_with_screw(
            lift_2,  # left
            lift_1,  # right
            # refine_steps=5
        )
        
        if result == -1:
            print("Failed to lift")
            return False
        
        return True
    
    except Exception as e:
        print("Exception during Motion Planning:", e)
        return False    

if __name__ == "__main__":
    main()