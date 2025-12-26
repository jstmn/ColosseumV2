import gymnasium as gym
import numpy as np
import sapien
import time
from mani_skill.examples.motionplanning.dual_panda.motionplanner import DualPandaMotionPlanningSolver
from mani_skill.envs.tasks import DualArmEmptyEnv

def main():
    """
    Test the dual panda motion planner with various scenarios.
    """
    env:DualArmEmptyEnv = gym.make(
        'DualArmEmpty-v0',
        obs_mode='none',
        control_mode="pd_joint_pos",  # Use pd_joint_pos for motion planning
        render_mode='human',  # Use 'human' for visualization
    )
    print("=== Testing Dual Panda Motion Planner ===\n")
    
    # Run multiple test scenarios
    test_scenarios = [
        # ("Independent Arm Motion", test_independent_motion),
        # ("Synchronized Dual Arm Motion", test_synchronized_motion),
        # ("Gripper Control", test_gripper_control),
        ("Sequential Pick and Place", test_sequential_tasks),
    ]
    
    for scenario_name, test_func in test_scenarios:
        print(f"\n{'='*60}")
        print(f"Test: {scenario_name}")
        print(f"{'='*60}")
        
        for seed in range(10):  # Test with 3 different seeds
            print(f"\n--- Seed {seed} ---")
            success = test_func(env, seed=seed, debug=True, vis=True)
            
            if success:
                print(f"✓ Test passed (seed={seed})")
            else:
                print(f"✗ Test failed (seed={seed})")
                
        
        # Ask user if they want to continue
        if input("\nContinue to next test? (y/n): ").lower() != 'y':
            break
    
    env.close()
    print("\n=== All tests completed ===")

def test_independent_motion(env, seed=None, debug=False, vis=False):
    """Test moving each arm independently."""
    env.reset(seed=seed)
    if vis:
        env.unwrapped.render_human()
    planner = DualPandaMotionPlanningSolver(
        env, 
        debug=debug, 
        vis=vis,
        print_env_info=True
    )
    if vis:
        env.unwrapped.render_human()
    try:
        # Get current poses
        tcp_1_pose = env.unwrapped.agent.tcp_1.pose
        tcp_2_pose = env.unwrapped.agent.tcp_2.pose
        
        # IMPORTANT: Extract as numpy and make copies
        if hasattr(tcp_1_pose.p, 'cpu'):
            # PyTorch tensors
            tcp_1_p = tcp_1_pose.p.cpu().numpy().copy().flatten()
            tcp_1_q = tcp_1_pose.q.cpu().numpy().copy().flatten()
            tcp_2_p = tcp_2_pose.p.cpu().numpy().copy().flatten()
            tcp_2_q = tcp_2_pose.q.cpu().numpy().copy().flatten()
        else:
            # Already numpy
            tcp_1_p = np.array(tcp_1_pose.p).flatten().copy()
            tcp_1_q = np.array(tcp_1_pose.q).flatten().copy()
            tcp_2_p = np.array(tcp_2_pose.p).flatten().copy()
            tcp_2_q = np.array(tcp_2_pose.q).flatten().copy()
        
        print(f"Initial TCP1 (right) p={tcp_1_p}, q={tcp_1_q}")
        print(f"Initial TCP2 (left) p={tcp_2_p}, q={tcp_2_q}")
        
        # Open both grippers
        planner.open_gripper(arm_index=None)
        
        # Move arm 1 - TINY movement, SAME orientation
        print("\n1. Moving right arm by +0.05m in X...")
        
        target_p1 = tcp_1_p + np.array([0.1, 0.1, 0.1])
        target_q1 = tcp_1_q.copy()  # Explicit copy
        
        print(f"  Target p={target_p1}, q={target_q1}")
        
        # Verify quaternion is valid
        if np.abs(np.linalg.norm(target_q1) - 1.0) > 0.01:
            print(f"  WARNING: Quaternion not normalized! norm={np.linalg.norm(target_q1)}")
            target_q1 = target_q1 / np.linalg.norm(target_q1)
        
        target_pose_1 = sapien.Pose(p=target_p1, q=target_q1)
        
        result = planner.move_arm_to_pose_with_RRTConnect(
            target_pose_1, 
            arm_index=1,
            refine_steps=10
        )
        
        if result == -1:
            print("Failed to move arm 1")
            return False
        
        print("\n✓ Independent motion test completed successfully")
        time.sleep(5)
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
def test_synchronized_motion(env, seed=None, debug=False, vis=False):
    """Test moving both arms simultaneously."""
    env.reset(seed=seed)
    if vis:
        env.unwrapped.render_human()
    planner = DualPandaMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        print_env_info=True
    )
    if vis:
        env.unwrapped.render_human()
    try:
        # Get current poses
        tcp_1_pose = env.unwrapped.agent.tcp_1_pose
        tcp_2_pose = env.unwrapped.agent.tcp_2_pose
        
        # --- FIX: Convert PyTorch Tensors to Flat Numpy Arrays ---
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
            
        # Open both grippers
        planner.open_gripper(arm_index=None)
        
        # Move both arms together
        print("\nMoving both arms simultaneously...")
        
        # Define target poses (move both arms forward and up)
        # Now we are adding numpy array to numpy array (Safe)
        left_target = sapien.Pose(
            p=p2 + np.array([0.15, 0.0, 0.1]),
            q=q2
        )
        
        right_target = sapien.Pose(
            p=p1 + np.array([0.15, 0.0, 0.1]),
            q=q1
        )
        
        result = planner.move_to_pose_pair_with_RRTConnect(
            left_target,
            right_target,
            refine_steps=10
        )
        
        if result == -1:
            print("Failed to move both arms")
            return False
        
        print("\n✓ Synchronized motion test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Error during synchronized motion test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
def test_gripper_control(env, seed=None, debug=False, vis=False):
    """Test gripper open/close for both arms."""
    env.reset(seed=seed)
    
    planner = DualPandaMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        print_env_info=True
    )
    
    try:
        print("\n1. Opening both grippers...")
        planner.open_gripper(arm_index=None, t=10)
        
        print("\n2. Closing right gripper (arm_1)...")
        planner.close_gripper(arm_index=1, t=10)
        
        print("\n3. Closing left gripper (arm_2)...")
        planner.close_gripper(arm_index=2, t=10)
        
        print("\n4. Opening both grippers again...")
        planner.open_gripper(arm_index=None, t=10)
        
        print("\n5. Closing both grippers...")
        planner.close_gripper(arm_index=None, t=10)
        
        print("\n✓ Gripper control test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Error during gripper control test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sequential_tasks(env, seed=None, debug=False, vis=False):
    """Test a sequence of coordinated movements."""
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

        # 1. Open grippers
        print("\n1. Opening grippers...")
        planner.open_gripper(arm_index=None)
        
        # 2. Move to pre-grasp positions (approach)
        print("\n2. Moving to approach positions...")
        approach_1 = sapien.Pose(
            p=np.array([-0.2, -0.141, 0.9]),
            q=q1
        )
        approach_2 = sapien.Pose(
            p=np.array([0.2, 0.141, 0.9]),
            q=q2
        )
        
        result = planner.move_to_pose_pair_with_RRTConnect(
            approach_2,  # left
            approach_1,  # right
            refine_steps=5
        )
        
        if result == -1:
            print("Failed to reach approach positions")
            return False
        
        # 3. Move to grasp positions (lower)
        print("\n3. Moving to grasp positions...")
        grasp_1 = sapien.Pose(
            p=np.array([-0.2, -0.141, 0.83]),
            q=q1
        )
        grasp_2 = sapien.Pose(
            p=np.array([0.2, 0.141, 0.9]),
            q=q2
        )
        
        result = planner.move_to_pose_pair_with_RRTConnect(
            grasp_2,  # left
            grasp_1,  # right
            refine_steps=5
        )
        
        if result == -1:
            print("Failed to reach grasp positions")
            return False
        
        # 4. Close grippers
        print("\n4. Closing grippers...")
        planner.close_gripper(arm_index=1, t=10)
        
        print("\n5. Lifting...")
        lift_1 = sapien.Pose(
            p=np.array([-0.333, 0.04, 1.1]),
            q=np.array([-0.484,0.476,0.523,0.515])
        )
        
        lift_2 = sapien.Pose(
            p=np.array([-0.333, 0.14, 1.1]),
            q=np.array([0, 0, 0.73, -0.683])
        )
        
        result = planner.move_to_pose_pair_with_RRTConnect(
            lift_2,  # left
            lift_1,  # right
            refine_steps=5
        )
        
        if result == -1:
            print("Failed to lift")
            return False
        
        # 5. Lift up
        print("\n5. Lifting...")
        lift_1 = sapien.Pose(
            p=np.array([-0.333, 0.04, 1.1]),
            q=np.array([-0.484,0.476,0.523,0.515])
        )
        
        lift_2 = sapien.Pose(
            p=np.array([-0.333, 0.04, 1.1]),
            q=np.array([0, 0, 0.73, -0.683])
        )
        
        result = planner.move_to_pose_pair_with_RRTConnect(
            lift_2,  # left
            lift_1,  # right
            refine_steps=5
        )
        
        if result == -1:
            print("Failed to lift")
            return False
        
        # 6. Open grippers
        print("\n6. Releasing...")
        planner.close_gripper(arm_index=2, t=10)
        planner.open_gripper(arm_index=1, t=10)
        print("\n✓ Sequential tasks test completed successfully")
        
        print("\n3. Moving to grasp positions...")
        grasp_1 = sapien.Pose(
            p=np.array([-0.2, -0.141, 0.83]),
            q=q1
        )
        grasp_2 = sapien.Pose(
            p=np.array([0.2, 0.141, 0.9]),
            q=q2
        )
        
        result = planner.move_to_pose_pair_with_RRTConnect(
            grasp_2,  # left
            grasp_1,  # right
            refine_steps=5
        )
        
        return True
        
    except Exception as e:
        print(f"✗ Error during sequential tasks test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()