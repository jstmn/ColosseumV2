import gymnasium as gym
import numpy as np
import sapien
import time
from mani_skill.examples.motionplanning.dual_panda.motionplanner import DualPandaMotionPlanningSolver
from mani_skill.envs.tasks import DualArmDrawerPlaceEnv
from scipy.spatial.transform import Rotation as R
import sapien.core as sapien
import sapien.physx as physx  # REQUIRED for SAPIEN 3 Physics classes

def main():
    """
    Test the dual panda motion planner with various scenarios.
    """
    env:DualArmDrawerPlaceEnv = gym.make(
        'DualArmDrawerPlace-v0',
        obs_mode='none',
        control_mode="pd_joint_pos",  # Use pd_joint_pos for motion planning
        render_mode='human',  # Use 'human' for visualization
    )
    # increase_handle_friction(env)
    # debug_collision_properties(env)
    for seed in range(10):  # Test with 3 different seeds
        print(f"\n--- Seed {seed} ---")
        success = solve(env, seed=seed, debug=True, vis=True)            
        print(f"Result: {'Success' if success else 'Failure'}")
        # env.reset()
    env.close()

def solve(env, seed, debug=False, vis=False):
    env.reset(seed=seed)
    if vis: 
        env.unwrapped.render_human()
    planner = DualPandaMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        print_env_info=True
    )
    
    try:
        tcp_1_pose = env.unwrapped.agent.tcp_1_pose
        tcp_2_pose = env.unwrapped.agent.tcp_2_pose
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
        lift_1 = sapien.Pose(
            p=np.array([0.04, -0.017, 1.26]),
            q=np.array([0.5,0.5,0.5,0.5])
        )
        
        lift_2 = sapien.Pose(
            p=np.array([-0.2, -0.141, 0.83]),
            q=np.array([0, -0.707, 0.707, 0])
        )
        
        ready_lift_1 = lift_1 * sapien.Pose(p=[0,0,-0.1])
        ready_lift_2 = lift_2 * sapien.Pose(p=[0,0,-0.1])
        result = planner.move_to_pose_pair_with_RRTConnect(
            ready_lift_1,  # left
            ready_lift_2
        )
        
        if result == -1:
            print("Failed to Ready lift")
            return False
        
        result = planner.move_to_pose_pair_with_screw(
            lift_1,  # left
            lift_2
        )
        
        if result == -1:
            print("Failed to lift")
            return False

        planner.close_gripper(arm_index=1)
        planner.close_gripper(arm_index=2)
        
        pull_1 = lift_1 * sapien.Pose(p=[0, 0, -0.3])
        hold_2 = pull_1 * sapien.Pose(p=[-0.3, 0.1, 0.1])
        hold_2.set_q(lift_2.q)
        
        result = planner.move_to_pose_with_screw(
            hold_2,
            arm_index=1
        )
        
        if result == -1:
            print("Failed to Hold")
            return False
        
        result = planner.move_to_pose_with_screw(
            pull_1,
            arm_index=2
        )
        
        if result == -1:
            print("Failed to pull")
            return False
        
        release_2 = pull_1 * sapien.Pose(p=[-0.1, 0.15, 0.2])
        release_2.set_q(np.array([0,0,1,0]))
        result = planner.move_to_pose_with_screw(
            release_2,
            arm_index=1
        )
        
        if result == -1:
            print("Failed to Release")
            return False
        
        planner.open_gripper(arm_index=1)
        
        result = planner.move_to_pose_with_screw(
            lift_1,
            arm_index=2
        )
        
        return True
    except Exception as e:
        print(e)
        return False

if __name__ == "__main__":
    main()