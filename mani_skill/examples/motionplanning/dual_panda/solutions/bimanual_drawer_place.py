import gymnasium as gym
import numpy as np
import sapien
import time
from mani_skill.examples.motionplanning.dual_panda.motionplanner import DualPandaMotionPlanningSolver
from mani_skill.envs.tasks import DualArmDrawerPlaceEnv
from scipy.spatial.transform import Rotation as R
import sapien.core as sapien
import sapien.physx as physx  # REQUIRED for SAPIEN 3 Physics classes
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb
from mani_skill.utils.geometry.trimesh_utils import get_component_meshes, merge_meshes

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
        
        FINGER_LENGTH = 0.025
        env = env.unwrapped
        
        link = env.open_cabinet.links_map['link_1']
        # Get collision meshes from all objects managed by this link
        meshes = []
        for obj in link._objs:
            meshes.extend(get_component_meshes(obj))
        mesh = merge_meshes(meshes)
        if mesh is None:
            # If no collision mesh, fall back to using root link
            link = env.open_cabinet.root
            meshes = []
            for obj in link._objs:
                meshes.extend(get_component_meshes(obj))
            mesh = merge_meshes(meshes)
        obb = mesh.bounding_box_oriented

        # Get approaching vector along pot's x-axis
        open_cabinet_transform = env.open_cabinet.pose.sp.to_transformation_matrix()
        if hasattr(open_cabinet_transform, 'cpu'):  # if it's a tensor
            approaching = open_cabinet_transform[:3, 0].cpu().numpy()
        else:
            approaching = np.array(open_cabinet_transform[:3, 0])
        # # get transformation matrix of the tcp pose, is default batched and on torch
        target_closing = env.agent.tcp_2.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        # # we can build a simple grasp pose using this information for Panda
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=FINGER_LENGTH,
        )
        closing, center = grasp_info["closing"], grasp_info["center"]
        grasp_1_pose = env.agent.build_grasp_pose(approaching, closing, env.open_cabinet.pose.sp.p)
        grasp_1_pose.set_q(np.array([0.5, 0.5, 0.5, 0.5]))

        obb_A = get_actor_obb(env.obj)
        # Open both grippers
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
        lift_2 = env.agent.build_grasp_pose(approaching, closing, env.obj.pose.sp.p)

        grasp_1_pose = grasp_1_pose * sapien.Pose(p=[0,0,-0.1])
        ready_lift_1 = grasp_1_pose * sapien.Pose(p=[0,0,-0.3])
        ready_lift_2 = lift_2 * sapien.Pose(p=[0,0,-0.1])
        result = planner.move_to_pose_pair_with_screw(
            ready_lift_1,  # Arm 2
            ready_lift_2
        )
        
        if result == -1:
            print("Failed to Ready lift")
            return False
        # planner.render_wait()
        result = planner.move_to_pose_pair_with_screw(
            grasp_1_pose,  # left
            lift_2
        )
        
        if result == -1:
            print("Failed to lift")
            return False
        # planner.render_wait()
        planner.close_gripper(arm_index=1)
        planner.close_gripper(arm_index=2)
        
        pull_1 = grasp_1_pose * sapien.Pose(p=[0, 0, -0.3])
        hold_2 = pull_1 * sapien.Pose(p=[-0.3, 0.1, 0.1])
        hold_2.set_q(lift_2.q)
        
        result = planner.move_to_pose_with_screw(
            hold_2,
            arm_index=1
        )
        # planner.render_wait()
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
        # planner.render_wait()
        release_2 = pull_1 * sapien.Pose(p=[-0.1, 0.15, 0.2])
        release_2.set_q(lift_2.q)
        release_2 = release_2 * sapien.Pose(q=[0.707, 0, 0, 0.707])
        result = planner.move_to_pose_with_screw(
            release_2,
            arm_index=1
        )
        
        if result == -1:
            print("Failed to Release")
            return False
        
        planner.open_gripper(arm_index=1)
        
        result = planner.move_to_pose_with_screw(
            grasp_1_pose,
            arm_index=2
        )
        
        if result == -1:
            print("Failed to Release")
            return False
        
        # planner.render_wait()
        return True
    except Exception as e:
        print(e)
        return False

if __name__ == "__main__":
    main()