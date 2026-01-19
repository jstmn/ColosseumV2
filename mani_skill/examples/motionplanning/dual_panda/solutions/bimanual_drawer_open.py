import gymnasium as gym
import numpy as np
import sapien
import time
from mani_skill.examples.motionplanning.dual_panda.motionplanner import DualPandaMotionPlanningSolver
from mani_skill.envs.tasks import DualArmDrawerOpenEnv
from scipy.spatial.transform import Rotation as R
import sapien.core as sapien
import sapien.physx as physx  # REQUIRED for SAPIEN 3 Physics classes
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb
from mani_skill.utils.geometry.trimesh_utils import get_component_meshes, merge_meshes

def main():
    """
    Test the dual panda motion planner with various scenarios.
    """
    env:DualArmDrawerOpenEnv = gym.make(
        'DualArmDrawerOpen-v0',
        obs_mode='none',
        control_mode="pd_joint_pos",  # Use pd_joint_pos for motion planning
        render_mode='human',  # Use 'human' for visualization
    )
    # increase_handle_friction(env)
    # debug_collision_properties(env)
    for seed in range(10):  # Test with 3 different seeds
        print(f"\n--- Seed {seed} ---")
        success = solve(env, seed=seed, debug=True, vis=True)            
        # print(f"res: {'Success' if success else 'Failure'}")
        # env.reset()
    env.close()

def solve(env:DualArmDrawerOpenEnv, seed, debug=False, vis=False):
    env.reset(seed=seed)
    if vis: 
        env.unwrapped.render_human()
    planner = DualPandaMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        print_env_info=True
    )
    
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

    # retrieves the object oriented bounding box (trimesh box object)
    # For articulation links, we need to get the collision meshes directly
    # print("LINKS:",env.open_cabinet.links_map)
    link = env.open_cabinet.links[1] if len(env.open_cabinet.links) > 1 else env.open_cabinet.root
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
    target_closing = env.agent.tcp_1.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # # we can build a simple grasp pose using this information for Panda
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_1_pose = env.agent.build_grasp_pose(approaching, closing, env.open_cabinet.pose.sp.p)
    grasp_1_pose = grasp_1_pose*sapien.Pose(p=[0,-0.15,-0.1])
    
    target_closing = env.agent.tcp_2.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    target_closing = env.agent.tcp_2.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_2_pose = env.agent.build_grasp_pose(approaching, closing, env.open_cabinet.pose.sp.p)     
    grasp_2_pose = grasp_2_pose*sapien.Pose(p=[0, 0.15, -0.1])
    
    grasp_1_approach_pose = grasp_1_pose*sapien.Pose(p=[0,0,-0.3])
    grasp_2_approach_pose = grasp_2_pose*sapien.Pose(p=[0, 0, -0.3])
    
    res = planner.move_to_pose_pair_with_RRTConnect(
        grasp_2_approach_pose,
        grasp_1_approach_pose  # left
    )
    # viewer = planner.base_env.render_human()
    # while True:
    #     if viewer.window.key_down("c"):
    #         break
    #     planner.base_env.render_human()

    if res==-1:
        print("Failed grasp_approach")
        return res
    
    res = planner.move_to_pose_pair_with_screw(
        grasp_2_pose,
        grasp_1_pose
    )
    
    if res == -1:
        print("Failed grasp")
        return res
    
    planner.close_gripper(arm_index=1)
    planner.close_gripper(arm_index=2)
    
    pull_1 = grasp_1_pose * sapien.Pose(p=[0, 0, -0.3])
    pull_2 = grasp_2_pose * sapien.Pose(p=[0, 0, -0.3])
    
    res = planner.move_to_pose_pair_with_screw(
        pull_1,
        pull_2
    )
    
    if res == -1:
        print("Failed to Pull")
        return res
    
    
    return res

if __name__ == "__main__":
    main()