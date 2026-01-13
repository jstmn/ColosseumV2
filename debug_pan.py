import numpy as np
import gymnasium as gym
from scipy.spatial.transform import Rotation
import mani_skill.envs

env = gym.make("CookItemInPan-v1", obs_mode="none", control_mode="pd_joint_pos")
env.reset(seed=0)

env_sim = env.unwrapped

pan_pos = env_sim.pan.pose.p[0].cpu().numpy()
pan_q = env_sim.pan.pose.q[0].cpu().numpy()  # wxyz format

print("=== Pan Info ===")
print(f"Pan pose p: {pan_pos}")
print(f"Pan pose q (wxyz): {pan_q}")

# Convert wxyz to xyzw for scipy
pan_q_xyzw = np.array([pan_q[1], pan_q[2], pan_q[3], pan_q[0]])
rot = Rotation.from_quat(pan_q_xyzw)
pan_rot_mat = rot.as_matrix()

print(f"\nRotation matrix:\n{pan_rot_mat}")

print("\n=== Handle Info ===")
handle_local = env_sim.pan_handle_local_point
handle_dir_local = env_sim.pan_handle_local_dir
print(f"pan_handle_local_point: {handle_local}")
print(f"pan_handle_local_dir: {handle_dir_local}")

# Transform to world
handle_world = pan_pos + pan_rot_mat @ handle_local
handle_dir_world = pan_rot_mat @ handle_dir_local

print(f"\nHandle world position: {handle_world}")
print(f"Handle direction world: {handle_dir_world}")

print("\n=== Pan Geometry ===")
print(f"pan_half_size: {env_sim.pan_half_size}")
print(f"pan_bottom_offset: {env_sim.pan_bottom_offset}")
print(f"pan_top_offset: {env_sim.pan_top_offset}")

# Compute bounding box corners in world
bbox_local = np.array([
    [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]
]) * env_sim.pan_half_size

bbox_world = np.array([pan_pos + pan_rot_mat @ p for p in bbox_local])
print(f"\nWorld bbox min: {bbox_world.min(axis=0)}")
print(f"World bbox max: {bbox_world.max(axis=0)}")

env.close()
