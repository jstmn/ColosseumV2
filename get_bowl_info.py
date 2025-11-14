#!/usr/bin/env python3
"""Get ceramic bowl dimensions and rim location for grasping"""

import numpy as np
import gymnasium as gym

# Import mani_skill to register environments
import mani_skill.envs

# Create environment
env = gym.make(
    "PlaceDishInRack-v1",
    num_envs=1,
    obs_mode="state",
    render_mode="human",
)

# Reset and get bowl info
obs, info = env.reset()
env_unwrapped = env.unwrapped

# Get bowl pose
bowl_pose = env_unwrapped.plate.pose
bowl_p = bowl_pose.p[0].cpu().numpy()
bowl_q = bowl_pose.q[0].cpu().numpy()

print("=" * 60)
print("BOWL INFORMATION")
print("=" * 60)
print(f"Bowl position: {bowl_p}")
print(f"Bowl quaternion [w,x,y,z]: {bowl_q}")
print(f"Bowl scale: {env_unwrapped._plate_scale}")

# Get bounding box from collision mesh
try:
    collision_mesh = env_unwrapped.plate.get_first_collision_mesh()
    bbox = collision_mesh.bounding_box
    print(f"\nBounding box min: {bbox.bounds[0]}")
    print(f"Bounding box max: {bbox.bounds[1]}")

    bbox_center = (bbox.bounds[0] + bbox.bounds[1]) / 2
    bbox_size = bbox.bounds[1] - bbox.bounds[0]

    print(f"\nBounding box center: {bbox_center}")
    print(f"Bounding box size (width, depth, height): {bbox_size}")
    print(f"  Width (X): {bbox_size[0]:.4f}m")
    print(f"  Depth (Y): {bbox_size[1]:.4f}m")
    print(f"  Height (Z): {bbox_size[2]:.4f}m")

    # Calculate rim position (assuming rim is at top edge)
    rim_height_world = bowl_p[2] + (bbox.bounds[1][2] - bbox_center[2])
    rim_radius = max(bbox_size[0], bbox_size[1]) / 2

    print(f"\nRIM INFORMATION:")
    print(f"  Rim height (world Z): {rim_height_world:.4f}m")
    print(f"  Rim radius: {rim_radius:.4f}m")
    print(f"  Bowl center to rim edge: {rim_radius:.4f}m")

    # Suggest grasp point at rim
    grasp_point = bowl_p.copy()
    grasp_point[1] -= rim_radius * 0.8  # 80% of radius from center
    grasp_point[2] = rim_height_world

    print(f"\nSUGGESTED GRASP POINT (at front rim):")
    print(f"  Position: {grasp_point}")
    print(f"  Offset from bowl center: [0, -{rim_radius * 0.8:.4f}, {rim_height_world - bowl_p[2]:.4f}]")

except Exception as e:
    print(f"\nError getting collision mesh: {e}")

print("=" * 60)

# Keep window open
input("\nPress Enter to close...")
env.close()
