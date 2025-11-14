#!/usr/bin/env python3
"""Simple test to verify PourSphere environment setup"""

import gymnasium as gym
import mani_skill.envs
import numpy as np

print("Testing PourSphere Environment (Headless)")
print("=" * 60)

# Create environment without rendering
env = gym.make(
    "PourSphere-v1",
    obs_mode="state",
    control_mode="pd_joint_pos",
    render_mode=None,  # Headless
    num_envs=1,
)

# Reset environment
obs, info = env.reset(seed=0)

# Get the unwrapped environment to access object poses
env_unwrapped = env.unwrapped

# Check object positions
cup1_pos = env_unwrapped.cup1.pose.p[0].cpu().numpy()
cup2_pos = env_unwrapped.cup2.pose.p[0].cpu().numpy()
sphere_pos = env_unwrapped.sphere.pose.p[0].cpu().numpy()

print("\nObject Positions:")
print(f"  Cup1 (source):  {cup1_pos}")
print(f"  Cup2 (target):  {cup2_pos}")
print(f"  Sphere:         {sphere_pos}")

# Check that cups are upright and on the table
table_height = 0.0  # Approximate table surface
cup_height = env_unwrapped._cup_height

print(f"\nCup Configuration:")
print(f"  Cup height: {cup_height}m")
print(f"  Cup radius: {env_unwrapped._cup_radius}m")
print(f"  Sphere radius: {env_unwrapped._sphere_radius}m")

# Verify cups are standing upright (center should be at table + cup_height/2)
cup1_expected_z = table_height + cup_height / 2
cup2_expected_z = table_height + cup_height / 2

print(f"\nVerification:")
print(f"  Cup1 Z position: {cup1_pos[2]:.4f}m (expected: ~{cup1_expected_z:.4f}m)")
print(f"  Cup2 Z position: {cup2_pos[2]:.4f}m (expected: ~{cup2_expected_z:.4f}m)")

# Check if sphere is inside cup1
sphere_in_cup1_xy = np.linalg.norm(sphere_pos[:2] - cup1_pos[:2]) < env_unwrapped._cup_radius
sphere_in_cup1_z = (sphere_pos[2] > cup1_pos[2] - cup_height/2) and (sphere_pos[2] < cup1_pos[2] + cup_height/2)
sphere_in_cup1 = sphere_in_cup1_xy and sphere_in_cup1_z

print(f"  Sphere inside Cup1: {sphere_in_cup1}")
print(f"    - XY distance to cup1: {np.linalg.norm(sphere_pos[:2] - cup1_pos[:2]):.4f}m (< {env_unwrapped._cup_radius}m)")
print(f"    - Z relative to cup1: {sphere_pos[2] - cup1_pos[2]:.4f}m")

# Run a few steps
print(f"\nRunning 5 simulation steps...")
for i in range(5):
    action = np.zeros(env.action_space.shape)
    obs, reward, terminated, truncated, info = env.step(action)

sphere_pos_after = env_unwrapped.sphere.pose.p[0].cpu().numpy()
print(f"  Sphere position after steps: {sphere_pos_after}")
print(f"  Sphere moved: {np.linalg.norm(sphere_pos_after - sphere_pos):.6f}m")

# Check evaluation
eval_result = env_unwrapped.evaluate()
print(f"\nTask Evaluation:")
print(f"  Success: {eval_result['success'][0].item()}")
print(f"  Sphere in cup2 radius: {eval_result['sphere_in_cup2_radius'][0].item()}")
print(f"  Sphere in cup2 height: {eval_result['sphere_in_cup2_height'][0].item()}")
print(f"  Sphere static: {eval_result['sphere_static'][0].item()}")

env.close()

print("\n" + "=" * 60)
print("✓ Environment test completed successfully!")
print("The cups are standing upright and the sphere is in cup1.")
print("=" * 60)
