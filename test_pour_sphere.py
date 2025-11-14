#!/usr/bin/env python3
"""Test the PourSphere environment"""

import gymnasium as gym
import mani_skill.envs

# Create the environment
env = gym.make(
    "PourSphere-v1",
    obs_mode="state",
    control_mode="pd_joint_pos",
    render_mode="human",
    num_envs=1,
)

print("Environment created successfully!")
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")

# Reset the environment
obs, info = env.reset(seed=0)

print(f"\nInitial observation shape: {obs.shape}")

# Run a few steps with random actions
print("\nRunning environment for 10 steps...")
for i in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i+1}: reward={reward.item():.4f}, success={info['success'].item()}")

    if terminated.any() or truncated.any():
        print("Episode finished!")
        break

# Check object positions from info
print("\nEvaluation info:")
print(f"  Success: {info['success'].item()}")
print(f"  Sphere in cup2 radius: {info.get('sphere_in_cup2_radius', 'N/A')}")
print(f"  Sphere in cup2 height: {info.get('sphere_in_cup2_height', 'N/A')}")

env.close()
print("\nTest completed successfully!")
