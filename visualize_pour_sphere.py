#!/usr/bin/env python3
"""Visualize the PourSphere environment without motion planning"""

import gymnasium as gym
import mani_skill.envs
import numpy as np

# Create the environment with visualization
env = gym.make(
    "PourSphere-v1",
    obs_mode="state",
    control_mode="pd_joint_pos",
    render_mode="human",
    num_envs=1,
)

print("PourSphere Environment Visualization")
print("=" * 60)
print("The scene should show:")
print("  - Two cylindrical cups standing UPRIGHT on the table")
print("  - A red sphere inside the first cup")
print("  - The Panda robot")
print("=" * 60)
print("\nPress Ctrl+C to exit")

# Reset and hold the visualization
obs, info = env.reset(seed=0)

# Keep the window open for viewing
try:
    for i in range(10000):
        # Do nothing action (hold position)
        action = np.zeros(env.action_space.shape)
        obs, reward, terminated, truncated, info = env.step(action)

        if i == 0:
            print("\nEnvironment is now visible!")
            print("You should see two upright cups with a sphere in one of them.")

        if terminated.any() or truncated.any():
            obs, info = env.reset()

except KeyboardInterrupt:
    print("\nClosing...")

env.close()
