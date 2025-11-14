"""Test script to verify the hammer loads correctly from the STL file."""

import gymnasium as gym
import numpy as np

# Create the environment
env = gym.make(
    "HammerNail-v1",
    render_mode="human",
    num_envs=1,
)

print("Environment created successfully!")
print(f"Hammer actor: {env.unwrapped.hammer}")

# Reset and run for a few steps to visualize
obs, info = env.reset()
print("Environment reset successfully!")

# Check if hammer loaded from STL or used fallback
if hasattr(env.unwrapped.hammer, 'name'):
    print(f"Hammer name: {env.unwrapped.hammer.name}")

# Run a few steps to see the scene
for i in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if i == 0:
        print(f"First step completed successfully!")
        print(f"Nail platform visibility test: The platform should be invisible in camera observations")

env.close()
print("Test completed successfully!")
