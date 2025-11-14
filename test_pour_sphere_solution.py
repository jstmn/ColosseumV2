#!/usr/bin/env python3
"""Test the PourSphere motion planning solution"""

import gymnasium as gym
import mani_skill.envs
from mani_skill.examples.motionplanning.panda.solutions.pour_sphere import solve

# Create the environment
env = gym.make(
    "PourSphere-v1",
    obs_mode="state",
    control_mode="pd_joint_pos",
    render_mode="human",
    num_envs=1,
)

print("Testing PourSphere motion planning solution...")
print("=" * 60)

# Run the solution
result = solve(env, seed=0, debug=True, vis=True)

print("=" * 60)
print(f"Solution result: {result}")
print("Test completed!")

env.close()
