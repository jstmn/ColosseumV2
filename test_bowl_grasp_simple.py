#!/usr/bin/env python
"""Simple test of bowl grasp - just run the solve function"""

import sys
sys.path.insert(0, '/home/ashvin/ManiSkill')

import gymnasium as gym
import mani_skill.envs
from mani_skill.examples.motionplanning.panda.solutions.place_dish_in_rack import solve

# Create environment
env = gym.make(
    "PlaceDishInRack-v1",
    num_envs=1,
    obs_mode="state",
    control_mode="pd_joint_pos",
    render_mode="human",
)

# Run solve with debug output
print("=" * 80)
print("TESTING BOWL GRASP - SYMMETRIC CONTACT")
print("Current parameters:")
print("  rim_offset = 100% (exactly at rim edge: 0.085m)")
print("  rim_height_offset = 0.016m (measured rim height)")
print("  tilt = 0° (no tilt for symmetric finger contact)")
print("  closing_time = 25 timesteps")
print("=" * 80)

result = solve(env, seed=0, debug=True, vis=True)

print("\n" + "=" * 80)
print(f"Result: {result}")
print("=" * 80)

env.close()
