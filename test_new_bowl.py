#!/usr/bin/env python3
"""Test the new bowl grasp with top-down approach"""

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

print("=" * 80)
print("TESTING NEW BOWL WITH TOP-DOWN GRASP")
print("=" * 80)

result = solve(env, seed=0, debug=True, vis=True)

print("\n" + "=" * 80)
print(f"Result type: {type(result)}")
if result == -1:
    print("FAILED: Motion planning returned -1")
elif isinstance(result, tuple) and len(result) == 5:
    obs, reward, terminated, truncated, info = result
    print(f"SUCCESS: Returned proper tuple")
    print(f"Reward: {reward}")
    if isinstance(info, dict):
        print(f"Success: {info.get('success', 'N/A')}")
        print(f"Elapsed steps: {info.get('elapsed_steps', 'N/A')}")
else:
    print(f"UNEXPECTED: Got {result}")

env.close()
