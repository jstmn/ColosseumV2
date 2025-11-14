#!/usr/bin/env python3
"""Test the PourSphere motion planning solution without rendering"""

import gymnasium as gym
import mani_skill.envs
from mani_skill.examples.motionplanning.panda.solutions.pour_sphere import solve

# Create the environment without rendering
env = gym.make(
    "PourSphere-v1",
    obs_mode="state",
    control_mode="pd_joint_pos",
    render_mode=None,  # No rendering
    num_envs=1,
)

print("Testing PourSphere motion planning solution (headless)...")
print("=" * 60)

try:
    # Run the solution
    result = solve(env, seed=0, debug=True, vis=False)

    print("=" * 60)
    print(f"Solution result: {result}")

    # Check if task was successful
    info = env.unwrapped.get_info()
    success = info.get('success', [False])[0]
    print(f"Task success: {success}")

    print("\nTest completed successfully!")

except Exception as e:
    print(f"Error during solution execution: {e}")
    import traceback
    traceback.print_exc()
finally:
    env.close()
