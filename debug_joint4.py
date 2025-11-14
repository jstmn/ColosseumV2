#!/usr/bin/env python3
"""Debug why joint4 keeps going to -0.070."""

import gymnasium as gym
import numpy as np
from mani_skill.envs.tasks.tabletop.place_dish_in_rack import PlaceDishInRackEnv

def debug_joint4():
    print("Debugging joint4 issue...")

    # Create environment
    env = gym.make(
        "PlaceDishInRack-v1",
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode=None,
    )

    # Before reset
    print("Before reset...")

    # Reset
    obs, info = env.reset()

    # Check qpos immediately
    agent = env.unwrapped.agent
    robot = agent.robot
    qpos = robot.get_qpos()

    print(f"\nActual qpos after reset:")
    for i in range(9):
        print(f"  Joint {i}: {qpos[0, i].item():.3f}")

    print(f"\nExpected joint4: -1.8")
    print(f"Actual joint4: {qpos[0, 3].item():.3f}")

    if abs(qpos[0, 3].item() - (-0.070)) < 0.001:
        print("❌ Joint4 is at limit despite our initialization!")
        print("\nPossible causes:")
        print("1. Keyframe override")
        print("2. Controller reset")
        print("3. Robot model constraint")

        # Check if there's a keyframe
        if hasattr(agent, 'keyframe'):
            print(f"\nAgent keyframe: {agent.keyframe}")

        # Check robot's rest qpos
        if hasattr(robot, 'rest_qpos'):
            print(f"\nRobot rest_qpos: {robot.rest_qpos}")

    env.close()


if __name__ == "__main__":
    debug_joint4()