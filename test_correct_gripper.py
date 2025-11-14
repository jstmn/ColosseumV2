#!/usr/bin/env python3
"""Test with correct gripper command."""

import gymnasium as gym
import numpy as np
from mani_skill.envs.tasks.tabletop.place_dish_in_rack import PlaceDishInRackEnv

def test_correct_gripper():
    print("Testing movement with correct gripper command (-1 for open)...")

    # Create environment
    env = gym.make(
        "PlaceDishInRack-v1",
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode=None,
    )

    obs, info = env.reset()

    # Get initial positions
    tcp_pos_init = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
    plate_pos_init = env.unwrapped.plate.pose.p.cpu().numpy()

    print(f"Initial TCP Z: {tcp_pos_init[0, 2]:.3f}")
    print(f"Initial Plate Z: {plate_pos_init[0, 2]:.3f}")
    print(f"Initial distance: {np.linalg.norm(tcp_pos_init - plate_pos_init):.3f}m")

    # Move down with gripper OPEN (-1)
    print("\nMoving down with gripper open...")
    for i in range(20):
        # Move down with gripper open
        action = np.array([0.0, 0.0, -0.1, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        if i % 5 == 0:
            tcp_pos = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
            z = tcp_pos[0, 2]
            print(f"Step {i:2d}: TCP Z={z:.3f}")

    # Check position
    tcp_pos = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
    plate_pos = env.unwrapped.plate.pose.p.cpu().numpy()
    dist = np.linalg.norm(tcp_pos - plate_pos)
    z_diff = tcp_pos[0, 2] - plate_pos[0, 2]

    print(f"\nAfter descent:")
    print(f"TCP Z: {tcp_pos[0, 2]:.3f}")
    print(f"Distance to plate: {dist:.3f}m")
    print(f"Z difference: {z_diff:.3f}m")

    if z_diff < 0.1:
        print("✓ Successfully reached near plate height!")
    else:
        print(f"⚠ Still {z_diff:.3f}m above plate")

    # If close enough, try to grasp
    if z_diff < 0.2:
        print("\nClosing gripper...")
        for i in range(20):
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)

        is_grasped = env.unwrapped.agent.is_grasping(env.unwrapped.plate)
        print(f"Grasp result: {'✓ SUCCESS' if is_grasped else '✗ FAILED'}")

    env.close()


if __name__ == "__main__":
    test_correct_gripper()