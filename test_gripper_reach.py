#!/usr/bin/env python3
"""Test if gripper can now reach the plate."""

import gymnasium as gym
import numpy as np
from mani_skill.envs.tasks.tabletop.place_dish_in_rack import PlaceDishInRackEnv

def test_gripper_reach():
    print("Testing gripper reach with fixed action scales...")

    # Create environment
    env = gym.make(
        "PlaceDishInRack-v1",
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode=None,
    )

    obs, info = env.reset()

    # Get initial positions
    tcp_pos = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
    plate_pos = env.unwrapped.plate.pose.p.cpu().numpy()

    print(f"Initial TCP Z: {tcp_pos[0, 2]:.3f}")
    print(f"Initial Plate Z: {plate_pos[0, 2]:.3f}")
    print(f"Initial distance: {np.linalg.norm(tcp_pos - plate_pos):.3f}m")

    # Move towards plate
    print("\nMoving towards plate...")
    for i in range(15):
        # Move down and towards plate XY position
        action = np.array([0.02, 0.02, -0.1, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        if i % 5 == 0:
            tcp_pos = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
            plate_pos = env.unwrapped.plate.pose.p.cpu().numpy()
            dist = np.linalg.norm(tcp_pos - plate_pos)
            z_diff = tcp_pos[0, 2] - plate_pos[0, 2]
            print(f"Step {i:2d}: Distance={dist:.3f}m, Z diff={z_diff:.3f}m")

    # Fine approach
    print("\nFine approach...")
    for i in range(10):
        action = np.array([0.01, 0.01, -0.02, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        if i % 5 == 0:
            tcp_pos = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
            plate_pos = env.unwrapped.plate.pose.p.cpu().numpy()
            dist = np.linalg.norm(tcp_pos - plate_pos)
            print(f"Step {i:2d}: Distance={dist:.3f}m")

    # Try to grasp
    print("\nClosing gripper...")
    for i in range(20):
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

    is_grasped = env.unwrapped.agent.is_grasping(env.unwrapped.plate)
    print(f"Grasp result: {'✓ SUCCESS' if is_grasped else '✗ FAILED'}")

    # Final analysis
    tcp_pos = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
    plate_pos = env.unwrapped.plate.pose.p.cpu().numpy()
    final_dist = np.linalg.norm(tcp_pos - plate_pos)
    z_diff = tcp_pos[0, 2] - plate_pos[0, 2]

    print(f"\nFinal distance: {final_dist:.3f}m")
    print(f"Final Z difference: {z_diff:.3f}m")

    if is_grasped:
        print("\n✓ Gripper successfully reached and grasped the plate!")
        print("The action scaling fix allows proper approach.")
    elif z_diff < 0.05:
        print("\n✓ Gripper reached the plate height successfully!")
        print("Grasp failed likely due to XY alignment.")
    else:
        print("\n⚠ Gripper still too high. May need more steps or different approach.")

    env.close()


if __name__ == "__main__":
    test_gripper_reach()