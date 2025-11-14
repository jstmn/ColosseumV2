#!/usr/bin/env python3
"""Test pure downward movement to diagnose control issues."""

import gymnasium as gym
import numpy as np
from mani_skill.envs.tasks.tabletop.place_dish_in_rack import PlaceDishInRackEnv

def test_pure_downward():
    print("Testing pure downward movement...")

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

    print(f"Initial TCP position: {tcp_pos_init}")
    print(f"Initial Plate position: {plate_pos_init}")
    print(f"Initial TCP Z: {tcp_pos_init[0, 2]:.3f}")
    print(f"Initial Plate Z: {plate_pos_init[0, 2]:.3f}")

    # Test 1: Pure negative Z movement
    print("\n--- Test 1: Pure negative Z movement (should go DOWN) ---")
    for i in range(10):
        # Only move in -Z direction (down)
        action = np.array([0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        tcp_pos = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
        print(f"Step {i+1}: TCP Z={tcp_pos[0, 2]:.3f}, XY=({tcp_pos[0, 0]:.3f}, {tcp_pos[0, 1]:.3f})")

    # Reset
    obs, info = env.reset()

    # Test 2: Pure positive Z movement
    print("\n--- Test 2: Pure positive Z movement (should go UP) ---")
    tcp_pos_start = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
    print(f"Starting Z: {tcp_pos_start[0, 2]:.3f}")

    for i in range(5):
        # Only move in +Z direction (up)
        action = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        tcp_pos = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
        print(f"Step {i+1}: TCP Z={tcp_pos[0, 2]:.3f}")

    # Reset
    obs, info = env.reset()

    # Test 3: Move to plate XY first, then down
    print("\n--- Test 3: Move to plate XY, then descend ---")
    tcp_pos = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
    plate_pos = env.unwrapped.plate.pose.p.cpu().numpy()

    xy_diff = plate_pos[0, :2] - tcp_pos[0, :2]
    print(f"XY difference to plate: {xy_diff}")

    # Move in XY only
    print("Moving in XY plane...")
    for i in range(20):
        # Move toward plate XY with small steps
        action = np.array([xy_diff[0]*0.05, xy_diff[1]*0.05, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        if i % 5 == 0:
            tcp_pos = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
            plate_pos = env.unwrapped.plate.pose.p.cpu().numpy()
            xy_dist = np.linalg.norm(tcp_pos[0, :2] - plate_pos[0, :2])
            print(f"  Step {i}: XY distance={xy_dist:.3f}")

    # Now descend
    print("Descending to plate...")
    for i in range(30):
        action = np.array([0.0, 0.0, -0.05, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        if i % 5 == 0:
            tcp_pos = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
            plate_pos = env.unwrapped.plate.pose.p.cpu().numpy()
            z_diff = tcp_pos[0, 2] - plate_pos[0, 2]
            dist = np.linalg.norm(tcp_pos - plate_pos)
            print(f"  Step {i}: Z diff={z_diff:.3f}, Total dist={dist:.3f}")

    # Final status
    tcp_pos = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
    plate_pos = env.unwrapped.plate.pose.p.cpu().numpy()
    final_dist = np.linalg.norm(tcp_pos - plate_pos)
    z_diff = tcp_pos[0, 2] - plate_pos[0, 2]

    print(f"\nFinal Results:")
    print(f"TCP position: {tcp_pos}")
    print(f"Plate position: {plate_pos}")
    print(f"Distance: {final_dist:.3f}m")
    print(f"Z difference: {z_diff:.3f}m")

    if z_diff < 0.05:
        print("✓ Successfully reached plate height!")
    else:
        print(f"⚠ Still {z_diff:.3f}m above plate")

    env.close()


if __name__ == "__main__":
    test_pure_downward()