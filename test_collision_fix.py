#!/usr/bin/env python3
"""Test the fixed collision geometry."""

import gymnasium as gym
import numpy as np
from mani_skill.envs.tasks.tabletop.place_dish_in_rack import PlaceDishInRackEnv

def test_collision_fix():
    print("Testing collision geometry fix...")

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

    print(f"Initial TCP height: {tcp_pos[0, 2]:.3f}m")
    print(f"Initial Plate height: {plate_pos[0, 2]:.3f}m")
    print(f"Starting distance: {np.linalg.norm(tcp_pos - plate_pos):.3f}m")

    # Test approach with new collision geometry
    print("\nTesting approach with fixed collision...")

    for i in range(100):
        # Move straight down
        action = np.array([0.0, 0.0, -0.1, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        if i % 20 == 0:
            tcp_pos = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
            plate_pos = env.unwrapped.plate.pose.p.cpu().numpy()
            z_diff = tcp_pos[0, 2] - plate_pos[0, 2]
            dist = np.linalg.norm(tcp_pos - plate_pos)
            print(f"  Step {i:3d}: Height above={z_diff:.3f}m, Distance={dist:.3f}m")

            # Check if we can get very close now
            if z_diff < 0.05:
                print(f"  ✅ SUCCESS! Reached very close to plate at step {i}!")
                print(f"     No more invisible bounding box!")
                break

    # Final check
    tcp_pos = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
    plate_pos = env.unwrapped.plate.pose.p.cpu().numpy()
    final_z_diff = tcp_pos[0, 2] - plate_pos[0, 2]

    if final_z_diff < 0.1:
        print(f"\n✅ Collision fix successful! Can reach within {final_z_diff:.3f}m of plate")
        print("   The invisible bounding box has been eliminated!")

        # Try to grasp
        print("\nAttempting grasp...")
        # Move to plate XY position
        for i in range(20):
            xy_offset = plate_pos[0, :2] - tcp_pos[0, :2]
            action = np.array([
                xy_offset[0] * 0.05,
                xy_offset[1] * 0.05,
                -0.01,
                0.0, 0.0, 0.0, -1.0
            ], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            tcp_pos = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
            plate_pos = env.unwrapped.plate.pose.p.cpu().numpy()

        # Close gripper
        for i in range(25):
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)

        is_grasped = env.unwrapped.agent.is_grasping(env.unwrapped.plate)
        if is_grasped:
            print("  ✅ GRASP SUCCESSFUL! Complete fix verified!")
        else:
            print("  Grasp failed (may need XY alignment)")

    else:
        print(f"\n⚠️ Still {final_z_diff:.3f}m above plate")
        print("   May still have collision issues or workspace limits")

    env.close()


if __name__ == "__main__":
    test_collision_fix()