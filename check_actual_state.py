#!/usr/bin/env python3
"""Check what's actually happening with robot and plate positions."""

import gymnasium as gym
import numpy as np
from mani_skill.envs.tasks.tabletop.place_dish_in_rack import PlaceDishInRackEnv

def check_actual_state():
    print("Checking actual state of environment...")

    env = gym.make(
        "PlaceDishInRack-v1",
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode="human",  # Enable visualization to see the problem
    )

    # Reset once
    obs, info = env.reset()

    # Get positions
    tcp_pos = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
    plate_pos = env.unwrapped.plate.pose.p.cpu().numpy()
    table_pos = env.unwrapped.table_scene.table.pose.p.cpu().numpy()

    # Get plate orientation
    plate_quat = env.unwrapped.plate.pose.q.cpu().numpy()

    print("\nPOSITIONS:")
    print(f"  Table Z: {table_pos.ravel()[2]:.3f}m")
    print(f"  TCP Z: {tcp_pos.ravel()[2]:.3f}m")
    print(f"  Plate Z: {plate_pos.ravel()[2]:.3f}m")

    print("\nPLATE ORIENTATION (quaternion):")
    print(f"  {plate_quat}")

    print("\nPROBLEMS:")
    if tcp_pos.ravel()[2] < table_pos.ravel()[2] + 0.5:
        print("  ❌ ROBOT IS TOO LOW / UNDER TABLE!")

    # Check if plate is upright (normal should be [0,0,1] for flat)
    # Quaternion [1,0,0,0] means no rotation (flat)
    # If plate is rotated, quaternion will be different
    if abs(plate_quat.ravel()[0] - 1.0) > 0.1:
        print("  ❌ PLATE IS NOT FLAT (rotated/upright)!")

    print("\n[Press Ctrl+C to close visualization]")

    try:
        while True:
            obs, reward, terminated, truncated, info = env.step(np.zeros(7, dtype=np.float32))
    except KeyboardInterrupt:
        pass

    env.close()

if __name__ == "__main__":
    check_actual_state()