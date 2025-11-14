#!/usr/bin/env python3
"""Debug why the plate flips to its side."""

import gymnasium as gym
import numpy as np
from mani_skill.envs.tasks.tabletop.place_dish_in_rack import PlaceDishInRackEnv
import time

def debug_plate_flip():
    print("Debugging plate flipping issue...")

    env = gym.make(
        "PlaceDishInRack-v1",
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode="human",  # Enable visualization
    )

    print("\nResetting environment...")
    obs, info = env.reset()

    # Check initial plate state
    plate_pos = env.unwrapped.plate.pose.p.cpu().numpy().ravel()
    plate_quat = env.unwrapped.plate.pose.q.cpu().numpy().ravel()

    print(f"\nInitial plate position: [{plate_pos[0]:.3f}, {plate_pos[1]:.3f}, {plate_pos[2]:.3f}]")
    print(f"Initial plate quaternion: [{plate_quat[0]:.3f}, {plate_quat[1]:.3f}, {plate_quat[2]:.3f}, {plate_quat[3]:.3f}]")

    if abs(plate_quat[0] - 1.0) < 0.1:
        print("  ✓ Plate starts FLAT")
    else:
        print("  ✗ Plate starts ROTATED")

    print("\nWatching plate for 100 steps (no robot movement)...")
    print("Step | Quaternion W | Status")
    print("-" * 40)

    for i in range(100):
        # No action - just observe
        action = np.zeros(7, dtype=np.float32)
        action[6] = -1.0  # Keep gripper open
        obs, reward, terminated, truncated, info = env.step(action)

        if i % 10 == 0:
            plate_quat = env.unwrapped.plate.pose.q.cpu().numpy().ravel()
            w = plate_quat[0]

            if abs(w - 1.0) < 0.1:
                status = "FLAT"
            elif abs(w) < 0.5:
                status = "ROTATING/FLIPPING!"
            else:
                status = "TILTED"

            print(f"{i:4d} | {w:11.3f} | {status}")

            # Check if plate flipped
            if abs(w) < 0.5:
                print(f"\n❌ PLATE FLIPPED at step {i}!")
                plate_pos = env.unwrapped.plate.pose.p.cpu().numpy().ravel()
                print(f"   Position: [{plate_pos[0]:.3f}, {plate_pos[1]:.3f}, {plate_pos[2]:.3f}]")
                print(f"   Full quaternion: [{plate_quat[0]:.3f}, {plate_quat[1]:.3f}, {plate_quat[2]:.3f}, {plate_quat[3]:.3f}]")
                break

    # Check velocity
    lin_vel = env.unwrapped.plate.get_linear_velocity().cpu().numpy().ravel()
    ang_vel = env.unwrapped.plate.get_angular_velocity().cpu().numpy().ravel()
    print(f"\nPlate velocities:")
    print(f"  Linear: [{lin_vel[0]:.3f}, {lin_vel[1]:.3f}, {lin_vel[2]:.3f}]")
    print(f"  Angular: [{ang_vel[0]:.3f}, {ang_vel[1]:.3f}, {ang_vel[2]:.3f}]")

    print("\n[Visualization showing - Press Ctrl+C to exit]")
    print("Look for:")
    print("  - Green/red collision boxes")
    print("  - Plate orientation")
    print("  - Any instability")

    try:
        while True:
            obs, reward, terminated, truncated, info = env.step(np.zeros(7, dtype=np.float32))
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nExiting...")

    env.close()

if __name__ == "__main__":
    debug_plate_flip()