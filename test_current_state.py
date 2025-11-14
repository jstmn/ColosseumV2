#!/usr/bin/env python3
"""Test the current state to see what's wrong."""

import gymnasium as gym
import numpy as np
from mani_skill.envs.tasks.tabletop.place_dish_in_rack import PlaceDishInRackEnv

def test_current_state():
    print("Testing current environment state...")

    env = gym.make(
        "PlaceDishInRack-v1",
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode=None,
    )

    # Single reset
    obs, info = env.reset()

    # Get all positions
    tcp_pos = env.unwrapped.agent.tcp_pose.p.cpu().numpy().ravel()
    plate_pos = env.unwrapped.plate.pose.p.cpu().numpy().ravel()
    table_pos = env.unwrapped.table_scene.table.pose.p.cpu().numpy().ravel()
    rack_pos = env.unwrapped.dish_rack.pose.p.cpu().numpy().ravel()

    # Get orientations
    plate_quat = env.unwrapped.plate.pose.q.cpu().numpy().ravel()

    # Get table info
    table_height = env.unwrapped.table_scene.table_height
    table_top_z = table_pos[2] + table_height

    print("\n=== POSITIONS ===")
    print(f"Table: X={table_pos[0]:.2f}, Y={table_pos[1]:.2f}, Z={table_pos[2]:.2f}")
    print(f"Table top: Z={table_top_z:.2f}")
    print(f"TCP:   X={tcp_pos[0]:.2f}, Y={tcp_pos[1]:.2f}, Z={tcp_pos[2]:.2f}")
    print(f"Plate: X={plate_pos[0]:.2f}, Y={plate_pos[1]:.2f}, Z={plate_pos[2]:.2f}")
    print(f"Rack:  X={rack_pos[0]:.2f}, Y={rack_pos[1]:.2f}, Z={rack_pos[2]:.2f}")

    print("\n=== PLATE ORIENTATION ===")
    print(f"Quaternion (w,x,y,z): [{plate_quat[0]:.3f}, {plate_quat[1]:.3f}, {plate_quat[2]:.3f}, {plate_quat[3]:.3f}]")
    if abs(plate_quat[0] - 1.0) < 0.1:
        print("  ✓ Plate is FLAT (identity quaternion)")
    else:
        print("  ✗ Plate is ROTATED/UPRIGHT")

    print("\n=== ANALYSIS ===")

    # Check if robot is under table
    if tcp_pos[2] < table_top_z:
        print(f"  ❌ ROBOT IS UNDER TABLE! (TCP Z={tcp_pos[2]:.2f} < Table top Z={table_top_z:.2f})")
    else:
        print(f"  ✓ Robot is above table (TCP Z={tcp_pos[2]:.2f} > Table top Z={table_top_z:.2f})")

    # Check if plate is on table
    expected_plate_z = table_top_z + env.unwrapped._plate_total_height/2
    if abs(plate_pos[2] - expected_plate_z) < 0.1:
        print(f"  ✓ Plate is on table (Z={plate_pos[2]:.2f}, expected={expected_plate_z:.2f})")
    else:
        print(f"  ✗ Plate position wrong (Z={plate_pos[2]:.2f}, expected={expected_plate_z:.2f})")

    # Check distances
    tcp_to_plate_xy = np.linalg.norm(tcp_pos[:2] - plate_pos[:2])
    tcp_to_plate_z = tcp_pos[2] - plate_pos[2]

    print(f"\n=== DISTANCES ===")
    print(f"TCP to Plate XY: {tcp_to_plate_xy:.2f}m")
    print(f"TCP to Plate Z: {tcp_to_plate_z:.2f}m")

    env.close()

if __name__ == "__main__":
    test_current_state()