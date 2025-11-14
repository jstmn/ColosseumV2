#!/usr/bin/env python3
"""Debug script to check actual positions."""

import gymnasium as gym
import numpy as np
from mani_skill.envs.tasks.tabletop.place_dish_in_rack import PlaceDishInRackEnv

def debug_positions():
    print("Debugging PlaceDishInRack positions...")

    # Create environment
    env = gym.make(
        "PlaceDishInRack-v1",
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode=None,
    )

    obs, info = env.reset()

    # Get all relevant positions
    tcp_pos = env.unwrapped.agent.tcp_pose.p
    plate_pos = env.unwrapped.plate.pose.p
    rack_pos = env.unwrapped.dish_rack.pose.p
    table_pos = env.unwrapped.table_scene.table.pose.p

    # Convert to numpy for easier printing
    tcp_np = tcp_pos.cpu().numpy() if hasattr(tcp_pos, 'cpu') else np.array(tcp_pos)
    plate_np = plate_pos.cpu().numpy() if hasattr(plate_pos, 'cpu') else np.array(plate_pos)
    rack_np = rack_pos.cpu().numpy() if hasattr(rack_pos, 'cpu') else np.array(rack_pos)
    table_np = table_pos.cpu().numpy() if hasattr(table_pos, 'cpu') else np.array(table_pos)

    # Handle potential batched tensors (1x3)
    tcp_np = tcp_np.ravel()[:3] if tcp_np.size > 3 else tcp_np.ravel()
    plate_np = plate_np.ravel()[:3] if plate_np.size > 3 else plate_np.ravel()
    rack_np = rack_np.ravel()[:3] if rack_np.size > 3 else rack_np.ravel()
    table_np = table_np.ravel()[:3] if table_np.size > 3 else table_np.ravel()

    print(f"\nDEBUG POSITIONS:")
    print(f"  Table:  X={table_np[0]:.3f}, Y={table_np[1]:.3f}, Z={table_np[2]:.3f}")
    print(f"  TCP:    X={tcp_np[0]:.3f}, Y={tcp_np[1]:.3f}, Z={tcp_np[2]:.3f}")
    print(f"  Plate:  X={plate_np[0]:.3f}, Y={plate_np[1]:.3f}, Z={plate_np[2]:.3f}")
    print(f"  Rack:   X={rack_np[0]:.3f}, Y={rack_np[1]:.3f}, Z={rack_np[2]:.3f}")

    # Check table height and offset
    table_height = env.unwrapped.table_scene.table_height
    table_height_offset = env.unwrapped.table_height_offset

    print(f"\nTABLE PARAMETERS:")
    print(f"  table_height: {table_height}")
    print(f"  table_height_offset: {table_height_offset}")
    print(f"  Expected table top: {table_np[2] + table_height:.3f}")

    # Check plate parameters
    plate_total_height = env.unwrapped._plate_total_height
    plate_spawn_buffer = env.unwrapped._plate_spawn_buffer

    print(f"\nPLATE PARAMETERS:")
    print(f"  _plate_total_height: {plate_total_height}")
    print(f"  _plate_spawn_buffer: {plate_spawn_buffer}")
    print(f"  Expected plate Z: {table_np[2] + table_height + plate_total_height/2 + plate_spawn_buffer:.3f}")

    # Distance calculations
    tcp_to_plate = np.linalg.norm(tcp_np - plate_np)
    tcp_to_plate_z = tcp_np[2] - plate_np[2]
    tcp_to_plate_xy = np.linalg.norm(tcp_np[:2] - plate_np[:2])

    print(f"\nDISTANCES:")
    print(f"  TCP to Plate (3D): {tcp_to_plate:.3f}m")
    print(f"  TCP to Plate (Z only): {tcp_to_plate_z:.3f}m")
    print(f"  TCP to Plate (XY only): {tcp_to_plate_xy:.3f}m")

    env.close()

    # Check for anomalies
    print("\nANOMALY CHECK:")
    if plate_np[2] > 2.0:
        print("  ⚠️ PLATE Z POSITION IS ABNORMALLY HIGH!")
    if tcp_to_plate_xy > 1.0:
        print("  ⚠️ PLATE XY DISTANCE IS ABNORMALLY LARGE!")
    if abs(table_np[2] - table_height_offset) > 0.01:
        print("  ⚠️ TABLE POSITION DOESN'T MATCH EXPECTED OFFSET!")

if __name__ == "__main__":
    debug_positions()