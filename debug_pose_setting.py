#!/usr/bin/env python3
"""Debug pose setting to find the issue."""

import gymnasium as gym
import numpy as np
import torch
from mani_skill.envs.tasks.tabletop.place_dish_in_rack import PlaceDishInRackEnv
from mani_skill.utils.structs import Pose
import sapien

def debug_pose_setting():
    print("Debugging pose setting...")

    env = gym.make(
        "PlaceDishInRack-v1",
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode=None,
    )

    # Manually test pose setting
    print("\nBefore reset:")
    obs, info = env.reset()

    # Get the plate
    plate = env.unwrapped.plate
    print(f"Plate after reset: {plate.pose.p}")

    # Try setting pose manually
    print("\nTrying manual pose setting...")

    # Method 1: Using sapien.Pose directly
    test_pose1 = sapien.Pose(p=[0.0, 0.0, 0.5])
    plate.set_pose(test_pose1)
    print(f"After sapien.Pose set: {plate.pose.p}")

    # Method 2: Using Pose.create_from_pq
    test_p = torch.tensor([[0.1, 0.1, 0.6]], device=env.device)
    test_q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)
    test_pose2 = Pose.create_from_pq(p=test_p, q=test_q)
    plate.set_pose(test_pose2)
    print(f"After Pose.create_from_pq set: {plate.pose.p}")

    # Check what's happening in the initialization
    print("\nChecking initialization values:")

    # Get table info
    table_pose = env.unwrapped.table_scene.table.pose
    table_p = np.asarray(table_pose.p).ravel()
    print(f"Table position: {table_p}")
    print(f"Table height: {env.unwrapped.table_scene.table_height}")
    print(f"Table height offset: {env.unwrapped.table_height_offset}")

    # Compute expected plate position
    table_top_z = table_p[-1] + float(env.unwrapped.table_scene.table_height)
    print(f"Computed table top Z: {table_top_z:.3f}")

    plate_half_height = env.unwrapped._plate_total_height / 2.0
    expected_plate_z = table_top_z + plate_half_height + env.unwrapped._plate_spawn_buffer
    print(f"Expected plate Z: {expected_plate_z:.3f}")

    # Try resetting again and watch what happens
    print("\nResetting again to trace the issue...")
    obs, info = env.reset()

    # Get positions immediately after reset
    plate_pos_after = env.unwrapped.plate.pose.p
    print(f"Plate position after reset: {plate_pos_after}")

    # Check the type of plate_pos_after
    print(f"Type of plate position: {type(plate_pos_after)}")
    if hasattr(plate_pos_after, 'shape'):
        print(f"Shape: {plate_pos_after.shape}")
    if hasattr(plate_pos_after, 'device'):
        print(f"Device: {plate_pos_after.device}")

    env.close()

if __name__ == "__main__":
    debug_pose_setting()