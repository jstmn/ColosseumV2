#!/usr/bin/env python3
"""Debug table and plate heights."""

import gymnasium as gym
import numpy as np
import torch
from mani_skill.envs.tasks.tabletop.place_dish_in_rack import PlaceDishInRackEnv

def debug_table_height():
    """Debug table and plate positioning."""

    print("=" * 60)
    print("Debugging Table and Plate Heights")
    print("=" * 60)

    # Create environment
    env = gym.make(
        "PlaceDishInRack-v1",
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode=None,
    )

    # Reset environment
    obs, info = env.reset()

    # Check table properties
    print("\nTable Properties:")
    print(f"Table scene exists: {hasattr(env.unwrapped, 'table_scene')}")

    if hasattr(env.unwrapped, 'table_scene'):
        table_scene = env.unwrapped.table_scene

        # Get table position
        if hasattr(table_scene, 'table'):
            table = table_scene.table
            table_pose = table.pose

            # Handle different pose formats
            if hasattr(table_pose, 'p'):
                table_p = table_pose.p
                if isinstance(table_p, torch.Tensor):
                    table_p = table_p.cpu().numpy()
                table_p = np.asarray(table_p).ravel()
                print(f"Table position: {table_p}")
                table_z = float(table_p[-1])
                print(f"Table Z position: {table_z}")
            else:
                print("Table pose.p not found")

        # Get table height attribute
        if hasattr(table_scene, 'table_height'):
            print(f"Table height attribute: {table_scene.table_height}")
        else:
            print("Table height attribute not found")

        # Try to calculate expected table top
        if hasattr(table_scene, 'table') and hasattr(table_scene, 'table_height'):
            table_p_arr = np.asarray(table_scene.table.pose.p).ravel()
            table_z = float(table_p_arr[-1])
            table_top_z = table_z + float(table_scene.table_height)
            print(f"Calculated table top Z: {table_top_z}")

    # Check plate properties
    print("\nPlate Properties:")
    plate = env.unwrapped.plate
    plate_pose = plate.pose

    if hasattr(plate_pose, 'p'):
        plate_p = plate_pose.p
        if isinstance(plate_p, torch.Tensor):
            plate_p = plate_p.cpu().numpy()
        print(f"Plate position: {plate_p}")
        plate_z = float(plate_p.ravel()[-1])
        print(f"Plate Z position: {plate_z}")

    print(f"\nPlate configuration:")
    print(f"Plate total height: {env.unwrapped._plate_total_height}")
    print(f"Plate spawn buffer: {env.unwrapped._plate_spawn_buffer}")

    # Check if plate is below table
    if hasattr(env.unwrapped, 'table_scene') and hasattr(env.unwrapped.table_scene, 'table_height'):
        table_p_arr = np.asarray(env.unwrapped.table_scene.table.pose.p).ravel()
        table_z = float(table_p_arr[-1])
        table_top_z = table_z + float(env.unwrapped.table_scene.table_height)
        plate_z = float(env.unwrapped.plate.pose.p.cpu().numpy().ravel()[-1])

        print(f"\n⚠ PROBLEM DIAGNOSIS:")
        print(f"Table top should be at Z = {table_top_z:.4f}")
        print(f"Plate is actually at Z = {plate_z:.4f}")

        if plate_z < table_z:
            print("❌ PLATE IS BELOW THE TABLE!")
            print("   The plate has fallen through or was spawned incorrectly.")

            # Check the initialization code execution
            print("\nDEBUGGING INITIALIZATION:")

            # Manually recalculate what should happen
            b = 1
            device = env.unwrapped.device
            with torch.device(device):
                xyz = torch.zeros((b, 3), device=device)
                xyz[:, 0] = -0.28
                xyz[:, 1] = -0.26
                plate_half_height = env.unwrapped._plate_total_height / 2.0
                xyz[:, 2] = table_top_z + plate_half_height + env.unwrapped._plate_spawn_buffer

                print(f"Expected plate Z from init code: {xyz[0, 2].item():.4f}")
                print(f"Actual plate Z: {plate_z:.4f}")
                print(f"Difference: {xyz[0, 2].item() - plate_z:.4f}")

        elif np.abs(plate_z - table_top_z) > 0.1:
            print(f"⚠ Plate is {np.abs(plate_z - table_top_z):.3f}m away from table top")

    # Test if physics simulation is causing the issue
    print("\n" + "=" * 60)
    print("Testing Physics Simulation:")
    print("=" * 60)

    # Run some steps to see if plate moves
    for i in range(10):
        obs, reward, terminated, truncated, info = env.step(np.zeros(7, dtype=np.float32))
        if i % 5 == 0:
            plate_z = float(env.unwrapped.plate.pose.p.cpu().numpy().ravel()[-1])
            print(f"Step {i}: Plate Z = {plate_z:.4f}")

    env.close()


if __name__ == "__main__":
    debug_table_height()