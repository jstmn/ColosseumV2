#!/usr/bin/env python3
"""Test the final fix for gripper reaching the plate."""

import gymnasium as gym
import numpy as np
from mani_skill.envs.tasks.tabletop.place_dish_in_rack import PlaceDishInRackEnv

def test_final_fix():
    print("=" * 60)
    print("Testing Final Fix for PlaceDishInRack")
    print("=" * 60)
    print("\nChanges made:")
    print("1. ✓ Convex collision for plate (prevents penetration)")
    print("2. ✓ Robot spawned 0.4m lower (closer to table)")
    print("3. ✓ Plate positioned closer to robot workspace")
    print("4. ✓ Rack positioned in better reach zone")
    print("=" * 60)

    # Create environment without visualization for testing
    env = gym.make(
        "PlaceDishInRack-v1",
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode=None,  # No visualization for quick test
    )

    obs, info = env.reset()

    # Get initial positions
    tcp_pos = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
    plate_pos = env.unwrapped.plate.pose.p.cpu().numpy()
    rack_pos = env.unwrapped.dish_rack.pose.p.cpu().numpy()

    print(f"\nInitial Positions:")
    print(f"TCP position: {tcp_pos}")
    print(f"Plate position: {plate_pos}")
    print(f"Rack position: {rack_pos}")
    print(f"Initial distance to plate: {np.linalg.norm(tcp_pos - plate_pos):.3f}m")

    # Approach sequence
    print("\n" + "=" * 60)
    print("Executing Approach and Grasp Sequence")
    print("=" * 60)

    # Step 1: Move above the plate
    print("\n1. Moving above plate...")
    for i in range(15):
        # Move towards plate XY position while descending
        plate_xy_offset = plate_pos[0, :2] - tcp_pos[0, :2]
        action = np.array([
            plate_xy_offset[0] * 0.05,  # Proportional XY movement
            plate_xy_offset[1] * 0.05,
            -0.05,  # Descend
            0.0, 0.0, 0.0,
            -1.0  # Keep gripper open
        ], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        tcp_pos = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
        plate_pos = env.unwrapped.plate.pose.p.cpu().numpy()

    print(f"   Distance to plate: {np.linalg.norm(tcp_pos - plate_pos):.3f}m")
    print(f"   Z difference: {tcp_pos[0, 2] - plate_pos[0, 2]:.3f}m")

    # Step 2: Fine approach
    print("\n2. Fine approach to plate...")
    for i in range(10):
        action = np.array([0.0, 0.0, -0.02, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

    tcp_pos = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
    plate_pos = env.unwrapped.plate.pose.p.cpu().numpy()
    print(f"   Final approach distance: {np.linalg.norm(tcp_pos - plate_pos):.3f}m")

    # Step 3: Grasp
    print("\n3. Grasping plate...")
    for i in range(20):
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

    is_grasped = env.unwrapped.agent.is_grasping(env.unwrapped.plate)
    print(f"   Grasp result: {'✓ SUCCESS' if is_grasped else '✗ FAILED'}")

    if is_grasped:
        # Step 4: Lift
        print("\n4. Lifting plate...")
        for i in range(15):
            action = np.array([0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)

        # Step 5: Rotate to vertical
        print("5. Rotating plate to vertical...")
        for i in range(20):
            action = np.array([0.0, 0.0, 0.0, 0.03, 0.0, 0.0, 1.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)

        # Step 6: Move to rack
        print("6. Moving to rack...")
        tcp_pos = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
        rack_pos = env.unwrapped.dish_rack.pose.p.cpu().numpy()
        rack_offset = rack_pos[0, :2] - tcp_pos[0, :2]

        for i in range(20):
            action = np.array([
                rack_offset[0] * 0.03,
                rack_offset[1] * 0.03,
                0.0, 0.0, 0.0, 0.0, 1.0
            ], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)

        # Step 7: Release
        print("7. Releasing plate...")
        for i in range(15):
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)

        # Check success
        eval_info = env.unwrapped.evaluate()
        print(f"\n   Task success: {eval_info['success'].item()}")

    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    test_final_fix()