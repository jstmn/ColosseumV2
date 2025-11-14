#!/usr/bin/env python3
"""Debug script to understand why gripper stays above the plate."""

import gymnasium as gym
import numpy as np
from mani_skill.envs.tasks.tabletop.place_dish_in_rack import PlaceDishInRackEnv

def debug_gripper_position():
    """Debug gripper positioning and plate grasping issues."""

    print("=" * 60)
    print("Debugging Gripper-Plate Position Issue")
    print("=" * 60)

    # Create environment without visualization for debugging
    env = gym.make(
        "PlaceDishInRack-v1",
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode=None,  # No viewer for debug output
    )

    obs, info = env.reset()

    # Get initial positions
    tcp_pos = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
    plate_pos = env.unwrapped.plate.pose.p.cpu().numpy()

    print(f"\nInitial Positions:")
    print(f"TCP (gripper) position: {tcp_pos}")
    print(f"Plate position: {plate_pos}")
    print(f"Initial distance: {np.linalg.norm(tcp_pos - plate_pos):.3f}m")

    # Check plate dimensions
    print(f"\nPlate Configuration:")
    print(f"Plate half-height: {env.unwrapped._plate_total_height / 2:.4f}m")
    print(f"Plate center Z: {plate_pos[0, 2]:.4f}m")
    print(f"Table height: {env.unwrapped.table_scene.table_height:.4f}m")

    # Calculate target grasp position
    target_grasp_height = plate_pos[0, 2] + 0.02  # Slightly above plate center
    print(f"\nTarget grasp height: {target_grasp_height:.4f}m")
    print(f"Current TCP height: {tcp_pos[0, 2]:.4f}m")
    print(f"Height difference: {tcp_pos[0, 2] - target_grasp_height:.4f}m")

    # Test different approach strategies
    print("\n" + "=" * 60)
    print("Testing Approach Strategies")
    print("=" * 60)

    # Strategy 1: Direct downward movement
    print("\n1. Direct Downward Approach:")
    for i in range(50):
        # Move straight down
        action = np.array([0.0, 0.0, -0.01, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        if i % 10 == 0:
            tcp_pos = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
            plate_pos = env.unwrapped.plate.pose.p.cpu().numpy()
            print(f"   Step {i}: TCP Z={tcp_pos[0, 2]:.3f}, Plate Z={plate_pos[0, 2]:.3f}, Distance={np.linalg.norm(tcp_pos - plate_pos):.3f}")

    # Check if we can reach the plate
    tcp_pos = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
    plate_pos = env.unwrapped.plate.pose.p.cpu().numpy()
    final_distance = np.linalg.norm(tcp_pos - plate_pos)

    print(f"\nFinal distance after downward movement: {final_distance:.3f}m")

    # Reset for second strategy
    obs, info = env.reset()

    # Strategy 2: Move XY first, then Z
    print("\n2. XY-then-Z Approach:")

    tcp_pos = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
    plate_pos = env.unwrapped.plate.pose.p.cpu().numpy()

    # Calculate XY offset
    xy_offset = plate_pos[0, :2] - tcp_pos[0, :2]
    print(f"   XY offset to plate: {xy_offset}")

    # Move in XY plane first
    print("   Moving in XY plane...")
    for i in range(30):
        # Proportional movement toward plate XY position
        action = np.array([xy_offset[0]*0.01, xy_offset[1]*0.01, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        if i % 10 == 0:
            tcp_pos = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
            plate_pos = env.unwrapped.plate.pose.p.cpu().numpy()
            xy_dist = np.linalg.norm(tcp_pos[0, :2] - plate_pos[0, :2])
            print(f"   Step {i}: XY distance={xy_dist:.3f}")

    # Then move down
    print("   Moving down to plate...")
    for i in range(50):
        action = np.array([0.0, 0.0, -0.008, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        if i % 10 == 0:
            tcp_pos = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
            plate_pos = env.unwrapped.plate.pose.p.cpu().numpy()
            z_diff = tcp_pos[0, 2] - plate_pos[0, 2]
            print(f"   Step {i}: Z difference={z_diff:.3f}")

    # Try to grasp
    print("\n3. Attempting Grasp:")
    for i in range(30):
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

    is_grasped = env.unwrapped.agent.is_grasping(env.unwrapped.plate)
    print(f"   Grasp result: {'SUCCESS' if is_grasped else 'FAILED'}")

    # Check final positions
    tcp_pos = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
    plate_pos = env.unwrapped.plate.pose.p.cpu().numpy()

    print("\n" + "=" * 60)
    print("Final Analysis:")
    print("=" * 60)
    print(f"Final TCP position: {tcp_pos}")
    print(f"Final Plate position: {plate_pos}")
    print(f"Final distance: {np.linalg.norm(tcp_pos - plate_pos):.3f}m")
    print(f"Z difference: {tcp_pos[0, 2] - plate_pos[0, 2]:.3f}m")

    # Check collision status
    print("\nPotential Issues:")
    if tcp_pos[0, 2] - plate_pos[0, 2] > 0.1:
        print("⚠ Gripper is too high above the plate")
        print("  Possible causes:")
        print("  - Action scaling is too conservative")
        print("  - Collision detection preventing approach")
        print("  - Initial spawn position too far")

    if np.linalg.norm(tcp_pos[0, :2] - plate_pos[0, :2]) > 0.1:
        print("⚠ Gripper XY alignment is off")
        print("  The gripper needs better XY positioning")

    env.close()


if __name__ == "__main__":
    try:
        debug_gripper_position()
    except KeyboardInterrupt:
        print("\nExiting...")