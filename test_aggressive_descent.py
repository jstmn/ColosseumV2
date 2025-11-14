#!/usr/bin/env python3
"""Test with more aggressive and persistent downward commands."""

import gymnasium as gym
import numpy as np
from mani_skill.envs.tasks.tabletop.place_dish_in_rack import PlaceDishInRackEnv

def test_aggressive_descent():
    print("Testing with aggressive descent strategy...")

    env = gym.make(
        "PlaceDishInRack-v1",
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode=None,
    )

    obs, info = env.reset()

    tcp_pos = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
    plate_pos = env.unwrapped.plate.pose.p.cpu().numpy()

    print(f"Starting height above plate: {tcp_pos[0, 2] - plate_pos[0, 2]:.3f}m")

    # AGGRESSIVE DESCENT: Maximum downward command for many steps
    print("\nPhase 1: Aggressive descent (200 steps)...")
    for i in range(200):
        # Maximum downward movement
        action = np.array([0.0, 0.0, -0.1, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        if i % 20 == 0:
            tcp_pos = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
            plate_pos = env.unwrapped.plate.pose.p.cpu().numpy()
            z_diff = tcp_pos[0, 2] - plate_pos[0, 2]
            print(f"  Step {i:3d}: Height above plate={z_diff:.3f}m")

            if z_diff < 0.1:
                print(f"  ✅ REACHED PLATE HEIGHT at step {i}!")
                break

    # Check final position
    tcp_pos = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
    plate_pos = env.unwrapped.plate.pose.p.cpu().numpy()
    final_z_diff = tcp_pos[0, 2] - plate_pos[0, 2]

    if final_z_diff < 0.15:
        print(f"\n✅ SUCCESS! Gripper reached plate zone (Z diff: {final_z_diff:.3f}m)")

        # Now approach XY position
        print("\nPhase 2: XY alignment...")
        for i in range(30):
            xy_offset = plate_pos[0, :2] - tcp_pos[0, :2]
            action = np.array([
                xy_offset[0] * 0.05,
                xy_offset[1] * 0.05,
                -0.01,  # Small downward to maintain height
                0.0, 0.0, 0.0, -1.0
            ], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)

            tcp_pos = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
            plate_pos = env.unwrapped.plate.pose.p.cpu().numpy()

        xy_dist = np.linalg.norm(tcp_pos[0, :2] - plate_pos[0, :2])
        print(f"  Final XY distance: {xy_dist:.3f}m")

        # Try grasp
        print("\nPhase 3: Grasping...")
        for i in range(30):
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)

        is_grasped = env.unwrapped.agent.is_grasping(env.unwrapped.plate)
        print(f"  Grasp result: {'✅ SUCCESS!' if is_grasped else '❌ Failed'}")

        if is_grasped:
            print("\n🎉 COMPLETE SUCCESS! The fixes work!")
    else:
        print(f"\n❌ Could not reach plate even with 200 aggressive steps")
        print(f"   Final height above: {final_z_diff:.3f}m")

    env.close()


if __name__ == "__main__":
    test_aggressive_descent()