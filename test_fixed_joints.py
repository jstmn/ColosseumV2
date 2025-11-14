#!/usr/bin/env python3
"""Test if fixed joint configuration allows proper movement."""

import gymnasium as gym
import numpy as np
from mani_skill.envs.tasks.tabletop.place_dish_in_rack import PlaceDishInRackEnv

def test_fixed_joints():
    print("Testing with fixed joint configuration...")

    # Create environment
    env = gym.make(
        "PlaceDishInRack-v1",
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode=None,
    )

    obs, info = env.reset()

    # Check initial state
    agent = env.unwrapped.agent
    robot = agent.robot
    qpos = robot.get_qpos()
    qlimits = robot.get_qlimits()

    print("Joint positions after fix:")
    joint_names = agent.arm_joint_names if hasattr(agent, 'arm_joint_names') else []
    for i, name in enumerate(joint_names):
        if i < len(qpos[0]):
            pos = qpos[0, i].item()
            lower = qlimits[0, i, 0].item()
            upper = qlimits[0, i, 1].item()
            margin = min(pos - lower, upper - pos)
            status = "✓" if margin > 0.1 else "⚠️"
            print(f"  {status} {name}: pos={pos:.3f}, margin={margin:.3f}")

    # Get initial positions
    tcp_pos_init = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
    plate_pos_init = env.unwrapped.plate.pose.p.cpu().numpy()

    print(f"\nInitial TCP Z: {tcp_pos_init[0, 2]:.3f}")
    print(f"Initial Plate Z: {plate_pos_init[0, 2]:.3f}")
    print(f"Initial distance: {np.linalg.norm(tcp_pos_init - plate_pos_init):.3f}m")

    # Test downward movement
    print("\nTesting downward movement...")
    for i in range(30):
        # Move down and towards plate
        action = np.array([0.02, 0.02, -0.08, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        if i % 10 == 0:
            tcp_pos = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
            plate_pos = env.unwrapped.plate.pose.p.cpu().numpy()
            dist = np.linalg.norm(tcp_pos - plate_pos)
            z_diff = tcp_pos[0, 2] - plate_pos[0, 2]
            print(f"  Step {i:2d}: Distance={dist:.3f}m, Z diff={z_diff:.3f}m")

    # Check if we reached the plate
    tcp_pos = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
    plate_pos = env.unwrapped.plate.pose.p.cpu().numpy()
    final_dist = np.linalg.norm(tcp_pos - plate_pos)
    z_diff = tcp_pos[0, 2] - plate_pos[0, 2]

    print(f"\nFinal distance: {final_dist:.3f}m")
    print(f"Final Z difference: {z_diff:.3f}m")

    if z_diff < 0.1:
        print("✅ SUCCESS! Gripper can now reach plate height!")

        # Try to grasp
        print("\nAttempting grasp...")
        for i in range(20):
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)

        is_grasped = env.unwrapped.agent.is_grasping(env.unwrapped.plate)
        print(f"Grasp result: {'✅ SUCCESS' if is_grasped else '❌ FAILED (may need better XY alignment)'}")
    else:
        print(f"⚠️ Still {z_diff:.3f}m above plate, but much better than before!")

    env.close()


if __name__ == "__main__":
    test_fixed_joints()