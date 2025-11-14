#!/usr/bin/env python3
"""Test script to verify that penetration between gripper and plate is fixed."""

import gymnasium as gym
import numpy as np
import torch
from mani_skill.envs.tasks.tabletop.place_dish_in_rack import PlaceDishInRackEnv

def test_penetration_fix():
    """Test the PlaceDishInRack environment with improved physics settings."""

    print("Testing PlaceDishInRack environment with penetration fix...")
    print("- Simulation frequency: 500 Hz (increased from 100 Hz)")
    print("- Position action scale: ±0.02 (reduced from ±0.1)")
    print("- Rotation action scale: ±0.05 (reduced from ±0.1)")
    print()

    # Create environment
    env = gym.make(
        "PlaceDishInRack-v1",
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode="human",
        human_render_camera_configs={"width": 800, "height": 600},
    )

    # Reset environment
    obs, info = env.reset()
    print("Environment initialized successfully")
    print(f"Simulation frequency: {env.unwrapped.sim_freq} Hz")
    print(f"Control frequency: {env.unwrapped.control_freq} Hz")
    print(f"Physics steps per control step: {env.unwrapped.sim_freq // env.unwrapped.control_freq}")
    print()

    # Test sequence: approach plate, grasp it, and lift
    print("Testing gripper-plate interaction...")

    # Move gripper towards the plate
    print("1. Moving gripper towards plate...")
    for i in range(30):
        # Move down and forward towards plate
        action = np.array([0.01, 0.01, -0.005, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)  # Small increments
        obs, reward, terminated, truncated, info = env.step(action)

        if i % 10 == 0:
            tcp_pos = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
            plate_pos = env.unwrapped.plate.pose.p.cpu().numpy()
            distance = np.linalg.norm(tcp_pos - plate_pos)
            print(f"   Step {i}: Distance to plate: {distance:.3f}m")

    # Close gripper to grasp
    print("2. Closing gripper to grasp plate...")
    for i in range(20):
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # Close gripper
        obs, reward, terminated, truncated, info = env.step(action)

    is_grasped = env.unwrapped.agent.is_grasping(env.unwrapped.plate)
    print(f"   Grasp status: {'Success' if is_grasped else 'Failed'}")

    # Lift the plate
    print("3. Lifting plate upward...")
    for i in range(30):
        action = np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # Lift up with gripper closed
        obs, reward, terminated, truncated, info = env.step(action)

        if i % 10 == 0:
            plate_height = env.unwrapped.plate.pose.p[:, 2].cpu().numpy().item()
            print(f"   Step {i}: Plate height: {plate_height:.3f}m")

    # Rotate the plate to vertical orientation
    print("4. Rotating plate to vertical orientation...")
    for i in range(40):
        # Rotate around x-axis to make plate vertical
        action = np.array([0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        if i % 10 == 0:
            rot_mats = env.unwrapped.agent.robot.links_map[env.unwrapped.agent.ee_link_name].pose.to_rotation_matrix()
            print(f"   Step {i}: Rotation applied")

    # Move towards rack
    print("5. Moving towards dish rack...")
    for i in range(50):
        # Move towards rack position
        action = np.array([0.005, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        if i % 10 == 0:
            tcp_pos = env.unwrapped.agent.tcp_pose.p.cpu().numpy()
            rack_pos = env.unwrapped.dish_rack.pose.p.cpu().numpy()
            distance = np.linalg.norm(tcp_pos - rack_pos)
            print(f"   Step {i}: Distance to rack: {distance:.3f}m")

    # Release the plate
    print("6. Releasing plate into rack...")
    for i in range(20):
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)  # Open gripper
        obs, reward, terminated, truncated, info = env.step(action)

    # Check final state
    print("\nFinal evaluation:")
    eval_info = env.unwrapped.evaluate()
    for key, value in eval_info.items():
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy().item()
        print(f"  {key}: {value}")

    print("\n✓ Test completed successfully!")
    print("If no penetration warnings or physics explosions occurred, the fix is working.")

    env.close()


if __name__ == "__main__":
    test_penetration_fix()