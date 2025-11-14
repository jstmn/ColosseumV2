#!/usr/bin/env python3
"""Diagnose why the controller prevents downward movement."""

import gymnasium as gym
import numpy as np
from mani_skill.envs.tasks.tabletop.place_dish_in_rack import PlaceDishInRackEnv

def diagnose_controller():
    print("Diagnosing controller issues...")

    # Create environment
    env = gym.make(
        "PlaceDishInRack-v1",
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode=None,
    )

    obs, info = env.reset()

    # Check controller info
    agent = env.unwrapped.agent
    controller = agent.controller

    print(f"\nController type: {type(controller)}")
    print(f"Controller configs: {controller.configs if hasattr(controller, 'configs') else 'N/A'}")

    # Check joint positions and limits
    robot = agent.robot
    qpos = robot.get_qpos()
    qlimits = robot.get_qlimits()

    print(f"\nJoint positions shape: {qpos.shape}")
    print(f"Joint limits shape: {qlimits.shape}")

    # Check each joint
    print("\nJoint analysis:")
    joint_names = agent.arm_joint_names if hasattr(agent, 'arm_joint_names') else []
    for i, name in enumerate(joint_names):
        if i < len(qpos[0]):
            pos = qpos[0, i].item()
            lower = qlimits[0, i, 0].item()
            upper = qlimits[0, i, 1].item()
            margin_lower = pos - lower
            margin_upper = upper - pos
            print(f"  {name}: pos={pos:.3f}, limits=[{lower:.3f}, {upper:.3f}], margins=[{margin_lower:.3f}, {margin_upper:.3f}]")

            if margin_lower < 0.1 or margin_upper < 0.1:
                print(f"    ⚠️ Joint near limit!")

    # Check TCP pose
    tcp_pose = agent.tcp.pose
    print(f"\nInitial TCP pose:")
    print(f"  Position: {tcp_pose.p.cpu().numpy()}")
    print(f"  Rotation: {tcp_pose.q.cpu().numpy()}")

    # Test controller response
    print("\n" + "=" * 60)
    print("Testing controller response to commands")
    print("=" * 60)

    # Store initial position
    initial_tcp_z = tcp_pose.p[0, 2].cpu().item()

    # Test 1: Send pure downward command
    print("\nTest 1: Pure downward command (-0.1 in Z)")
    for i in range(5):
        action = np.array([0.0, 0.0, -0.1, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        tcp_z = env.unwrapped.agent.tcp.pose.p[0, 2].cpu().item()
        delta_z = tcp_z - initial_tcp_z
        print(f"  Step {i+1}: TCP Z={tcp_z:.3f}, Total delta={delta_z:.3f}")

    # Check if we're stuck
    final_tcp_z = env.unwrapped.agent.tcp.pose.p[0, 2].cpu().item()
    total_movement = final_tcp_z - initial_tcp_z

    if abs(total_movement) < 0.05:
        print(f"\n⚠️ PROBLEM: TCP barely moved ({total_movement:.3f}m total)")
        print("Possible causes:")
        print("  1. Controller is maintaining a target pose")
        print("  2. Joint limits are preventing movement")
        print("  3. Controller gains are too high")

        # Check controller internals
        if hasattr(controller, 'ee_target_pose'):
            target = controller.ee_target_pose
            print(f"\nController target pose: {target}")

        if hasattr(controller, 'use_target'):
            print(f"Controller use_target: {controller.use_target}")

        # Check individual joint movements
        qpos_after = robot.get_qpos()
        qpos_delta = qpos_after - qpos
        print(f"\nJoint position changes:")
        for i, name in enumerate(joint_names):
            if i < len(qpos_delta[0]):
                delta = qpos_delta[0, i].item()
                print(f"  {name}: {delta:.4f}")

    env.close()

    print("\n" + "=" * 60)
    print("Diagnosis complete")
    print("=" * 60)


if __name__ == "__main__":
    diagnose_controller()