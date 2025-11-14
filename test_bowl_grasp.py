#!/usr/bin/env python3
"""Test just the grasp and lift for the ceramic bowl"""

import numpy as np
import sapien
import gymnasium as gym
import mani_skill.envs

from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

# Create environment
env = gym.make(
    "PlaceDishInRack-v1",
    num_envs=1,
    obs_mode="state",
    control_mode="pd_joint_pos",
    render_mode="human",
)

# Reset
obs, info = env.reset()
env_sim = env.unwrapped

# Setup planner
robot_base_pose_batched = env_sim.agent.robot.pose
robot_base_pose = sapien.Pose(
    p=robot_base_pose_batched.p[0].cpu().numpy(),
    q=robot_base_pose_batched.q[0].cpu().numpy()
)

planner = PandaArmMotionPlanningSolver(
    env,
    debug=True,
    vis=True,
    base_pose=robot_base_pose,
    visualize_target_grasp_pose=True,
    print_env_info=True,
)

# Get bowl info
plate_p = env_sim.plate.pose.p[0].cpu().numpy()
print(f"\nBowl position: {plate_p}")

# Measured bowl dimensions
rim_radius = 0.085
rim_offset = rim_radius * 0.8
rim_height_offset = 0.016

# Grasp point at front rim
center = plate_p.copy()
center[1] -= rim_offset
center[2] += rim_height_offset

print(f"Grasp center at rim: {center}")

# Approach from above (top-down), fingers close horizontally to pinch rim
approaching = np.array([0, 0, -1])  # Top-down approach
closing = np.array([0, 1, 0])  # Fingers close in Y direction (across the rim)

grasp_pose = env_sim.agent.build_grasp_pose(approaching, closing, center)

print(f"\n=== STEP 1: REACH ===")
# Approach from 8cm above the grasp point
reach_pose = grasp_pose * sapien.Pose([0, 0, -0.08])
planner.move_to_pose_with_RRTConnect(reach_pose)
input("Press Enter to continue to grasp...")

print(f"\n=== STEP 2: GRASP ===")
planner.move_to_pose_with_RRTConnect(grasp_pose)
input("Press Enter to close gripper...")

print(f"\n=== STEP 3: CLOSE GRIPPER ===")
qpos_before = env_sim.agent.robot.get_qpos()[0, 7:9].cpu().numpy()
print(f"Gripper before close: {qpos_before}")

planner.close_gripper()

qpos_after = env_sim.agent.robot.get_qpos()[0, 7:9].cpu().numpy()
print(f"Gripper after close: {qpos_after}")

is_grasped = env_sim.agent.is_grasping(env_sim.plate)[0].item()
print(f"Is grasped: {is_grasped}")

if not is_grasped:
    print("\n❌ GRASP FAILED - bowl is not grasped")
else:
    print("\n✓ GRASP SUCCESS!")

input("Press Enter to lift...")

print(f"\n=== STEP 4: LIFT ===")
bowl_before = env_sim.plate.pose.p[0].cpu().numpy()
print(f"Bowl before lift: {bowl_before}")

lift_pose = sapien.Pose([0, 0, 0.15]) * grasp_pose
planner.move_to_pose_with_screw(lift_pose)

bowl_after = env_sim.plate.pose.p[0].cpu().numpy()
is_grasped_after = env_sim.agent.is_grasping(env_sim.plate)[0].item()
print(f"Bowl after lift: {bowl_after}")
print(f"Still grasped: {is_grasped_after}")

if is_grasped_after and bowl_after[2] > bowl_before[2] + 0.05:
    print("\n✓ LIFT SUCCESS!")
else:
    print("\n❌ LIFT FAILED")

input("\nPress Enter to close...")
planner.close()
env.close()
