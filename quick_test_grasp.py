#!/usr/bin/env python
"""Quick test to verify grasp checking works"""
import gymnasium as gym
import numpy as np
import mani_skill.envs

print("Creating environment...")
env = gym.make(
    "PlaceDishInRack-v1",
    obs_mode="state",
    control_mode="pd_joint_pos",
    render_mode=None,
    reward_mode="dense"
)

print("Resetting environment...")
env.reset(seed=0)
env_sim = env.unwrapped

print(f"Plate position: {env_sim.plate.pose.p[0].cpu().numpy()}")

# Test with base joint config
grasp_qpos = np.array([-0.0, 0.377, -0.195, -2.7, 0.0, 3.078, 0.55])
print(f"Testing grasp with qpos: {grasp_qpos}")

action = np.concatenate([grasp_qpos, [0.0]])
for _ in range(100):
    env.step(action)

tcp_pos = env_sim.agent.tcp.pose.p[0].cpu().numpy()
plate_pos = env_sim.plate.pose.p[0].cpu().numpy()
dist = np.linalg.norm(tcp_pos - plate_pos)

print(f"TCP position: {tcp_pos}")
print(f"Plate position: {plate_pos}")
print(f"Distance: {dist:.4f}m")

# Close gripper
action_close = np.concatenate([grasp_qpos, [0.04]])
for _ in range(50):
    env.step(action_close)

is_grasped = env_sim.agent.is_grasping(env_sim.plate)[0].item()
print(f"Is grasped: {is_grasped}")

env.close()
print("Done!")
