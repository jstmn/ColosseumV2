"""Detailed test to see if plate lifts with gripper."""
import gymnasium as gym
import mani_skill.envs
import numpy as np

env = gym.make(
    "PlaceDishInRack-v1",
    obs_mode="state",
    control_mode="pd_joint_pos",
    render_mode="rgb_array",
    num_envs=1,
)

obs, info = env.reset(seed=0)

plate = env.unwrapped.plate
agent = env.unwrapped.agent

print("=" * 60)
print("INITIAL STATE:")
print("=" * 60)
plate_pos_init = plate.pose.p.cpu().numpy()[0]
print(f"Plate position: {plate_pos_init}")
print(f"Plate Z: {plate_pos_init[2]*1000:.3f}mm")

# Manually move gripper to grasp position
plate_outer_radius = env.unwrapped._plate_outer_radius
plate_inner_radius = env.unwrapped._plate_inner_radius
plate_base_thickness = env.unwrapped._plate_base_thickness
plate_rim_height = env.unwrapped._plate_rim_height

rim_grasp_radius = (plate_outer_radius + plate_inner_radius) / 2.0
grasp_x = plate_pos_init[0]
grasp_y = plate_pos_init[1] + rim_grasp_radius
grasp_z = plate_pos_init[2] + plate_base_thickness + plate_rim_height * 0.2

print(f"\nGrasp target: [{grasp_x:.3f}, {grasp_y:.3f}, {grasp_z:.3f}]")
print(f"Rim radius: {rim_grasp_radius*1000:.1f}mm")

# Check if plate is being grasped
for step in range(100):
    action = env.action_space.sample() * 0
    obs, reward, terminated, truncated, info = env.step(action)

    if step % 20 == 0:
        plate_pos = plate.pose.p.cpu().numpy()[0]
        is_grasped = agent.is_grasping(plate)[0].item()
        print(f"Step {step}: Plate Z={plate_pos[2]*1000:.3f}mm, Grasped={is_grasped}")

env.close()
print("\n" + "=" * 60)
print("Plate is not moving - grasp may not be secure")
print("=" * 60)
