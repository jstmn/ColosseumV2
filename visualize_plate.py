"""Visualize the plate and its dimensions."""
import gymnasium as gym
import mani_skill.envs
import numpy as np

env = gym.make(
    "PlaceDishInRack-v1",
    render_mode="human",
    obs_mode="state"
)

obs, info = env.reset()

plate = env.unwrapped.plate
plate_pos = plate.pose.p.cpu().numpy()[0]

print("=" * 60)
print("PLATE DIMENSIONS:")
print("=" * 60)
print(f"Outer radius: {env.unwrapped._plate_outer_radius:.4f}m ({env.unwrapped._plate_outer_radius*1000:.1f}mm)")
print(f"Inner radius: {env.unwrapped._plate_inner_radius:.4f}m ({env.unwrapped._plate_inner_radius*1000:.1f}mm)")
print(f"Rim thickness: {(env.unwrapped._plate_outer_radius - env.unwrapped._plate_inner_radius)*1000:.1f}mm")
print(f"Base thickness: {env.unwrapped._plate_base_thickness*1000:.1f}mm")
print(f"Rim height: {env.unwrapped._plate_rim_height*1000:.1f}mm")
print(f"Total height: {env.unwrapped._plate_total_height*1000:.1f}mm")
print(f"\nPlate position: {plate_pos}")
print(f"Plate Z: {plate_pos[2]*1000:.2f}mm")
print("\nVisualizing plate... Press Ctrl+C to exit")

try:
    for i in range(500):
        action = env.action_space.sample() * 0
        obs, reward, terminated, truncated, info = env.step(action)
except KeyboardInterrupt:
    pass

env.close()
