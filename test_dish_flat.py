"""Test that the plate starts flat on the table in PlaceDishInRack task."""
import gymnasium as gym
import numpy as np
import mani_skill.envs

# Create environment
env = gym.make(
    "PlaceDishInRack-v1",
    render_mode="human",
    obs_mode="state"
)

# Reset and check initial plate orientation
obs, info = env.reset()

# Get the plate actor
plate = env.unwrapped.plate

# Get plate pose
plate_pose = plate.pose
plate_quat = plate_pose.q.cpu().numpy()[0]  # Get quaternion
plate_pos = plate_pose.p.cpu().numpy()[0]   # Get position

print(f"Plate position: {plate_pos}")
print(f"Plate quaternion (w,x,y,z): {plate_quat}")

# Compute plate normal direction (z-axis in plate frame)
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
import torch
rot_mat = quaternion_to_matrix(torch.tensor([plate_quat]))
plate_normal = rot_mat[0, :, 2].numpy()  # Z-axis column

print(f"Plate normal direction: {plate_normal}")
print(f"Plate normal Z-component: {plate_normal[2]}")

# For a plate lying flat, the normal should point mostly in +Z direction
# (i.e., Z-component should be close to 1.0)
if abs(plate_normal[2]) > 0.9:
    print("✓ SUCCESS: Plate is lying flat on the table!")
else:
    print(f"✗ FAIL: Plate is NOT flat. Normal Z={plate_normal[2]:.3f} (should be ~1.0)")

# Visualize for a few steps
for _ in range(100):
    action = env.action_space.sample() * 0  # Zero action
    obs, reward, terminated, truncated, info = env.step(action)

env.close()
