"""Check plate position immediately after reset, before any steps."""
import gymnasium as gym
import mani_skill.envs
import torch

env = gym.make(
    "PlaceDishInRack-v1",
    obs_mode="state",
    num_envs=1
)

# Reset environment
obs, info = env.reset()

# Check position IMMEDIATELY after reset, before any steps
plate = env.unwrapped.plate
plate_pose = plate.pose
plate_pos = plate_pose.p.cpu().numpy()[0]
plate_quat = plate_pose.q.cpu().numpy()[0]

# Compute plate normal
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
rot_mat = quaternion_to_matrix(torch.from_numpy(plate_quat).unsqueeze(0))
plate_normal = rot_mat[0, :, 2].numpy()

print("=== IMMEDIATELY AFTER RESET (before any steps) ===")
print(f"Plate position: {plate_pos}")
print(f"Plate quaternion: {plate_quat}")
print(f"Plate normal: {plate_normal}")
print(f"Plate Z height: {plate_pos[2]:.6f}m")
print(f"Expected Z height: ~0.013m (plate half-height)")

# Check velocity
lin_vel = plate.linear_velocity.cpu().numpy()[0]
ang_vel = plate.angular_velocity.cpu().numpy()[0]
print(f"\nLinear velocity: {lin_vel}")
print(f"Angular velocity: {ang_vel}")

# Now step once with zero action
action = env.action_space.sample() * 0
obs, reward, terminated, truncated, info = env.step(action)

plate_pos_after = plate.pose.p.cpu().numpy()[0]
print(f"\n=== AFTER 1 STEP ===")
print(f"Plate position: {plate_pos_after}")
print(f"Position change: {plate_pos_after - plate_pos}")

env.close()
