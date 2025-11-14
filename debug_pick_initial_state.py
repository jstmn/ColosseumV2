"""Debug initial state of pick dish from rack."""
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
import torch

env = gym.make(
    "PickDishFromRack-v1",
    obs_mode="state",
    control_mode="pd_joint_pos",
    render_mode="human",
    num_envs=1,
)

print("=" * 60)
print("Debug Initial State")
print("=" * 60)

obs, info = env.reset(seed=0)
env_sim = env.unwrapped

plate_pos = env_sim.plate.pose.p[0].cpu().numpy()
plate_quat = env_sim.plate.pose.q[0].cpu().numpy()
rack_pos = env_sim.dish_rack.pose.p[0].cpu().numpy()

print(f"\nPlate position: {plate_pos}")
print(f"Plate quaternion: {plate_quat}")
print(f"Rack position: {rack_pos}")

# Compute plate orientation
rot_mat = quaternion_to_matrix(torch.from_numpy(plate_quat).unsqueeze(0))
plate_x = rot_mat[0, :, 0].numpy()  # X-axis in plate frame
plate_y = rot_mat[0, :, 1].numpy()  # Y-axis in plate frame
plate_z = rot_mat[0, :, 2].numpy()  # Z-axis (normal) in plate frame

print(f"\nPlate X-axis (rim direction): {plate_x}")
print(f"Plate Y-axis: {plate_y}")
print(f"Plate Z-axis (normal, should be horizontal): {plate_z}")

# The plate should be vertical, so Z-axis should be pointing horizontally
print(f"\nPlate normal Z-component: {plate_z[2]:.3f}")
if abs(plate_z[2]) < 0.3:
    print("✓ Plate is VERTICAL (as expected)")
else:
    print("✗ Plate is not vertical")

# Let it sit for visualization
for i in range(200):
    action = env.action_space.sample() * 0
    env.step(action)
    env.render()

env.close()
