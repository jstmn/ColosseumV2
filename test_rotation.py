"""Test and verify plate rotation to vertical."""
import gymnasium as gym
import mani_skill.envs
from mani_skill.examples.motionplanning.panda.solutions.place_dish_in_rack import solve
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
import torch

env = gym.make(
    "PlaceDishInRack-v1",
    obs_mode="state",
    control_mode="pd_joint_pos",
    render_mode="rgb_array",
    num_envs=1,
)

print("=" * 60)
print("Testing plate rotation to vertical orientation")
print("=" * 60)

result = solve(env.unwrapped, seed=0, debug=True, vis=False)

# Check final plate orientation
env_sim = env.unwrapped
plate = env_sim.plate
plate_quat = plate.pose.q.cpu().numpy()[0]
plate_pos = plate.pose.p.cpu().numpy()[0]

# Compute plate normal vector (Z-axis in plate local frame)
rot_mat = quaternion_to_matrix(torch.from_numpy(plate_quat).unsqueeze(0))
plate_normal = rot_mat[0, :, 2].numpy()

print("\n" + "=" * 60)
print("FINAL PLATE ORIENTATION:")
print("=" * 60)
print(f"Plate position: {plate_pos}")
print(f"Plate quaternion: {plate_quat}")
print(f"Plate normal vector: {plate_normal}")
print(f"Normal Z-component: {plate_normal[2]:.3f}")
print()

# Check if plate is vertical (normal should be horizontal, Z-component near 0)
if abs(plate_normal[2]) < 0.3:
    print("✓ SUCCESS: Plate is VERTICAL (perpendicular to table)!")
    print(f"  Normal is pointing horizontally with Z={plate_normal[2]:.3f}")
else:
    print(f"✗ Plate is still tilted. Normal Z={plate_normal[2]:.3f}")
    print(f"  (Should be near 0 for vertical orientation)")

env.close()
