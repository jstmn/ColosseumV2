"""Check if plate is clipping into the table."""
import gymnasium as gym
import mani_skill.envs
import numpy as np

env = gym.make(
    "PlaceDishInRack-v1",
    render_mode="human",
    obs_mode="state"
)

obs, info = env.reset()

# Get positions
plate = env.unwrapped.plate
table_scene = env.unwrapped.table_scene

plate_pos = plate.pose.p.cpu().numpy()[0]
table_pose = table_scene.table.pose
table_p_arr = np.asarray(table_pose.p).ravel()
table_z = float(table_p_arr[-1])
table_top_z = table_z + float(table_scene.table_height)

# Plate dimensions
plate_half_height = env.unwrapped._plate_total_height / 2.0
plate_bottom = plate_pos[2] - plate_half_height
plate_top = plate_pos[2] + plate_half_height

print(f"Table top Z: {table_top_z:.6f}m")
print(f"Plate center Z: {plate_pos[2]:.6f}m")
print(f"Plate half-height: {plate_half_height:.6f}m")
print(f"Plate bottom Z: {plate_bottom:.6f}m")
print(f"Plate top Z: {plate_top:.6f}m")
print(f"")
print(f"Plate bottom vs table top: {plate_bottom - table_top_z:.6f}m")

if plate_bottom < table_top_z:
    penetration = table_top_z - plate_bottom
    print(f"⚠️  WARNING: Plate is penetrating table by {penetration*1000:.2f}mm")
else:
    clearance = plate_bottom - table_top_z
    print(f"✓ Plate has {clearance*1000:.2f}mm clearance above table")

# Hold view
print("\nPress Ctrl+C to exit...")
try:
    for _ in range(200):
        action = env.action_space.sample() * 0
        obs, reward, terminated, truncated, info = env.step(action)
except KeyboardInterrupt:
    pass

env.close()
