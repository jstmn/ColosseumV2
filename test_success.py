"""Test successful grasp and lift."""
import gymnasium as gym
import mani_skill.envs
from mani_skill.examples.motionplanning.panda.solutions.place_dish_in_rack import solve

env = gym.make(
    "PlaceDishInRack-v1",
    obs_mode="state",
    control_mode="pd_joint_pos",
    render_mode="rgb_array",
    num_envs=1,
)

print("=" * 60)
print("Testing plate grasp from RIGHT SIDE...")
print("=" * 60)

result = solve(env.unwrapped, seed=0, debug=True, vis=False)

env_sim = env.unwrapped
plate_final = env_sim.plate.pose.p[0].cpu().numpy()

print("\n" + "=" * 60)
print("RESULTS:")
print("=" * 60)
print(f"Final plate position: {plate_final}")
print(f"Final plate Z height: {plate_final[2]*1000:.1f}mm")

if plate_final[2] > 0.05:  # Lifted more than 50mm
    print("✓ SUCCESS: Plate was successfully lifted from the rim!")
else:
    print("✗ FAILED: Plate did not lift significantly")

env.close()
