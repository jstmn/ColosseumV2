"""Test kitchen environment with the place dish solution."""
import gymnasium as gym
import mani_skill.envs
from mani_skill.examples.motionplanning.panda.solutions.place_dish_in_rack import solve
import numpy as np

env = gym.make(
    "PlaceDishInRackKitchen-v1",
    obs_mode="state",
    control_mode="pd_joint_pos",
    render_mode="rgb_array",
    num_envs=1,
)

print("=" * 60)
print("Testing Kitchen Environment with Solution")
print("=" * 60)

result = solve(env.unwrapped, seed=0, debug=True, vis=False)

env_sim = env.unwrapped
final_plate_pos = env_sim.plate.pose.p[0].cpu().numpy()
rack_pos = env_sim.dish_rack.pose.p[0].cpu().numpy()

print("\n" + "=" * 60)
print("FINAL POSITIONS:")
print("=" * 60)
print(f"Plate: {final_plate_pos}")
print(f"Rack: {rack_pos}")
print(f"Distance to rack: {np.linalg.norm(final_plate_pos[:2] - rack_pos[:2]):.3f}m")

# Check success
success_conditions = env_sim.evaluate()
print(f"\nSuccess: {success_conditions['success'][0].item()}")
print(f"Plate close to goal: {success_conditions['plate_close_to_goal'][0].item()}")
print(f"Plate vertical: {success_conditions['plate_vertical'][0].item()}")

env.close()
