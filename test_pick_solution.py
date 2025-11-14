"""Test the pick dish from rack solution."""
import gymnasium as gym
import mani_skill.envs
from mani_skill.examples.motionplanning.panda.solutions.pick_dish_from_rack import solve
import numpy as np

env = gym.make(
    "PickDishFromRack-v1",
    obs_mode="state",
    control_mode="pd_joint_pos",
    render_mode="rgb_array",
    num_envs=1,
)

print("=" * 60)
print("Testing Pick Dish From Rack Solution")
print("=" * 60)

result = solve(env.unwrapped, seed=0, debug=True, vis=False)

env_sim = env.unwrapped
final_plate_pos = env_sim.plate.pose.p[0].cpu().numpy()
goal_pos = env_sim._plate_goal_position

print("\n" + "=" * 60)
print("FINAL POSITIONS:")
print("=" * 60)
print(f"Plate: {final_plate_pos}")
print(f"Goal: {goal_pos}")
print(f"Distance to goal: {np.linalg.norm(final_plate_pos[:2] - goal_pos[:2]):.3f}m")

# Check success
success_conditions = env_sim.evaluate()
print(f"\nSuccess: {success_conditions['success'][0].item()}")
print(f"Plate close to goal: {success_conditions['plate_close_to_goal'][0].item()}")
print(f"Plate horizontal: {success_conditions['plate_horizontal'][0].item()}")
print(f"Away from rack: {success_conditions['away_from_rack'][0].item()}")

env.close()
