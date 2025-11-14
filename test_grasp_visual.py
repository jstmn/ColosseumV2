"""Visually test grasping the plate from the rim."""
import gymnasium as gym
import mani_skill.envs
from mani_skill.examples.motionplanning.panda.solutions.place_dish_in_rack import solve

# Create environment with visualization
env = gym.make(
    "PlaceDishInRack-v1",
    obs_mode="state",
    control_mode="pd_joint_pos",
    render_mode="human",
    num_envs=1,
)

print("Testing plate grasping with visualization...")
print("You'll be able to see the grasp attempt")
print("Press 'c' to continue at each step")
print("=" * 60)

# Run the solution with visualization
result = solve(env.unwrapped, seed=0, debug=True, vis=True)

if result != -1:
    print("\n" + "=" * 60)
    print("✓ Motion planning succeeded")
else:
    print("\n" + "=" * 60)
    print("✗ Motion planning failed")

env.close()
