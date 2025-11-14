"""Test grasping the plate from the rim without interactive viewer."""
import gymnasium as gym
import mani_skill.envs
from mani_skill.examples.motionplanning.panda.solutions.place_dish_in_rack import solve

# Create environment without human render mode
env = gym.make(
    "PlaceDishInRack-v1",
    obs_mode="state",
    control_mode="pd_joint_pos",
    render_mode="rgb_array",  # Use rgb_array instead of human
    num_envs=1,
)

print("Testing plate grasping from rim...")
print("=" * 60)

# Run the solution without visualization
result = solve(env.unwrapped, seed=0, debug=True, vis=False)

if result != -1:
    print("\n" + "=" * 60)
    print("✓ SUCCESS: Plate was grasped from rim and lifted!")
    print("=" * 60)
else:
    print("\n" + "=" * 60)
    print("✗ FAILED: Could not grasp or lift the plate")
    print("=" * 60)

env.close()
