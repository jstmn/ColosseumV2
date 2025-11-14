"""Test the PickDishFromRack environment."""
import gymnasium as gym
import mani_skill.envs

env = gym.make(
    "PickDishFromRack-v1",
    obs_mode="state",
    control_mode="pd_joint_pos",
    render_mode="human",
    num_envs=1,
)

print("=" * 60)
print("Testing PickDishFromRack-v1 Environment")
print("=" * 60)

obs, info = env.reset(seed=0)
env_sim = env.unwrapped

# Print initial state
plate_pos = env_sim.plate.pose.p[0].cpu().numpy()
rack_pos = env_sim.dish_rack.pose.p[0].cpu().numpy()

print(f"\nInitial plate position: {plate_pos}")
print(f"Rack position: {rack_pos}")
print(f"Goal position: {env_sim._plate_goal_position}")

# Check evaluation
eval_info = env_sim.evaluate()
print(f"\nInitial evaluation:")
for key, value in eval_info.items():
    print(f"  {key}: {value[0].item()}")

# Let it run for a bit to see the plate in the rack
for i in range(100):
    action = env.action_space.sample() * 0  # Zero action to keep still
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

env.close()
