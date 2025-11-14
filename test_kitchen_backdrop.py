"""Test the kitchen backdrop environment."""
import gymnasium as gym
import mani_skill.envs

env = gym.make(
    "PlaceDishInRackKitchen-v1",
    obs_mode="state",
    control_mode="pd_joint_pos",
    render_mode="human",
    num_envs=1,
)

print("=" * 60)
print("Testing Kitchen Backdrop Environment")
print("=" * 60)

obs, info = env.reset(seed=0)
env_sim = env.unwrapped

# Print scene info
plate_pos = env_sim.plate.pose.p[0].cpu().numpy()
rack_pos = env_sim.dish_rack.pose.p[0].cpu().numpy()
counter_height = env_sim.kitchen_scene.counter_top_height

print(f"\nCounter height: {counter_height}m")
print(f"Plate position: {plate_pos}")
print(f"Rack position: {rack_pos}")

# Let it run for visualization
print("\nViewing kitchen scene with counter, backsplash, and upper cabinet...")
print("The plate and dish rack should be on the kitchen counter.")

for i in range(500):
    action = env.action_space.sample() * 0  # Zero action to keep still
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

env.close()

print("\n✓ Kitchen backdrop created successfully!")
