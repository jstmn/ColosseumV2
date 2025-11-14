"""Visualize the kitchen environment layout."""
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
print("Kitchen Environment Visualization")
print("=" * 60)

obs, info = env.reset(seed=0)
env_sim = env.unwrapped

plate_pos = env_sim.plate.pose.p[0].cpu().numpy()
rack_pos = env_sim.dish_rack.pose.p[0].cpu().numpy()

print(f"\nKitchen Layout:")
print(f"  Counter: 1.2m x 0.6m at height 0.9m")
print(f"  Backsplash: Behind counter (Y = -0.3m)")
print(f"  Upper Cabinet: Above counter at ~1.5m height")
print(f"\nObject Positions:")
print(f"  Plate: {plate_pos}")
print(f"  Rack: {rack_pos}")
print(f"  Robot base: [-0.615, 0, 0.7]")

print("\n" + "=" * 60)
print("The kitchen has:")
print("- Gray counter surface (granite/marble look)")
print("- Dark wood cabinet base")
print("- White backsplash wall")
print("- Dark wood upper cabinet")
print("=" * 60)

# Run for visualization
for i in range(1000):
    action = env.action_space.sample() * 0
    env.step(action)
    env.render()

env.close()
