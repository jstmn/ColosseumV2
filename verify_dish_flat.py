"""Visually verify the plate starts flat on the table."""
import gymnasium as gym
import mani_skill.envs

# Create environment with rendering
env = gym.make(
    "PlaceDishInRack-v1",
    render_mode="human",
    obs_mode="state"
)

# Reset environment
obs, info = env.reset()

print("✓ Environment initialized")
print("✓ Plate should be lying flat on the table")
print("✓ Dish rack should be visible on the table")
print("\nPress Ctrl+C to exit...\n")

# Hold the view for inspection
try:
    for i in range(500):
        action = env.action_space.sample() * 0  # Zero action to keep things still
        obs, reward, terminated, truncated, info = env.step(action)

        if i == 0:
            # Print initial state info
            plate_pos = env.unwrapped.plate.pose.p.cpu().numpy()[0]
            rack_pos = env.unwrapped.dish_rack.pose.p.cpu().numpy()[0]
            print(f"Plate position: {plate_pos}")
            print(f"Rack position: {rack_pos}")
            print(f"Plate height above table: {plate_pos[2]:.4f}m")
except KeyboardInterrupt:
    print("\n\nExiting...")

env.close()
print("✓ Test complete")
