"""Test that the dish rack is now properly centered."""

import gymnasium as gym
import numpy as np

# Test with PlaceDishInRack environment
env = gym.make(
    "PlaceDishInRack-v1",
    obs_mode="state",
    render_mode="human",
    num_envs=1,
)

obs, info = env.reset()
print("\n✓ Environment created successfully")

# Get rack position
rack_pos = env.unwrapped.dish_rack.pose.p.cpu().numpy()[0]
print(f"\nRack position: {rack_pos}")
print("\nThe rack should now be centered at its specified position without any manual offsets.")
print("Visual inspection: The rack should appear properly centered in the scene.")

# Keep scene open for visual inspection
input("\nPress Enter to close...")
env.close()
