#!/usr/bin/env python3
"""Test script to verify the procedural plate loads correctly in the PlaceDishInRack environment"""

import gymnasium as gym
from mani_skill.envs.tasks.tabletop.place_dish_in_rack import PlaceDishInRackEnv

def test_plate_loading():
    """Test that the procedural plate loads without errors"""
    print("Creating PlaceDishInRack environment with procedural plate...")

    env = gym.make(
        "PlaceDishInRack-v1",
        num_envs=1,
        obs_mode="state",
        render_mode="rgb_array",
    )

    print("Resetting environment...")
    obs, info = env.reset()

    # Get the plate actor
    plate = env.unwrapped.plate
    print(f"\nPlate actor created: {plate.name}")
    print(f"Plate pose: {plate.pose}")

    # Check the plate's collision shapes
    if hasattr(plate, "get_collision_shapes"):
        collision_shapes = plate.get_collision_shapes()
    elif hasattr(plate, "collision_shapes"):
        collision_shapes = plate.collision_shapes
    else:
        collision_shapes = []

    print(f"\nNumber of collision shapes: {len(collision_shapes)}")
    for i, shape in enumerate(collision_shapes):
        print(f"  Shape {i}: {type(shape).__name__}")

    # Check the plate's visual shapes
    if hasattr(plate, "get_visual_bodies"):
        visual_bodies = plate.get_visual_bodies()
    elif hasattr(plate, "visual_bodies"):
        visual_bodies = plate.visual_bodies
    else:
        visual_bodies = []

    print(f"\nNumber of visual bodies: {len(visual_bodies)}")

    # Test a few steps
    print("\nRunning a few simulation steps...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

    plate_pos = plate.pose.p.cpu().numpy()
    print(f"\nPlate position after 10 steps: {plate_pos}")

    env.close()
    print("\n✓ Plate loaded successfully!")

if __name__ == "__main__":
    test_plate_loading()
