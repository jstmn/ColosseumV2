#!/usr/bin/env python3
import gymnasium as gym
import mani_skill.envs

env = gym.make(
    "PourSphere-v1",
    render_mode="human",
)

print("Resetting environment...")
obs, info = env.reset(seed=0)
print("Environment ready! You should see a window with two cups and a sphere.")

# Just hold still so you can see it
for _ in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample() * 0.0)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
