"""Create a demo video of the PourSphere environment"""
import gymnasium as gym
import numpy as np
import mani_skill.envs

env = gym.make(
    "PourSphere-v1",
    obs_mode="none",
    render_mode="rgb_array",
    num_envs=1,
)

from mani_skill.utils.wrappers.record import RecordEpisode

env = RecordEpisode(
    env,
    output_dir="demos/PourSphere-v1/fixed",
    save_video=True,
    info_on_video=True,
)

print("Recording PourSphere environment...")
obs, info = env.reset(seed=0)

for i in range(200):
    # Small random actions
    action = env.action_space.sample() * 0.1
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break

env.flush_video()
env.close()
print("✓ Video saved to demos/PourSphere-v1/fixed/")
