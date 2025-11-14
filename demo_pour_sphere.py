"""
Simple demo to visualize the PourSphere environment with random actions.
Usage: python demo_pour_sphere.py
"""
import gymnasium as gym
import numpy as np
import mani_skill.envs

def main():
    env = gym.make(
        "PourSphere-v1",
        obs_mode="state",
        control_mode="pd_joint_pos",
        render_mode="human",
    )

    print("PourSphere Environment Demo")
    print("Press ESC or close window to exit")
    print("=" * 60)

    obs, info = env.reset(seed=0)

    for step in range(500):
        # Small random actions to move the robot slightly
        action = env.action_space.sample() * 0.1
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()
            print(f"Episode reset at step {step}")

    env.close()
    print("Demo finished")

if __name__ == "__main__":
    main()
