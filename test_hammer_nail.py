"""Quick test to visualize HammerNail environment"""
import gymnasium as gym
import mani_skill.envs

env = gym.make("HammerNail-v1", render_mode="human", obs_mode="none")
env.reset()

print("Hammer position:", env.unwrapped.hammer.pose.p)
print("Nail position:", env.unwrapped.nails[0].pose.p)
print("Block position:", env.unwrapped.block.pose.p)

# Run for a few steps to visualize
for _ in range(500):
    action = env.action_space.sample() * 0  # Zero action
    env.step(action)

env.close()
