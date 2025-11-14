import gymnasium as gym
from mani_skill.examples.motionplanning.panda.solutions.pour_sphere import solve

# Create environment
env = gym.make(
    "PourSphere-v1",
    obs_mode="none",
    control_mode="pd_joint_pos",
    render_mode="rgb_array",
    reward_mode="dense",
    sim_backend="auto",
)

# Run solution with debug output but no visualization window
env.reset(seed=0)
result = solve(env, seed=0, debug=True, vis=False)

print(f"\n{'='*60}")
print(f"Solution completed successfully!")
print(f"Result: {result}")
print(f"{'='*60}")

env.close()
