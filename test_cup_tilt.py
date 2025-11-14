import gymnasium as gym
from mani_skill.examples.motionplanning.panda.solutions.pour_sphere import solve

# Create environment
env = gym.make(
    "PourSphere-v1",
    obs_mode="none",
    control_mode="pd_joint_pos",
    render_mode="human",
    reward_mode="dense",
    sim_backend="auto",
)

# Run solution with debug output and visualization
env.reset(seed=0)
result = solve(env, seed=0, debug=True, vis=True)

print(f"\n{'='*60}")
print(f"Solution result: {result}")

env.close()
