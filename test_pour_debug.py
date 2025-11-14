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

# Run solution with debug output
env.reset(seed=0)
result = solve(env, seed=0, debug=True, vis=False)

print(f"\n{'='*60}")
print(f"Solution result: {result}")

# Check final evaluation
eval_result = env.unwrapped.evaluate()
print(f"Success: {eval_result['success'][0].item()}")
print(f"Sphere in cup2 radius: {eval_result['sphere_in_cup2_radius'][0].item()}")
print(f"Sphere in cup2 height: {eval_result['sphere_in_cup2_height'][0].item()}")
print(f"Sphere static: {eval_result['sphere_static'][0].item()}")

env.close()
