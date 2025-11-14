import gymnasium as gym
import mani_skill.envs
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

# Reset and run solution
env.reset(seed=0)
env_sim = env.unwrapped

print("Initial state:")
sphere_pos = env_sim.sphere.pose.p[0].cpu().numpy()
print(f"  Sphere in cup1: {sphere_pos}")

# Run solution
print("\nRunning solution...")
result = solve(env, seed=0, debug=False, vis=False)

# Check final state
sphere_final = env_sim.sphere.pose.p[0].cpu().numpy()
cup2_pos = env_sim.cup2.pose.p[0].cpu().numpy()

print(f"\nFinal state:")
print(f"  Sphere position: {sphere_final}")
print(f"  Cup2 position: {cup2_pos}")
print(f"  Sphere relative to cup2: {sphere_final - cup2_pos}")

# Check success
eval_result = env_sim.evaluate()
success = eval_result['success'][0].item()

print(f"\n{'='*60}")
print(f"SUCCESS: {success}")
print(f"{'='*60}")

env.close()
