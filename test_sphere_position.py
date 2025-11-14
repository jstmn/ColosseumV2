import gymnasium as gym
from mani_skill.examples.motionplanning.panda.solutions.pour_sphere import solve

# Create environment with rendering
env = gym.make(
    "PourSphere-v1",
    obs_mode="none",
    control_mode="pd_joint_pos",
    render_mode="human",
    reward_mode="dense",
    sim_backend="auto",
)

# Reset and check initial positions
env.reset(seed=0)

# Get positions
env_sim = env.unwrapped
cup1_pos = env_sim.cup1.pose.p[0].cpu().numpy()
cup2_pos = env_sim.cup2.pose.p[0].cpu().numpy()
sphere_pos = env_sim.sphere.pose.p[0].cpu().numpy()

print(f"\nInitial Positions:")
print(f"Cup1: {cup1_pos}")
print(f"Cup2: {cup2_pos}")
print(f"Sphere: {sphere_pos}")
print(f"\nSphere relative to Cup1: {sphere_pos - cup1_pos}")
print(f"Sphere height above table: {sphere_pos[2]:.4f}m")
print(f"Cup1 height: {env_sim._cup_height}m")

# Let physics settle for a bit
print("\nLetting physics settle...")
for _ in range(100):
    env_sim.scene.step()

sphere_pos_after = env_sim.sphere.pose.p[0].cpu().numpy()
print(f"\nSphere position after settling: {sphere_pos_after}")
print(f"Sphere moved: {sphere_pos_after - sphere_pos}")

# Run solution
print("\n" + "="*60)
print("Running solution...")
print("="*60)
result = solve(env, seed=0, debug=False, vis=True)

# Check final positions
sphere_final = env_sim.sphere.pose.p[0].cpu().numpy()
print(f"\nFinal Positions:")
print(f"Sphere: {sphere_final}")
print(f"Sphere relative to Cup2: {sphere_final - cup2_pos}")

# Check success
eval_result = env_sim.evaluate()
print(f"\nSuccess: {eval_result['success'][0].item()}")

env.close()
