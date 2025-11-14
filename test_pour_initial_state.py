import gymnasium as gym
import mani_skill.envs  # Register ManiSkill environments

# Create environment
env = gym.make(
    "PourSphere-v1",
    obs_mode="none",
    control_mode="pd_joint_pos",
    render_mode="rgb_array",
    reward_mode="dense",
    sim_backend="auto",
)

# Reset and check initial state
env.reset(seed=0)

# Let physics settle
for i in range(100):
    env.unwrapped.scene.step()

# Check positions
env_sim = env.unwrapped
cup1_pos = env_sim.cup1.pose.p[0].cpu().numpy()
cup2_pos = env_sim.cup2.pose.p[0].cpu().numpy()
sphere_pos = env_sim.sphere.pose.p[0].cpu().numpy()

cup1_quat = env_sim.cup1.pose.q[0].cpu().numpy()
cup2_quat = env_sim.cup2.pose.q[0].cpu().numpy()

print("="*60)
print("INITIAL STATE CHECK")
print("="*60)
print(f"Cup1 position: {cup1_pos}")
print(f"Cup1 quaternion: {cup1_quat}")
print(f"Cup1 upright? {abs(cup1_quat[0] - 1.0) < 0.1}")  # w should be ~1 for identity
print()
print(f"Cup2 position: {cup2_pos}")
print(f"Cup2 quaternion: {cup2_quat}")
print(f"Cup2 upright? {abs(cup2_quat[0] - 1.0) < 0.1}")
print()
print(f"Sphere position: {sphere_pos}")
print(f"Sphere in cup1? {abs(sphere_pos[0] - cup1_pos[0]) < 0.04 and abs(sphere_pos[1] - cup1_pos[1]) < 0.04 and sphere_pos[2] > 0}")
print()
print(f"Cup height: {env_sim._cup_height}")
print(f"Cup radius: {env_sim._cup_radius}")

env.close()
