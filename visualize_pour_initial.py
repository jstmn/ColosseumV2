import gymnasium as gym
import mani_skill.envs  # Register ManiSkill environments

# Create environment with rendering
env = gym.make(
    "PourSphere-v1",
    obs_mode="none",
    control_mode="pd_joint_pos",
    render_mode="human",  # Human rendering to see it
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

print("="*60)
print("VISUAL CHECK - Look at the viewer window!")
print("="*60)
print(f"Cup1 center: {cup1_pos}")
print(f"Sphere center: {sphere_pos}")
print(f"")
print(f"Cup height: {env_sim._cup_height}m")
print(f"Cup radius: {env_sim._cup_radius}m")
print(f"Sphere radius: {env_sim._sphere_radius}m")
print(f"")
print(f"Cup1 bottom Z: {cup1_pos[2] - env_sim._cup_height/2:.4f}m")
print(f"Cup1 top Z: {cup1_pos[2] + env_sim._cup_height/2:.4f}m")
print(f"Sphere bottom Z: {sphere_pos[2] - env_sim._sphere_radius:.4f}m")
print(f"Sphere top Z: {sphere_pos[2] + env_sim._sphere_radius:.4f}m")
print(f"")
print("Press Enter to close...")

input()

env.close()
