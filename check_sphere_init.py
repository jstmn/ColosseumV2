import gymnasium as gym
import mani_skill.envs

env = gym.make(
    "PourSphere-v1",
    obs_mode="none",
    control_mode="pd_joint_pos",
    render_mode="rgb_array",
    reward_mode="dense",
    sim_backend="auto",
)

env.reset(seed=0)
env_sim = env.unwrapped

cup1_pos = env_sim.cup1.pose.p[0].cpu().numpy()
cup2_pos = env_sim.cup2.pose.p[0].cpu().numpy()
sphere_pos = env_sim.sphere.pose.p[0].cpu().numpy()

print(f"\nInitial Positions:")
print(f"Cup1: {cup1_pos}")
print(f"Cup2: {cup2_pos}")
print(f"Sphere: {sphere_pos}")
print(f"Cup height: {env_sim._cup_height}m")
print(f"Sphere radius: {env_sim._sphere_radius}m")
print(f"\nSphere relative to Cup1 center: {sphere_pos - cup1_pos}")

# Let physics settle
for _ in range(100):
    env_sim.scene.step()

sphere_pos_after = env_sim.sphere.pose.p[0].cpu().numpy()
print(f"\nSphere after settling: {sphere_pos_after}")
print(f"Sphere moved by: {sphere_pos_after - sphere_pos}")

# Check if sphere is still in cup1
dist_from_cup1 = ((sphere_pos_after[:2] - cup1_pos[:2])**2).sum()**0.5
print(f"\nSphere XY distance from cup1 center: {dist_from_cup1:.4f}m")
print(f"Cup radius: {env_sim._cup_radius}m")
print(f"Sphere in cup1: {dist_from_cup1 < env_sim._cup_radius}")

env.close()
