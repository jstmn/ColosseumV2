import gymnasium as gym
import numpy as np
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.agents.robots.panda.dual_panda import DualPanda 
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.building import actors
import torch
import sys
from mani_skill.utils.building.actors.needle import build_needle
from mani_skill.utils.building.actors.ring_tripod import build_ring_tripod

# filepath: /home/prajwal-vijay/Documents/ManiSkill/mani_skill/envs/tasks/tabletop/dual_panda_threading.py
import sapien.core as sapien
import mani_skill.agents.robots.panda.dual_panda
from mani_skill.utils.structs.pose import Pose
sys.path.insert(0, '/home/prajwal-vijay/Documents/ManiSkill/mani_skill/utils/building/actors')


@register_env("DualPandaThreading-v0", max_episode_steps=1000)
class DualPandaThreadingEnv(BaseEnv):
    """
    A threading task environment for Dual Panda arms.
    One arm holds the needle, the other manipulates the ring tripod.
    """
    SUPPORTED_ROBOTS = ["dual_panda"]
    agent: DualPanda
    
    def __init__(self, *args, robot_uids="dual_panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
    
    def _load_scene(self, options: dict):
        """Load the needle and ring tripod actors into the scene."""
        # Build the needle
        self.needle = build_needle(
            self.scene,
            name="needle",
            length=0.1,
            shaft_radius=0.01,
            tip_length=0.05,
            eye_radius=0.01,
            eye_distance_from_end=0.02,
            density=8000.0,
            color=np.array([0.6, 0.6, 0.65, 1.0]),
            initial_pose=sapien.Pose(p=[0.0, 0.0, 0.85]),
        )
        
        # Build the ring tripod
        self.ring_tripod = build_ring_tripod(
            self.scene,
            name="ring_tripod",
            base_size = 0.15,
            base_thickness = 0.01,
            pole_height = 0.12,
            pole_radius = 0.008,
            ring_radius = 0.03,
            ring_thickness = 0.01,
            density=1000.0,
            color=np.array([110/255, 38/255, 14/255, 1.0]),
            initial_pose=sapien.Pose(p=[0.3, 0.0, 0.9]),
        )
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Reset actor poses for each episode."""
        with torch.device(self.device):
            b = len(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[..., :2] = torch.rand((b, 2)) * 0.2
            xyz[..., 0] = xyz[..., 0] - 0.1
            xyz[..., 1] = xyz[..., 1] + 0.1
            xyz[..., 2] = 0.9
            theta_by_2 = (torch.rand(b))*np.pi/12  # -pi/2 to pi/2
            for i in range(b):
                init_pose = Pose.create_from_pq(p=xyz[i:i+1],q=[1,0,0,0])
                # Convert tensors to numpy float32 arrays
                p_np = init_pose.p.squeeze(0).cpu().numpy().astype(np.float32)
                q_np = init_pose.q.squeeze(0).cpu().numpy().astype(np.float32)
                init_pose_sapien = sapien.Pose(p=p_np, q=q_np)
                rotation_pose = sapien.Pose(p=[0,0,0],q=[float(np.cos(theta_by_2[i])),0,0,float(np.sin(theta_by_2[i]))])
                init_pose_sapien = rotation_pose * init_pose_sapien
                self.ring_tripod.set_pose(init_pose_sapien)
                
            xyz[..., :2] = torch.rand((b, 2)) * 0.3 - 0.4
            xyz[..., 2]=0.85
            # theta_by_2 = (torch.rand(b))*np.pi/12  # -pi/2 to pi/2
            # self.needle.set_pose(sapien.Pose(p=list(xyz[0]), q=[float(np.cos(theta_by_2[i])),0,0,float(np.sin(theta_by_2[i]))]))
            self.needle.set_pose(sapien.Pose(p=list(xyz[0])))
            # self.ring_tripod.set_pose(sapien.Pose(p=[0.3, 0.0, 0.9]))
    
    def _initialize_agent(self):
        """Reset the dual panda arms to a neutral position."""
        qpos = np.zeros(self.agent.robot.dof)
        self.agent.reset(qpos)
    
    def _get_obs_extra(self, info: dict):
        """Return observation data for both arms."""
        obs = dict()
        
        def pose_to_vec(pose):
            return np.hstack([pose.p, pose.q])
        
        # TCP poses for both arms
        obs["tcp_pose_left"] = pose_to_vec(self.agent.tcp_1_pose)
        obs["tcp_pose_right"] = pose_to_vec(self.agent.tcp_2_pose)
        
        # Object poses
        obs["needle_pose"] = pose_to_vec(self.needle.pose)
        obs["ring_tripod_pose"] = pose_to_vec(self.ring_tripod.pose)
        
        return obs
    
    def compute_dense_reward(self, obs, action, info):
        """Compute reward (placeholder for now)."""
        return 0.0
    
    def compute_normalized_dense_reward(self, obs, action, info):
        """Compute normalized reward."""
        return 0.0


if __name__ == "__main__":
    env = gym.make(
        "DualPandaThreading-v0",
        robot_uids="dual_panda",
        obs_mode="state_dict",
        control_mode="pd_joint_delta_pos",
        render_mode="human"
    )
    
    print("Threading Environment Created Successfully!")
    obs, _ = env.reset()
    
    print(f"Observation Keys: {obs.keys()}")
    
    while True:
        env.render()