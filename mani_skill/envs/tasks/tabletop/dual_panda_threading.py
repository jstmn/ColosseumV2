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
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils

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
    
    @property
    def _default_human_render_camera_configs(self):
        """Configure camera for rendering videos and visualization"""
        pose = sapien_utils.look_at(eye=[0.6, 0.2, 0.4+0.83], target=[-0.1, 0, 0.1+0.83])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    
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
            xyz[..., 0] = xyz[..., 0]
            xyz[..., 1] = xyz[..., 1] - 0.1
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
            xyz[..., 1] += 0.1
            xyz[..., 2]=0.85
            
            # theta_by_2 = (torch.rand(b))*np.pi/12  # -pi/2 to pi/2
            # self.needle.set_pose(sapien.Pose(p=list(xyz[0]), q=[float(np.cos(theta_by_2[i])),0,0,float(np.sin(theta_by_2[i]))]))
            self.needle.set_pose(sapien.Pose(p=list(xyz[0])))
            # self.ring_tripod.set_pose(sapien.Pose(p=[0.3, 0.0, 0.9]))
        self._initialize_agent()
        
    def evaluate(self):
        """
        Evaluate if the needle is successfully threaded through the ring.
        Returns a dictionary with individual checks and overall success status.
        Similar to the dual_panda_drawer_place evaluate function.
        """
        # Get positions and orientations
        needle_pose = self.needle.pose
        ring_pose = self.ring_tripod.pose # (1.0626)
        ring_pose = ring_pose * sapien.Pose(p=[0,0, 0.165])
        # -0.191119, -0.314381, 0.850000
        # Needle parameters (from build_needle call)
        needle_length = 0.1
        eye_distance_from_end = 0.02
        eye_radius = 0.01
        ring_radius = 0.03
        
        # Convert to torch tensors
        needle_pos = torch.as_tensor(needle_pose.p, dtype=torch.float32, device=self.device)
        ring_pos = torch.as_tensor(ring_pose.p, dtype=torch.float32, device=self.device)
        needle_quat = torch.as_tensor(needle_pose.q, dtype=torch.float32, device=self.device)
        ring_quat = torch.as_tensor(ring_pose.q, dtype=torch.float32, device=self.device)
        
        # Compute needle eye position in world frame
        needle_direction = self._get_needle_direction_torch(needle_quat[0])
        # needle_tip = needle_pos - needle_direction * needle_length
        needle_tip = needle_pose * sapien.Pose(p=[needle_length,0,0])
        needle_tip = needle_tip.p
        needle_eye = needle_tip - needle_direction * eye_distance_from_end
        
        # Compute ring center and normal
        ring_center = ring_pos
        ring_normal = self._get_ring_normal_torch(ring_quat[0])
        
        # Check if needle eye is near the ring plane
        vec_to_ring = needle_tip - ring_center
        # Use element-wise multiplication and sum for robust dot product
        distance_to_plane = torch.abs((vec_to_ring * ring_normal).sum())
        
        # Check if needle eye is within ring bounds
        intersection_on_plane = needle_tip - distance_to_plane * ring_normal
        # projection_on_plane = needle_eye + distance_to_plane * ring_normal
        distance_to_ring_center = torch.norm(intersection_on_plane - ring_center)
        
        # Success criteria:
        # 1. Needle eye is close to the ring plane (within tolerance)
        # 2. Needle eye projection is within the ring radius (with some margin)
        plane_tolerance = 0.05  # 5cm tolerance
        ring_margin = 0.001  # 0.1cm margin inside the ring
        
        is_near_plane = distance_to_plane < plane_tolerance
        is_within_ring = distance_to_ring_center < (ring_radius - ring_margin)
        success = is_near_plane * is_within_ring
        # print(distance_to_ring_center, needle_tip)
        # print({"dist_to_plane": distance_to_plane, "dist_to_centre": ring_radius - distance_to_ring_center - ring_margin, "success": success})
        return {"is_near_plane": is_near_plane, "is_within_ring": is_within_ring, "success": success}
    
    def _get_needle_direction_torch(self, quat):
        """Extract needle forward direction from quaternion tensor [w, x, y, z]."""
        rotation_matrix = self._quat_to_rotation_matrix_torch(quat)
        # Needle points in +Z direction (forward)
        z_direction = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device)
        needle_direction = rotation_matrix @ z_direction
        return needle_direction
    
    def _get_ring_normal_torch(self, quat):
        """Extract ring normal direction from quaternion tensor [w, x, y, z]."""
        rotation_matrix = self._quat_to_rotation_matrix_torch(quat)
        # Ring normal is along the local +X axis direction of the tripod
        x_direction = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        ring_normal = rotation_matrix @ x_direction
        return ring_normal
    
    def _quat_to_rotation_matrix_torch(self, quat):
        """Convert quaternion tensor [w, x, y, z] to 3x3 rotation matrix tensor."""
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        return torch.tensor([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ], dtype=torch.float32, device=self.device)

    def _initialize_agent(self):
        """Reset the dual panda arms to a neutral position."""
        qpos = np.array([1.873, -1.094, 0.142, -0.935, -0.409, -2.296, -2.725, -2.236, 0.202, -2.214, 2.852, 1.062, 2.057, -1.205, 0.04, 0.04, 0.04, 0.04])
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