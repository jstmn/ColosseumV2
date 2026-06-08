import gymnasium as gym
import numpy as np
from mani_skill.utils.registration import register_env
from mani_skill.agents.robots.panda.dual_panda import DualPanda 
import torch
from mani_skill.utils.building.actors.needle import build_needle
from mani_skill.utils.building.actors.ring_tripod import build_ring_tripod
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix, quaternion_apply
from mani_skill.utils import sapien_utils
import sapien.core as sapien
from mani_skill.utils.structs.pose import Pose
from mani_skill.envs.tasks.tabletop.colosseum_v2.colosseum_v2_core import ColosseumV2Env, DisabledPerturbationFactors, PlacementRegion


@register_env("DualArmThreading-v1", max_episode_steps=1000)
class DualPandaThreadingEnv(ColosseumV2Env):
    """
    A threading task environment for Dual Panda arms.
    One arm holds the needle, the other manipulates the ring tripod.
    """
    SUPPORTED_ROBOTS = ["dual_panda_wristcam"]
    agent: DualPanda
    
    DISABLED_PERTURBATION_FACTORS = DisabledPerturbationFactors(
        MO_size=True,
        RO_size=True,
    )
    needle_length = 0.125
    ring_radius = 0.05
    pole_height = 0.15

    def __init__(self, *args, robot_uids="dual_panda_wristcam", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)


    @property
    def _default_sensor_configs(self):
        pose1 = sapien_utils.look_at(eye=[0.6, 0.0, 0.5 + 0.83], target=[-0.2, 0, 0.1 + 0.83]) # 0.83: height of the table
        pose2 = sapien_utils.look_at(eye=[-0.5, 0.0, 0.7 + 0.83], target=[-0.2, 0, 0.3 + 0.83]) # 0.83: height of the table
        return self.update_camera_configs([
            CameraConfig(
                "external1_camera",
                pose=pose1,
                width=224,
                height=224,
                fov=np.pi / 3,
                near=0.01,
                far=10,
            ),
            CameraConfig(
                "external2_camera",
                pose=pose2,
                width=224,
                height=224,
                fov=np.pi / 3,
                near=0.01,
                far=10,
            )
        ])


    @property
    def _default_human_render_camera_configs(self):
        """Configure camera for rendering videos and visualization"""
        pose = sapien_utils.look_at(eye=[0.6, 0.2, 0.4+0.83], target=[-0.1, 0, 0.1+0.83])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        """Load the needle and ring tripod actors into the scene."""
        # Build the needle
        needle_builder = lambda: build_needle(
            self.scene,
            name="needle",
            length=self.needle_length,
            shaft_radius=0.02,
            tip_length=0.05,
            density=8000.0,
            color=np.array([0.3, 0.3, 0.3, 1.0]),
            initial_pose=sapien.Pose(p=[0.0, 0.0, 0.85]),
            return_builder=True,
        )
        
        # Build the ring tripod
        ring_tripod_builder = lambda: build_ring_tripod(
            self.scene,
            name="ring_tripod",
            base_size = 0.125,
            base_thickness = 0.01,
            pole_height = self.pole_height,
            pole_radius = 0.01,
            ring_radius = self.ring_radius,
            ring_thickness = 0.01,
            density=1000.0,
            color=np.array([110/255, 38/255, 14/255, 1.0]),
            initial_pose=sapien.Pose(p=[0.3, 0.0, 0.9]),
            return_builder=True,
        )
        self.needle = self.add_asset_to_scene(needle_builder, name="needle", physics_type="dynamic", object_type="MO")
        self.ring_tripod = self.add_asset_to_scene(ring_tripod_builder, name="ring_tripod", physics_type="dynamic", object_type="RO")
        self.load_scene_hook(manipulation_objects=[self.needle], receiving_objects=[self.ring_tripod])

        self._ring_tripod_region = self.update_placement_region(
            PlacementRegion.from_center_and_width(center=(0.2, 0.0), width=(0.1, 0.1))
        )
        self._needle_region = self.update_placement_region(
            PlacementRegion.from_center_and_width(center=(-0.3, 0.0), width=(0.1, 0.1))
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Reset actor poses for each episode."""
        with torch.device(self.device):
            b = len(env_idx)
            ring_xyz = torch.zeros((b, 3), device=self.device)
            ring_xyz[..., :2] = self._ring_tripod_region.sample_xy(b, device=self.device)
            # ring_xyz[..., :2] = torch.rand((b, 2), device=self.device) * 0.2
            # ring_xyz[..., 1] -= 0.1
            ring_xyz[..., 2] = 0.9
            theta_by_2 = torch.rand(b, device=self.device) * np.pi / 12
            cos_vals = torch.cos(theta_by_2)
            sin_vals = torch.sin(theta_by_2)
            rot_q = torch.zeros((b, 4), device=self.device)
            rot_q[:, 0] = cos_vals
            rot_q[:, 3] = sin_vals
            self.ring_tripod.set_pose(Pose.create_from_pq(p=ring_xyz, q=rot_q))
            
            needle_xyz = torch.zeros((b, 3), device=self.device)
            needle_xyz[..., :2] = self._needle_region.sample_xy(b, device=self.device)
            # ring_xyz[..., :2] = torch.rand((b, 2), device=self.device) * 0.2
            # ring_xyz[..., 1] -= 0.1
            needle_xyz[..., 2] = 0.85
            self.needle.set_pose(Pose.create_from_pq(p=needle_xyz))
            self.initialize_episode_hook(env_idx, mo_pose=self.needle.pose)
        self._initialize_agent()

    def evaluate(self):
        """
        Evaluate if the needle is successfully threaded through the ring.
        This function is vectorized to support multiple parallel environments.
        """
        # Get positions and orientations
        needle_pose = self.needle.pose
        # The tripod actor's origin is at the base, the ring is at the top.
        # The offset of 0.165 is a magic number to get to the ring's center.
        ring_center_pose = self.ring_tripod.pose * sapien.Pose(p=[0, 0, self.pole_height + self.ring_radius])

        # Position of the needle tip in world frame
        tip_local_pos = torch.tensor([self.needle_length, 0, 0], device=self.device)
        needle_tip_pos = needle_pose.p + quaternion_apply(needle_pose.q, tip_local_pos)

        # --- Get Ring Plane and Center (Vectorized) ---
        ring_center = ring_center_pose.p
        # Assuming the ring's opening (normal) is along the tripod's local X-axis.
        ring_rot_mat = quaternion_to_matrix(ring_center_pose.q)
        local_x_axis = torch.tensor([1.0, 0, 0], device=self.device)
        ring_normal = ring_rot_mat @ local_x_axis

        # --- Success Criteria (Vectorized) ---
        # Check if needle eye is near the ring plane
        vec_to_ring_center = needle_tip_pos - ring_center
        distance_to_plane = torch.abs(torch.einsum('bi,bi->b', vec_to_ring_center, ring_normal))

        # Check if needle eye is within ring bounds
        projection_on_plane = needle_tip_pos - distance_to_plane.unsqueeze(1) * ring_normal
        distance_to_ring_center = torch.linalg.norm(projection_on_plane - ring_center, dim=1)

        plane_tolerance = 0.05  # 5cm tolerance
        is_near_plane = distance_to_plane < plane_tolerance
        ring_margin = 0.0
        is_within_ring = distance_to_ring_center < (self.ring_radius - ring_margin)
        success = is_near_plane * is_within_ring

        return {
            "is_near_plane": is_near_plane,
            "is_within_ring": is_within_ring,
            "distance_to_ring_center": distance_to_ring_center,
            "ring_radius - ring_margin": self.ring_radius - ring_margin,
            "success": success,
        }

    def _initialize_agent(self):
        """Reset the dual panda arms to a neutral position."""
        qpos = np.array([1.873, -1.094, 0.142, -0.935, -0.409, -2.296, -2.725, -2.236, 0.202, -2.214, 2.852, 1.062, 2.057, -1.205, 0.04, 0.04, 0.04, 0.04])
        self.agent.reset(qpos)
    
    
