import gymnasium as gym
import numpy as np
import sapien.core as sapien
from mani_skill.utils.registration import register_env
from mani_skill.agents.robots.panda.dual_panda import DualPanda 
from mani_skill.utils.structs import Pose
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply
from mani_skill.utils import sapien_utils
from mani_skill.envs.tasks.tabletop.colosseum_v2.colosseum_v2_core import ColosseumV2Env, DisabledPerturbationFactors, PlacementRegion

import torch
import os
from mani_skill import PACKAGE_ASSET_DIR

# 1. Define the Empty Environment
@register_env("DualArmLiftTray-v1", max_episode_steps=1000)
class DualArmLiftTrayEnv(ColosseumV2Env):
    """
    A minimal environment for Dual Panda motion planning.
    No cubes, no tasks, just the robot.
    """
    cube_half_size = 0.03
    # Explicitly tell ManiSkill to use the DualPanda agent
    SUPPORTED_ROBOTS = ["dual_panda_wristcam"]
    agent: DualPanda # Type hinting for IDE support
    
    DISABLED_PERTURBATION_FACTORS = DisabledPerturbationFactors(
        RO_color=True,
        RO_texture=True,
        RO_size=True,
    )

    def __init__(self, *args, robot_uids="dual_panda_wristcam", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
    
    @property
    def _default_sensor_configs(self):
        pose1 = sapien_utils.look_at(eye=[0.75, 0.0, 0.75 + 0.83], target=[-0.2, 0, 0.3 + 0.83]) # 0.83: height of the table
        pose2 = sapien_utils.look_at(eye=[-0.5, -0.25, 0.75 + 0.83], target=[-0.2, 0, 0.3 + 0.83]) # 0.83: height of the table
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
        # Load a simple floor and lighting
        tray_builder = lambda: self.get_glb_asset_builder(
            os.path.join(PACKAGE_ASSET_DIR, "pour_pot/plastic_tray.glb"),
            initial_pose=sapien.Pose(),
            object_type="MO",
            scale=(0.4,0.4,0.4),
            color=np.array([48/255, 49/255, 51/255, 1]),
        )
        self.tray = self.add_asset_to_scene(tray_builder, name="tray", physics_type="dynamic", object_type="MO")
        self.load_scene_hook(manipulation_objects=[self.tray])

        # Placement regions
        self._tray_region = self.update_placement_region(
            # Ground-truth from the legacy sampling:
            # xyz[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
            PlacementRegion.from_center_and_width(center=(0.0, 0.0), width=(0.2, 0.2))
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            xyz = torch.zeros((b, 3), device=self.device)
            xyz[..., :2] = self._tray_region.sample_xy(b, device=self.device)
            # xyz[..., :2] = torch.rand((b, 2), device=self.device) * 0.2 - 0.1
            # ^ correct, but doesn't use PlacementRegion
            xyz[..., 2] = self.cube_half_size+0.83
            theta_by_2 = torch.rand(b, device=self.device)*np.pi/8 - np.pi/16  # -pi/2 to pi/2

            base_q = torch.tensor([0.5, 0.5, 0.5, 0.5], device=self.device).repeat(b, 1)
            cos_vals = torch.cos(theta_by_2)
            sin_vals = torch.sin(theta_by_2)
            rot_q = torch.zeros((b, 4), device=self.device)
            rot_q[:, 0] = cos_vals
            rot_q[:, 3] = sin_vals
            final_q = quaternion_multiply(rot_q, base_q)
            final_pose = Pose.create_from_pq(p=xyz, q=final_q)
            self.tray.set_pose(final_pose)
            self.init_pose = final_pose
            self.initialize_episode_hook(env_idx, mo_pose=final_pose)
        
    def _initialize_agent(self):
        # Reset the robot to a neutral position
        # Dual Panda has 14+ gripper joints. 
        # You can define a custom "qpos" (joint positions) here if you want.
        # 0-6: Left Arm, 7-8: Left Gripper, 9-15: Right Arm, 16-17: Right Gripper
        qpos = np.zeros(self.agent.robot.dof)
        
        # Example: Set arms to a ready position (optional)
        # qpos[0] = 0.5  # Move left shoulder
        # qpos[9] = -0.5 # Move right shoulder
        
        self.agent.reset(qpos)



    def evaluate(self):
        curr_pose = self.tray.pose
        offset = curr_pose.p - self.init_pose.p
        is_tray_grasped_left = self.agent.is_grasping(self.tray, arm_index=1)
        is_tray_grasped_right = self.agent.is_grasping(self.tray, arm_index=2)
        is_tray_grasped = torch.logical_or(is_tray_grasped_left, is_tray_grasped_right)
        offset_x = torch.abs(offset[..., 2])
        success = torch.logical_and(offset_x > 0.15, is_tray_grasped)
        return {"left_grasped": is_tray_grasped_left, "right_grasped": is_tray_grasped_right, "grasped": is_tray_grasped, "success": success}
    