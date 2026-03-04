import importlib
import os
# Directly import the specific submodule 
pm = importlib.import_module("mani_skill.utils.building.articulations.partnet_mobility")

import gymnasium as gym
import numpy as np
import sapien.core as sapien
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.agents.robots.panda.dual_panda import DualPanda 
from mani_skill.utils.structs import Pose
from mani_skill import PACKAGE_ASSET_DIR
import torch
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils import sapien_utils
from mani_skill.envs.tasks.tabletop.colosseum_v2.colosseum_v2_core import ColosseumV2Env, DisabledVariationFactors, PlacementRegion


# 1. Define the Empty Environment
@register_env("DualArmPenCap-v1", max_episode_steps=1000)
class DualArmPenCapEnv(ColosseumV2Env):
    """
    A minimal environment for Dual Panda motion planning.
    No cubes, no tasks, just the robot.
    """
    # Explicitly tell ManiSkill to use the DualPanda agent
    SUPPORTED_ROBOTS = ["dual_panda"]
    agent: DualPanda # Type hinting for IDE support

    DISABLED_VARIATION_FACTORS = DisabledVariationFactors(
        MO_size=True,
        RO_size=True,
    )


    def __init__(self, *args, robot_uids="dual_panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids,  **kwargs)
    
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.75, 0.0, 0.75 + 0.83], target=[-0.2, 0, 0.3 + 0.83]) # 0.83: height of the table
        return self.update_camera_configs([
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
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
        cap_builder_fn = lambda: self.get_glb_asset_builder(
                                        os.path.join(PACKAGE_ASSET_DIR,"pen_in_cap/cap.glb"),
                                        initial_pose=sapien.Pose(p=[0.055, -0.158, 0.], q=[0.854,0.471,0.212,0.068]),
                                        scale=[0.3,0.3,0.2],
                                        object_type="RO",
                                        )
        
        pen_builder_fn = lambda: self.get_glb_asset_builder(os.path.join(PACKAGE_ASSET_DIR, "pen_in_cap/pen.glb"),
                                          initial_pose=sapien.Pose(p=[0.055, -0.158, 0.5], q=[0.854,0.471,0.212,0.068]),
                                          scale=(0.19,0.19,0.2),
                                          object_type="MO",
                                          )

        self.cap = self.add_asset_to_scene(cap_builder_fn, name="cap", physics_type="dynamic", object_type="RO")
        self.pen = self.add_asset_to_scene(pen_builder_fn, name="pen", physics_type="dynamic", object_type="MO")
        self.load_scene_hook(manipulation_objects=[self.pen], receiving_objects=[self.cap])

        # Placement regions
        self._cap_region = self.update_placement_region(
            # Ground-truth from the legacy sampling: -torch.rand((b, 2), device=self.device) * 0.3 - 0.1
            # => x,y in [-0.4, -0.1]
            PlacementRegion(x_lims=(-0.4, -0.1), y_lims=(-0.4, -0.1))
        )
        self._pen_region = self.update_placement_region(
            # Ground-truth from the legacy sampling: torch.rand((b, 2), device=self.device) * 0.3 - 0.1
            # => x,y in [-0.1, 0.2]
            PlacementRegion(x_lims=(-0.1, 0.2), y_lims=(-0.1, 0.2))
        )


        
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            cap_xyz = torch.zeros((b, 3), device=self.device)
            # cap_xyz[..., :2] = -torch.rand((b, 2), device=self.device) * 0.3 - 0.1
            cap_xyz[..., :2] = self._cap_region.sample_xy(b, device=self.device)
            cap_xyz[..., 2] = 0.84
            q = [0, 0, 0.707, 0.707]
            self.cap.set_pose(Pose.create_from_pq(p=cap_xyz, q=q))
            
            pen_xyz = torch.zeros((b, 3), device=self.device)
            # pen_xyz[..., :2] = torch.rand((b, 2), device=self.device) * 0.3 - 0.1
            pen_xyz[..., :2] = self._pen_region.sample_xy(b, device=self.device)
            pen_xyz[..., 2] = 0.84
            q = [0, 0, 0.707, 0.707]
            self.pen.set_pose(Pose.create_from_pq(p=pen_xyz, q=q))
            self.initialize_episode_hook(env_idx, mo_pose=self.pen.pose)
        # self._initialize_agent()
        
    def _initialize_agent(self):
        # Reset the robot to a neutral position
        # Dual Panda has 14+ gripper joints. 
        # You can define a custom "qpos" (joint positions) here if you want.
        # 0-6: Left Arm, 7-8: Left Gripper, 9-15: Right Arm, 16-17: Right Gripper
        qpos = np.array([1.433, 1.45, -0.389, 0.263, -0.399, 0.79, -2.917, -2.34, -0.263, -0.315, 2.539, 2.502, -2.626, -1.437, 0.04, 0.04, 0.04, 0.04])
        # Example: Set arms to a ready position (optional)
        # qpos[0] = 0.5  # Move left shoulder
        # qpos[9] = -0.5 # Move right shoulder
        
        self.agent.reset(qpos)



    def evaluate(self):
        """
        Evaluate if the pen tip is successfully inserted into the cap.
        """
        pen_pose = self.pen.pose
        cap_pose = self.cap.pose
        
        # Get pen tip position in world frame
        pen_tip_pose = pen_pose * sapien.Pose(p=[0, 0, -0.16])
        pen_tip_pos = pen_tip_pose.p
        
        # Get cap center (opening) in world frame
        cap_opening_pose = cap_pose * sapien.Pose(p=[0.02, 0, 0.12])
        cap_center = cap_opening_pose.p
        
        # Get cap opening direction (local -Z axis pointing into the cap)
        cap_quat = cap_pose.q
        rotation_matrices = quaternion_to_matrix(cap_quat)
        neg_z_direction = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device)
        cap_direction = rotation_matrices @ neg_z_direction
        
        # Check if pen tip is along the cap opening axis
        vec_to_cap = pen_tip_pos - cap_center
        
        # Batched dot product to get projection scalar
        projection_scalar = torch.einsum('bi,bi->b', vec_to_cap, cap_direction)
        
        # Project vector onto cap direction
        projected_vec = projection_scalar.unsqueeze(1) * cap_direction
        
        # Get perpendicular distance to the axis
        perp_vec = vec_to_cap - projected_vec
        distance_to_axis = torch.linalg.norm(perp_vec, dim=1)
        
        # Check depth (how far along the cap direction the pen tip is)
        depth_into_cap = projection_scalar

        # Success criteria
        axis_tolerance = 0.02  # 2cm tolerance from cap center axis
        min_depth = 0.03  # Pen should be at least 3cm inside
        
        is_aligned = distance_to_axis < axis_tolerance
        is_deep_enough = depth_into_cap > min_depth
        success = is_aligned * is_deep_enough
        
        return {
            "is_aligned": is_aligned,
            "is_deep_enough": is_deep_enough,
            "success": success
        }

# 2. Main Execution Block
if __name__ == "__main__":
    # Now you can load this safe environment
    env = gym.make(
        "DualArmPenCap-v1", 
        robot_uids="dual_panda", # Force the dual panda
        obs_mode="state_dict", 
        control_mode="pd_joint_delta_pos",
        render_mode="human"
    )

    print("Environment Created Successfully!")
    obs, _ = env.reset()
    
    print(f"Observation Keys: {obs.keys()}")
    if "agent" in obs:
        print(f"Joint Positions Shape: {obs['agent']['qpos'].shape}")
    
    # NOW you can run your IK loop here
    # 2. You MUST run a loop, or the window will close immediately
    while True:
        env.render()  # <--- Updates the GUI
    
    env.close()