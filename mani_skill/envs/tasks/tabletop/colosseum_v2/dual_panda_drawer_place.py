import gymnasium as gym
import numpy as np
import sapien.core as sapien
import mani_skill.agents.robots.panda.dual_panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.agents.robots.panda.dual_panda import DualPanda 
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.building import actors
from mani_skill.utils.building import articulations
import torch
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
import os
from mani_skill.utils.structs import Pose
from mani_skill.envs.tasks.tabletop.colosseum_v2.distraction_set import DistractionSet

@register_env("DualArmDrawerPlace-v1", max_episode_steps=1000, asset_download_ids=["partnet_mobility_cabinet"])
class DualArmDrawerPlaceEnv(BaseEnv):
    """
    Uses PartNet-Mobility dataset (ID 1005).
    """
    cube_half_size = 0.02
    # Explicitly tell ManiSkill to use the DualPanda agent
    SUPPORTED_ROBOTS = ["dual_panda"]
    agent: DualPanda # Type hinting for IDE support
    
    def __init__(self, *args, robot_uids="dual_panda", **kwargs):
        distraction_set: DistractionSet | dict | None = kwargs.pop("distraction_set", None)
        self._distraction_set: DistractionSet | None = DistractionSet(**distraction_set) if isinstance(distraction_set, dict) else distraction_set
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        if self.scene is not None:
            print(f"Is GPU simulation enabled for this scene? {self.scene.gpu_sim_enabled}")
    
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[-0.3, 0.5, 1.0+0.83], target=[0.1, 0, 0.1+0.83])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 3,
                near=0.01,
                far=10,
            )
        ]
    @property
    def _default_human_render_camera_configs(self):
        """Configure camera for rendering videos and visualization"""
        pose = sapien_utils.look_at(eye=[-0.6, -0.2, 0.4+0.83], target=[0.1, 0, 0.1+0.83])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)
        

    def _load_scene(self, options: dict):
        # Load PartNet-Mobility Drawer (ID 1005 is a standard table with drawer)
        model_id = "1005"
        builder = articulations.get_articulation_builder(
            self.scene, f"partnet-mobility:{model_id}"
        )
        
        # Set initial pose to match the previous drawer's location
        # Position: [0.25, 0, 1.256] (0.456 + 0.8)
        # Orientation: 90 degree rotation around X (q=[0.7071, 0, 0, -0.7071])
        builder.initial_pose = sapien.Pose(
            p=[0.25, 0, 0.456+0.8], 
            q=[0.7071, 0, 0, -0.7071]
        )
        self.obj = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array([12, 42, 160, 255]) / 255,
            name="cube",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[-0.2, -0.141, 0.83+self.cube_half_size]),
        )
        scene_idxs = [i for i in range(self.num_envs)]
        builder.set_scene_idxs(scene_idxs=scene_idxs)
        self.open_cabinet = builder.build(name=f"drawer-{model_id}")
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            # Reset cabinet pose
            xyz = torch.zeros((b, 3), device=self.device)
            xyz[..., :2] = -torch.rand((b, 2), device=self.device) * 0.2 + 0.1
            xyz[..., 0] += 0.2
            xyz[..., 2] = 0.456+0.8
            theta_by_2 = torch.rand(b, device=self.device)*np.pi/16 - np.pi/32
            
            dof_tensor = self.open_cabinet.dof
            if isinstance(dof_tensor, torch.Tensor):
                dof = int(dof_tensor.flatten()[0].cpu().item())
            else:
                dof = int(dof_tensor)

            # Vectorized pose setting for the cabinet
            cos_vals = torch.cos(theta_by_2)
            sin_vals = torch.sin(theta_by_2)
            qs = torch.zeros((b, 4), device=self.device)
            qs[:, 0] = cos_vals
            qs[:, 3] = sin_vals
            cabinet_pose = Pose.create_from_pq(p=xyz, q=qs)
            self.open_cabinet.set_pose(cabinet_pose)
            
            # Vectorized: Set all object poses
            obj_xyz = torch.rand((b, 2), device=self.device) * 0.1 - 0.05 - 0.25
            obj_z = torch.full((b, 1), 0.85, device=self.device)
            obj_poses_xyz = torch.cat([obj_xyz, obj_z], dim=1)
            self.obj.set_pose(Pose.create_from_pq(p=obj_poses_xyz))
            
            # Close the drawers
            self.open_cabinet.set_qpos(torch.zeros((b, dof), device=self.device))
            self.open_cabinet.set_qvel(torch.zeros((b, dof), device=self.device))
        
        self._initialize_agent()
        
    def _initialize_agent(self):
        # Reset the robot to a neutral position
        qpos = np.array([1.326, 1.373, -0.15, -0.569, -0.305, -0.1, -2.887, -2.768, -0.115, 1.35, 2.742, 1.358, 0.345, 3.281, 0.04, 0.04, 0.04, 0.04])
        self.agent.reset(qpos)

    def evaluate(self):
        box_pos = self.obj.pose.p
        drawer_pos = self.open_cabinet.pose.p
        
        above_ground = box_pos[..., 2] > 0.9
        inside = torch.norm(box_pos[..., :2] - drawer_pos[..., :2]) < 0.25
        success = above_ground * inside
        return {"above_ground": above_ground, "inside": inside, "success": success}
        
    def _get_obs_extra(self, info: dict):
        obs = dict()
        
        # Helper to concatenate Pose p and q while keeping them on the correct device
        def pose_to_vec(pose):
            # p and q are already tensors on the correct device (GPU)
            # We just need to concatenate them using torch instead of numpy
            return torch.cat([pose.p, pose.q], dim=-1)
        
        if hasattr(self.agent, "tcp_pose"):
             obs["tcp_pose"] = self.agent.tcp_pose.raw_pose
        else:
            obs["left_arm_tcp"] = pose_to_vec(self.agent.tcp_1_pose)
            obs["right_arm_tcp"] = pose_to_vec(self.agent.tcp_2_pose)

        return obs
    
    def compute_dense_reward(self, obs, action, info):
        # Return 0 since we are not training RL
        return 0.0

    def compute_normalized_dense_reward(self, obs, action, info):
        # Return 0 to bypass the NotImplementedError
        return 0.0

# 2. Main Execution Block
if __name__ == "__main__":
    # Now you can load this safe environment
    env = gym.make(
        "DualArmDrawerPlace-v1", 
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
        # Render the frame
        env.render()  # <--- Updates the GUI
    
    env.close()
