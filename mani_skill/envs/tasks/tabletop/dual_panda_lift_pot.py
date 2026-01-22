import gymnasium as gym
import numpy as np
import sapien.core as sapien
import mani_skill.agents.robots.panda.dual_panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.agents.robots.panda.dual_panda import DualPanda 
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.building import actors
from mani_skill.utils.structs import Pose
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

import torch
import os
from mani_skill import PACKAGE_ASSET_DIR

# 1. Define the Empty Environment
@register_env("DualArmLiftPot-v1", max_episode_steps=1000)
class DualArmLiftPotEnv(BaseEnv):
    """
    A minimal environment for Dual Panda motion planning.
    No cubes, no tasks, just the robot.
    """
    cube_half_size = 0.02
    # Explicitly tell ManiSkill to use the DualPanda agent
    SUPPORTED_ROBOTS = ["dual_panda"]
    agent: DualPanda # Type hinting for IDE support
    def __init__(self, *args, robot_uids="dual_panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
    
    @property
    def _default_human_render_camera_configs(self):
        """Configure camera for rendering videos and visualization"""
        pose = sapien_utils.look_at(eye=[0.6, 0.2, 0.4+0.83], target=[-0.1, 0, 0.1+0.83])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)
        
    def _load_scene(self, options: dict):
        # Load a simple floor and lighting
        # self.add_ground(altitude=0)
        # self._setup_lighting()
        # self.ground = build_ground(self.scene, floor_width=floor_width, altitude=-self.table_height, name=f"ground{name_suffix}")
        
        self.pot = self.load_glb_as_actor(self.scene, 
                                        os.path.join(PACKAGE_ASSET_DIR,"pour_pot/pot.glb"),
                                        sapien.Pose(p=[0.055, -0.158, 0.], q=[0.854,0.471,0.212,0.068]),
                                        name="pot",
                                        scale=[1,1,1],
                                        type="dynamic", color=np.array((129/255, 133/255, 137/255, 1)))
        
    @staticmethod
    def load_glb_as_actor(scene, glb_file_path, pose, name, scale, type="static", color=None):
        """Load GLB file as a static actor in the scene"""
        builder = scene.create_actor_builder()
        if color is not None:
            custom_material = sapien.render.RenderMaterial()
            custom_material.base_color = color  # Green [R, G, B, A]
            custom_material.roughness = 0.0
            custom_material.metallic = 0.8
            builder.add_visual_from_file(glb_file_path, scale=scale, material=custom_material)
        else:
            builder.add_visual_from_file(glb_file_path, scale=scale)
        # builder.add_visual_from_file(glb_file_path, scale=scale)
        builder.add_multiple_convex_collisions_from_file(glb_file_path, decomposition="coacd", scale=scale)
        builder.set_initial_pose(pose)
        if type=="dynamic":
            actor = builder.build_dynamic(name)
        else:
            actor = builder.build_static(name)
        return actor
        
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[..., 2] = self.cube_half_size+0.83
            theta_by_2 = torch.rand(b)*np.pi/8 - np.pi/16  # -pi/2 to pi/2
            
            # Set poses for each environment in the batch
            for i in range(b):
                init_pose = Pose.create_from_pq(p=xyz[i:i+1],q=[0.5,0.5,0.5,0.5])
                # Convert tensors to numpy float32 arrays
                p_np = init_pose.p.squeeze(0).cpu().numpy().astype(np.float32)
                q_np = init_pose.q.squeeze(0).cpu().numpy().astype(np.float32)
                init_pose_sapien = sapien.Pose(p=p_np, q=q_np)
                rotation_pose = sapien.Pose(p=[0,0,0],q=[float(np.cos(theta_by_2[i])),0,0,float(np.sin(theta_by_2[i]))])
                init_pose_sapien = rotation_pose * init_pose_sapien
                self.pot.set_pose(init_pose_sapien)
            self.init_pose = init_pose_sapien
            # xyz[..., 2] = 0.9
            # self.ball.set_pose(Pose.create_from_pq(p=xyz,q=[1,0,0,0]))
        
        # Initialize agent after scene objects
        self._initialize_agent()
            
    def _initialize_agent(self):
        # Reset the robot to a neutral position
        # Dual Panda has 14+ gripper joints. 
        # You can define a custom "qpos" (joint positions) here if you want.
        # 0-6: Left Arm, 7-8: Left Gripper, 9-15: Right Arm, 16-17: Right Gripper
        # qpos = np.zeros(self.agent.robot.dof)
        qpos = np.array([0.599, 2.358, 0.4, 0.442, -0.561, 0.697, -2.511, -2.513, 1.695, -1.775, 1.395, 1.347, -0.479, 2.031, 0.04, 0.04, 0.04, 0.04])
        # Example: Set arms to a ready position (optional)
        # qpos[0] = 0.5  # Move left shoulder
        # qpos[9] = -0.5 # Move right shoulder
        
        self.agent.reset(qpos)

    def _get_obs_extra(self, info: dict):
        # THIS FIXES YOUR ERROR.
        # We manually define what "extra" info we want, handling both arms correctly.
        
        # Access the specific attributes for Dual Panda
        # (Using getattr to be safe, but usually it's tcp_pose_1 / tcp_pose_2 or similar)
        
        # Note: In many ManiSkill versions, dual agents might return a list for tcp_pose
        # But if the error says 'tcp_1_pose', we use that.
        
        obs = dict()
        # Helper to convert sapien.Pose to numpy array (Pos + Quat)
        def pose_to_vec(pose):
            # pose.p is [x,y,z], pose.q is [w,x,y,z]
            return np.hstack([pose.p, pose.q])
        
        if hasattr(self.agent, "tcp_pose"):
             obs["tcp_pose"] = self.agent.tcp_pose.raw_pose
        else:
            # Fallback for the error you saw
            # We construct the 14D array manually if needed, or just return separate ones
            obs["tcp_pose_left"] = pose_to_vec(self.agent.tcp_1_pose)
            obs["tcp_pose_right"] = pose_to_vec(self.agent.tcp_2_pose)
        obs["pot_pose"] = self.pot.pose.raw_pose
        # obs["ball_pose"] = self.ball.pose.raw_pose
        return obs

    def evaluate(self):
        curr_pose = self.pot.pose
        offset = curr_pose.p - self.init_pose.p
        is_pot_grasped_left = self.agent.is_grasping(self.pot, arm_index=1)
        is_pot_grasped_right = self.agent.is_grasping(self.pot, arm_index=2)
        is_pot_grasped = torch.logical_or(is_pot_grasped_left, is_pot_grasped_right)
        # print(is_pot_grasped_left, is_pot_grasped_right)
        offset_x = torch.abs(offset[..., 2])
        success = offset_x > 0.15
        # print(is_pot_grasped, success)
        return {"left_grasped": is_pot_grasped_left, "right_grasped": is_pot_grasped_right, "grasped": is_pot_grasped, "success": success}
    
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
        "DualArmLiftPot-v1", 
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
        # Create a dummy action (stay still)
        # action = np.zeros(env.action_space.shape)
        
        # # Step the environment
        # obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the frame
        env.render()  # <--- Updates the GUI
        
        # if terminated or truncated:
        #     obs, _ = env.reset()
    
    env.close()