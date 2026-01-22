import importlib
import os
# Directly import the specific submodule 
pm = importlib.import_module("mani_skill.utils.building.articulations.partnet_mobility")

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
from mani_skill.utils.building import articulations
from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

import torch

# 1. Define the Empty Environment
@register_env("DualArmPickBottle-v1", max_episode_steps=1000)
class DualArmPickBottleEnv(BaseEnv):
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
        self.obj = self.load_glb_as_actor(self.scene, 
                                        os.path.join(PACKAGE_ASSET_DIR,"pick_bottle/plastic_bottle.glb"),
                                        sapien.Pose(p=[0.055, -0.158, 0.], q=[0.854,0.471,0.212,0.068]),
                                        name="bottle",
                                        scale=[0.06,0.06,0.08],
                                        type="dynamic")
    
    
        
    @staticmethod
    def load_glb_as_actor(scene, glb_file_path, pose, name, scale, type="static"):
        """Load GLB file as a static actor in the scene"""
        builder = scene.create_actor_builder()
        builder.add_visual_from_file(glb_file_path, scale=scale)
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
            xyz[..., :2] = torch.rand((b, 2)) * 0.3 - 0.1
            xyz[..., 2] = self.cube_half_size+0.83
            q = [0.707, 0.707, 0, 0]
            self.obj.set_pose(Pose.create_from_pq(p=xyz, q=q))
        self._initialize_agent()
        
    def _initialize_agent(self):
        # Reset the robot to a neutral position
        # Dual Panda has 14+ gripper joints. 
        # You can define a custom "qpos" (joint positions) here if you want.
        # 0-6: Left Arm, 7-8: Left Gripper, 9-15: Right Arm, 16-17: Right Gripper
        # qpos = np.array([-0.186, 1.571, 2.051, 0, 1.478, 0, -1.74, -2.749, -0.965, 0, 3.149, 2.356, -0.309, 0.785, 0.04, 0.04, 0.04, 0.04])
        # Example: Set arms to a ready position (optional)
        # qpos[0] = 0.5  # Move left shoulder
        # qpos[9] = -0.5 # Move right shoulder
        qpos = np.array([0.066, 1.571, 0.573, 0, 0.158, 0, -2.084, -2.749, 1.701, 0, 1.763, 2.356, -1.882, 0.785, 0.04, 0.04, 0.04, 0.04])
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
        obs["obj_pose"] = self.obj.pose.raw_pose
        return obs

    def evaluate(self):
        pos_1 = self.agent.tcp_1_pose.p
        pos_2 = self.agent.tcp_2_pose.p
        obj_pos = self.obj.pose.p
        offset_1 = pos_1 - obj_pos
        offset_2 = pos_2 - obj_pos
        dist_1 = torch.linalg.norm(offset_1, dim=-1)
        dist_2 = torch.linalg.norm(offset_2, dim=-1)
        grasped_2 = self.agent.is_grasping(self.obj, arm_index=2)        
        success = torch.logical_and(dist_2 <= dist_1, grasped_2)
        return {"grasping_bottle": grasped_2, "success": success}
    
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
        "DualArmPickBottle-v1", 
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