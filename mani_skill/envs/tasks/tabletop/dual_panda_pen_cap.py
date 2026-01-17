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
import torch

# 1. Define the Empty Environment
@register_env("DualArmPenCap-v0", max_episode_steps=1000)
class DualArmPenCapEnv(BaseEnv):
    """
    A minimal environment for Dual Panda motion planning.
    No cubes, no tasks, just the robot.
    """
    # cube_half_size = 0.02
    # Explicitly tell ManiSkill to use the DualPanda agent
    SUPPORTED_ROBOTS = ["dual_panda"]
    agent: DualPanda # Type hinting for IDE support
    def __init__(self, *args, robot_uids="dual_panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        
    def _load_scene(self, options: dict):
        self.cap = self.load_glb_as_actor(self.scene, 
                                        os.path.join(PACKAGE_ASSET_DIR,"pen_in_cap/cap.glb"),
                                        sapien.Pose(p=[0.055, -0.158, 0.], q=[0.854,0.471,0.212,0.068]),
                                        name="cap",
                                        scale=[0.3,0.3,0.2],
                                        type="dynamic")
        
        self.pen = self.load_glb_as_actor(self.scene,
                                          os.path.join(PACKAGE_ASSET_DIR, "pen_in_cap/pen.glb"),
                                          sapien.Pose(p=[0.055, -0.158, 0.5], q=[0.854,0.471,0.212,0.068]),
                                          name="pen",
                                          scale=[0.19,0.19,0.2],
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
        print(f"{name} imported successfully")
        return actor
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[..., :2] = -torch.rand((b, 2)) * 0.3 - 0.1
            xyz[..., 2] = 0.84
            q = [0, 0, 0.707, 0.707]
            self.cap.set_pose(Pose.create_from_pq(p=xyz, q=q))
            
            xyz[..., :2] = torch.rand((b, 2)) * 0.3 - 0.1
            xyz[..., 2] = 0.84
            q = [0, 0, 0.707, 0.707]
            self.pen.set_pose(Pose.create_from_pq(p=xyz, q=q))
            
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
        obs["pen_pose"] = self.pen.pose.raw_pose
        obs["cap_pose"] = self.cap.pose.raw_pose
        return obs

    def compute_dense_reward(self, obs, action, info):
        # Return 0 since we are not training RL
        return 0.0


    def compute_normalized_dense_reward(self, obs, action, info):
        # Return 0 to bypass the NotImplementedError
        return 0.0
    
    def _get_cap_opening_direction_torch(self, quat):
        """Extract cap opening direction from quaternion tensor [w, x, y, z].
        Returns the local -Z axis direction (pointing into the cap).
        """
        rotation_matrix = self._quat_to_rotation_matrix_torch(quat)
        # Cap opening points in -Z direction (into the cap)
        neg_z_direction = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device)
        cap_direction = rotation_matrix @ neg_z_direction
        return cap_direction
    
    def _quat_to_rotation_matrix_torch(self, quat):
        """Convert quaternion tensor [w, x, y, z] to 3x3 rotation matrix tensor."""
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        return torch.tensor([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ], dtype=torch.float32, device=self.device)

    def evaluate(self):
        """
        Evaluate if the pen tip is successfully inserted into the cap.
        """
        pen_pose = self.pen.pose
        cap_pose = self.cap.pose
        
        # Get pen tip position in world frame
        pen_tip_pose = pen_pose * sapien.Pose(p=[0, 0, -0.16])
        pen_tip_pos = torch.as_tensor(pen_tip_pose.p, dtype=torch.float32, device=self.device)
        
        # Get cap center (opening) in world frame
        cap_opening_pose = cap_pose * sapien.Pose(p=[0.02, 0, 0.12])
        cap_center = torch.as_tensor(cap_opening_pose.p, dtype=torch.float32, device=self.device)
        
        # Convert quaternions to tensors
        cap_quat = torch.as_tensor(cap_pose.q, dtype=torch.float32, device=self.device)
        
        # Get cap opening direction (local -Z axis pointing into the cap)
        cap_direction = self._get_cap_opening_direction_torch(cap_quat[0])
        
        # Check if pen tip is along the cap opening axis
        vec_to_cap = pen_tip_pos - cap_center
        distance_to_axis = torch.norm(vec_to_cap - (vec_to_cap @ cap_direction) * cap_direction)
        
        # Check depth (how far along the cap direction the pen tip is)
        depth_into_cap = (vec_to_cap @ cap_direction).item()
        
        # Success criteria
        axis_tolerance = 0.02  # 1cm tolerance from cap center axis
        min_depth = 0.03  # Pen should be at least 5cm inside
        
        is_aligned = distance_to_axis < axis_tolerance
        is_deep_enough = depth_into_cap > min_depth
        success = is_aligned * is_deep_enough
        
        # print({
        #     "distance_to_axis": distance_to_axis.item(), 
        #     "depth_into_cap": depth_into_cap, 
        #     "is_aligned": is_aligned.item(), 
        #     "is_deep_enough": is_deep_enough,
        #     "success": success
        # })
        
        return {
            "is_aligned": is_aligned,
            "is_deep_enough": is_deep_enough,
            "success": success
        }
        
# 2. Main Execution Block
if __name__ == "__main__":
    # Now you can load this safe environment
    env = gym.make(
        "DualArmPenCap-v0", 
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