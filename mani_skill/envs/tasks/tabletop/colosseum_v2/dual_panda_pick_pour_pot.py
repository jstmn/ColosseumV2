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
from mani_skill.envs.tasks.tabletop.colosseum_v2.distraction_set import DistractionSet

import torch
import os
from mani_skill import PACKAGE_ASSET_DIR

# 1. Define the Empty Environment
@register_env("DualArmPourPot-v1", max_episode_steps=1000)
class DualArmPourPotEnv(BaseEnv):
    """
    A minimal environment for Dual Panda motion planning.
    No cubes, no tasks, just the robot.
    """
    cube_half_size = 0.02
    tray_half_width = 0.2
    tray_half_length = 0.12
    # Explicitly tell ManiSkill to use the DualPanda agent
    SUPPORTED_ROBOTS = ["dual_panda"]
    agent: DualPanda # Type hinting for IDE support
    
    def __init__(self, *args, robot_uids="dual_panda", **kwargs):
        distraction_set: DistractionSet | dict | None = kwargs.pop("distraction_set", None)
        self._distraction_set: DistractionSet | None = DistractionSet(**distraction_set) if isinstance(distraction_set, dict) else distraction_set
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[-1.0, 0.0, 0.75 + 0.83], target=[0.0, 0, 0.2 + 0.83]) # 0.83: height of the table
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
        pose = sapien_utils.look_at(eye=[0.6, 0.2, 0.4+0.83], target=[-0.1, 0, 0.1+0.83])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    
    def _load_scene(self, options: dict):
        self.ball = self.load_glb_as_actor(self.scene,
                                           os.path.join(PACKAGE_ASSET_DIR, "pour_pot/tomato.glb"),
                                           sapien.Pose(p=[-0.2, -0.141, 0.83+self.cube_half_size]),
                                           name="tomato",
                                           scale=[1,1,1],
                                           type="dynamic")
        self.pot = self.load_glb_as_actor(self.scene, 
                                        os.path.join(PACKAGE_ASSET_DIR,"pour_pot/pot.glb"),
                                        sapien.Pose(p=[0.055, -0.158, 0.], q=[0.854,0.471,0.212,0.068]),
                                        name="pot",
                                        scale=[1,1,1],
                                        type="dynamic", color=np.array((129/255, 133/255, 137/255, 1)))
        self.tray = self.load_glb_as_actor(self.scene,
                                           os.path.join(PACKAGE_ASSET_DIR, "pour_pot/plastic_tray.glb"),
                                           sapien.Pose(),
                                           name="tray",
                                           scale=[0.4,0.4,0.4],
                                           type="dynamic", color=np.array([48/255, 49/255, 51/255, 1]))
    @staticmethod
    def load_glb_as_actor(scene, glb_file_path, pose, name, scale, type="static",color=None):
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
            pot_xyz = torch.zeros((b, 3), device=self.device)
            pot_xyz[..., 0] = 0.1
            pot_xyz[..., 1] = torch.rand(b, device=self.device) * 0.2 - 0.1
            pot_xyz[..., 2] = self.cube_half_size+0.83
            
            pot_q = torch.tensor([0.5, 0.5, 0.5, 0.5], device=self.device).repeat(b, 1)
            self.pot.set_pose(Pose.create_from_pq(p=pot_xyz, q=pot_q))
            
            ball_xyz = pot_xyz.clone()
            ball_xyz[..., 2] = 0.9
            self.ball.set_pose(Pose.create_from_pq(p=ball_xyz,q=[1,0,0,0]))

            tray_xyz = pot_xyz.clone()
            tray_xyz[..., 0] = -0.2
            tray_xyz[..., 2] = 0.83 + self.cube_half_size
            self.tray.set_pose(Pose.create_from_pq(p=tray_xyz, q=[0.5, 0.5, 0.5, 0.5]))
        self._initialize_agent()
        
    def _initialize_agent(self):
        # Reset the robot to a neutral position
        # Dual Panda has 14+ gripper joints. 
        # You can define a custom "qpos" (joint positions) here if you want.
        # 0-6: Left Arm, 7-8: Left Gripper, 9-15: Right Arm, 16-17: Right Gripper
        qpos = np.array([0.599, 2.358, 0.4, 0.442, -0.561, 0.697, -2.511, -2.513, 1.695, -1.775, 1.395, 1.347, -0.479, 2.031, 0.04, 0.04, 0.04, 0.04])
        # Example: Set arms to a ready position (optional)
        # qpos[0] = 0.5  # Move left shoulder
        # qpos[9] = -0.5 # Move right shoulder
        
        self.agent.reset(qpos)

    def _get_obs_extra(self, info: dict):
        obs = dict()
        # Helper to convert sapien.Pose to numpy array (Pos + Quat)
        def pose_to_vec(pose):
            # p and q are already tensors on the correct device (GPU)
            # We just need to concatenate them using torch instead of numpy
            return torch.cat([pose.p, pose.q], dim=-1)
        
        if hasattr(self.agent, "tcp_pose"):
             obs["tcp_pose"] = self.agent.tcp_pose.raw_pose
        else:
            # Fallback for the error you saw
            # We construct the 14D array manually if needed, or just return separate ones
            obs["left_arm_tcp"] = pose_to_vec(self.agent.tcp_1_pose)
            obs["right_arm_tcp"] = pose_to_vec(self.agent.tcp_2_pose)
        if "state" in self.obs_mode:
            obs["pot_pose"] = self.pot.pose.raw_pose
            obs["ball_pose"] = self.ball.pose.raw_pose
        return obs

    def evaluate(self):
        ball_pos = self.ball.pose.p
        tray_pos = self.tray.pose.p
        offset = tray_pos - ball_pos
        in_tray = (torch.abs(offset[..., 1]) < DualArmPourPotEnv.tray_half_width) * (torch.abs(offset[..., 0]) < DualArmPourPotEnv.tray_half_length)
        success = (ball_pos[..., 2] >= 0.83) * (ball_pos[..., 2] < 0.9) * in_tray
        return {"inside_tray": in_tray,"success": success}
    
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
        "DualArmPourPot-v1", 
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