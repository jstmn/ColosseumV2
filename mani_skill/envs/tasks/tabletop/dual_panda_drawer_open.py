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

# 1. Define the Empty Environment
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Ensure GPU 0 is used for both sim and render

@register_env("DualArmDrawerOpen-v1", max_episode_steps=1000, asset_download_ids=["partnet_mobility_cabinet"])
class DualArmDrawerOpenEnv(BaseEnv):
    """
    Two hold the handles of drawer and open the doors.
    Uses PartNet-Mobility dataset (ID 1005).
    """
    cube_half_size = 0.02
    # Explicitly tell ManiSkill to use the DualPanda agent
    SUPPORTED_ROBOTS = ["dual_panda"]
    agent: DualPanda # Type hinting for IDE support
    
    def __init__(self, *args, robot_uids="dual_panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        if self.scene is not None:
            print(f"Is GPU simulation enabled for this scene? {self.scene.gpu_sim_enabled}")
    
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[-0.3, 0.5, 1.0+0.83], target=[-0.1, 0, 0.2+0.83])
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
        self.open_cabinet = builder.build(name=f"drawer-{model_id}")
        
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[..., :2] = -torch.rand((b, 2)) * 0.2 + 0.1
            xyz[..., 0] += 0.2
            xyz[..., 2] = 0.456+0.8
            theta_by_2 = torch.rand(b)*np.pi/16 - np.pi/32
            for i in range(b):
                # Reset cabinet pose
                init_pose = Pose.create_from_pq(
                    p=xyz[i:i+1], 
                    q=[np.cos(theta_by_2), 0, 0, np.sin(theta_by_2)])
                p_np = init_pose.p.squeeze(0).cpu().numpy().astype(np.float32)
                q_np = init_pose.q.squeeze(0).cpu().numpy().astype(np.float32)
                init_pose_sapien = sapien.Pose(p=p_np, q=q_np)

                self.open_cabinet.set_pose(init_pose_sapien)
            # Close the drawer (reset joint positions to 0)
            self.open_cabinet.set_qpos(np.zeros(self.open_cabinet.dof))
            self.open_cabinet.set_qvel(np.zeros(self.open_cabinet.dof))
        self._initialize_agent()
        
    def _initialize_agent(self):
        # Reset the robot to a neutral position
        qpos = np.array([1.683, 1.357, 0.284, 0.393, -0.103, 0.249, -1.529, -2.074, -1.497, 1.647, 1.409, 1.758, -2.106, -0.114, 0.04, 0.04, 0.04, 0.04])
        self.agent.reset(qpos)

    def evaluate(self):
        pose_1 = self.open_cabinet.links_map['link_0'].pose.p
        pose_2 = self.open_cabinet.links_map['link_2'].pose.p

        drawer_1_open = pose_1[..., 0] < -0.07
        drawer_2_open = pose_2[..., 0] < -0.07
        
        success = drawer_1_open * drawer_2_open
        # print(success)
        return {"drawer_1_open": drawer_1_open, "drawer_2_open": drawer_2_open, "success": success}
    
    def _get_obs_extra(self, info: dict):
        obs = dict()
        # Helper to convert sapien.Pose to numpy array (Pos + Quat)
        def pose_to_vec(pose):
            # pose.p is [x,y,z], pose.q is [w,x,y,z]
            return np.hstack([pose.p, pose.q])
        
        if hasattr(self.agent, "tcp_pose"):
             obs["tcp_pose"] = self.agent.tcp_pose.raw_pose
        else:
            obs["tcp_pose_left"] = pose_to_vec(self.agent.tcp_1_pose)
            obs["tcp_pose_right"] = pose_to_vec(self.agent.tcp_2_pose)

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
        "DualArmDrawerOpen-v1", 
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