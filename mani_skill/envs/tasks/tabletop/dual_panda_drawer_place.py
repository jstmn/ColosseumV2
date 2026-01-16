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

@register_env("DualArmDrawerPlace-v0", max_episode_steps=1000, asset_download_ids=["partnet_mobility_cabinet"])
class DualArmDrawerPlaceEnv(BaseEnv):
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
        pose = sapien_utils.look_at(eye=[-0.3, 0, 0.6], target=[-0.1, 0, -0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([-0.6, -0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        # Load a simple floor and lighting
        # self.add_ground(altitude=0)
        # self._setup_lighting()
        # self.ground = build_ground(self.scene, floor_width=floor_width, altitude=-self.table_height, name=f"ground{name_suffix}")

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
        self.open_cabinet = builder.build(name=f"drawer-{model_id}")
        
        # Ensure the cabinet is static (fixed root)
        # PartNet assets loaded via builder usually default to dynamic, 
        # so we lock the root link to prevent it from falling if it's not on the ground.
        # if self.open_cabinet.root.entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent):
        #      self.open_cabinet.root.entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).kinematic = True
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            # Reset cabinet pose
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
            
            xyz[..., :2] = torch.rand((b,2)) * 0.1 - 0.05 - 0.25
            xyz[..., 2] = 0.85
            
            self.obj.set_pose(sapien.Pose(p=xyz[0].cpu().numpy()))
            # Close the drawer (reset joint positions to 0)
            self.open_cabinet.set_qpos(np.zeros(self.open_cabinet.dof))
            self.open_cabinet.set_qvel(np.zeros(self.open_cabinet.dof))
        
    def _initialize_agent(self):
        # Reset the robot to a neutral position
        qpos = np.zeros(self.agent.robot.dof)
        self.agent.reset(qpos)

    def evaluate(self):
        box_pos = self.obj.pose.p
        drawer_pos = self.open_cabinet.pose.p
        
        above_ground = box_pos[..., 2] > 0.9
        inside = torch.norm(box_pos[..., :2] - drawer_pos[..., :2]) < 0.25
        success = above_ground * inside
        print(success)
        return {"above_ground": above_ground, "inside": inside, "success": success}
        
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
        "DualArmDrawerPlace-v0", 
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
        
    
    env.close()
    
#     mani_skill/utils/scene_builder/table/scene_builder.py
# qpos = np.array(
#                 [0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, -np.pi / 4, 0.04, 0.04]
#             )

# Jeremy Morgan
# 11:11 PM
# def initialize(self, env_idx: torch.Tensor, table_z_rotation_angle: float = np.pi/2.0, qpos_0: Optional[np.ndarray] = None):
# use qpos_0
