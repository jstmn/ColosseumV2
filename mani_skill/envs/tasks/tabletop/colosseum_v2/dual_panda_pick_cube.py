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
from mani_skill.envs.distraction_set import DistractionSet
import torch

# 1. Define the Empty Environment
@register_env("DualArmPickCube-v1", max_episode_steps=1000)
class DualArmPickCubeEnv(BaseEnv):
    """
    A minimal environment for Dual Panda motion planning.
    No cubes, no tasks, just the robot.
    """
    cube_half_size = 0.02
    # Explicitly tell ManiSkill to use the DualPanda agent
    SUPPORTED_ROBOTS = ["dual_panda"]
    agent: DualPanda # Type hinting for IDE support
    
    def __init__(self, *args, robot_uids="dual_panda", **kwargs):
        distraction_set: DistractionSet | dict | None = kwargs.pop("distraction_set", None)
        self._distraction_set: DistractionSet | None = DistractionSet(**distraction_set) if isinstance(distraction_set, dict) else distraction_set
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
    
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.75, 0.0, 0.75 + 0.83], target=[-0.2, 0, 0.3 + 0.83]) # 0.83: height of the table
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=500,
                height=500,
                fov=np.pi / 3,
                near=0.01,
                far=10,
            )
        ]
    
    @property
    def _default_human_render_camera_configs(self):
        """Configure camera for rendering videos and visualization"""
        pose = sapien_utils.look_at(eye=[0.6, 0.2, 0.4+0.83], target=[-0.1, 0, 0.1+0.83])
        return CameraConfig("render_camera", pose, 500, 500, 1, 0.01, 100, shader_pack="rt")

    
    def _load_scene(self, options: dict):
        # Load a simple floor and lighting
        self.obj = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array([12, 42, 160, 255]) / 255,
            name="cube",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[-0.2, -0.141, 0.83+self.cube_half_size]),
        )
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[..., 2] = self.cube_half_size+0.83
            q = [1, 0, 0, 0]
            self.obj.set_pose(Pose.create_from_pq(p=xyz, q=q))
        
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
            obs["left_arm_tcp"] = pose_to_vec(self.agent.tcp_1_pose)
            obs["right_arm_tcp"] = pose_to_vec(self.agent.tcp_2_pose)
        if "state" in self.obs_mode:
            obs["obj_pose"] = self.obj.pose.raw_pose
        return obs

    # Is the object 100 times closer to tcp1 than tcp2?
    def evaluate(self):
        pos_1 = self.agent.tcp_1_pose.p
        pos_2 = self.agent.tcp_2_pose.p
        obj_pos = self.obj.pose.p
        offset_1 = pos_1 - obj_pos
        offset_2 = pos_2 - obj_pos
        dist_1 = torch.linalg.norm(offset_1, dim=-1)
        dist_2 = torch.linalg.norm(offset_2, dim=-1)
        
        success = dist_2 <= dist_1
        return {"success": success}
    
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
        "DualArmPickCube-v1", 
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