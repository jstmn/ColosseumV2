from typing import Any, Tuple

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat
import gymnasium as gym
from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.panda.dual_panda import DualPanda 
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb
from mani_skill.envs.distraction_set import DistractionSet

@register_env("DualArmPushBox-v1", max_episode_steps=100)
class DualPandaPushBoxEnv(BaseEnv):
    """
    **Task Description:**
    A collaborative task where two robot arms need to work together to stack two cubes. One robot must pick up the green cube and place it on the target region, while the other robot picks up the blue cube and stacks it on top of the green cube.

    The cubes are initially positioned such that each robot can only reach one cube - the green cube is near the right robot and the blue cube is near the left robot. This requires coordination between the robots to complete the stacking task.

    **Randomizations:**
    - Both cubes have random rotations around their z-axis
    - The xy positions of both cubes on the table are randomized, while ensuring:
        - The cubes do not collide with each other
        - The green cube remains reachable by the right robot
        - The blue cube remains reachable by the left robot
    - The goal region is placed along the midline between the robots (y=0), with randomized x position

    **Success Conditions:**
    - The blue cube is stacked on top of the green cube (within half a cube size)
    - The green cube is placed on the red/white target region
    - Both cubes are released by the robots (not being grasped)

    """

    SUPPORTED_ROBOTS = ["dual_panda"]
    agent: DualPanda

    def __init__(self, *args, robot_uids="dual_panda", **kwargs):
        distraction_set: DistractionSet | dict | None = kwargs.pop("distraction_set", None)
        self._distraction_set: DistractionSet | None = DistractionSet(**distraction_set) if isinstance(distraction_set, dict) else distraction_set
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.6, 0.0, 0.8 + 0.83], target=[-0.2, 0, 0.1 + 0.83]) # 0.83: height of the table
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

        self.cube_half_size = [0.08,0.12,0.04]

        self.box = actors.build_box(
            self.scene,
            half_sizes=self.cube_half_size,
            color=[0, 0.5, 0, 1],
            name="Box",
            initial_pose=sapien.Pose(p=[1, 0, 0.02], q=[1,0,0,0]),
        )
        
        self.goal_region = actors.build_box_target(
            self.scene,
            half_sizes=self.cube_half_size[:2],
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            box_xyz = torch.zeros((b, 3))
            box_xyz[:, 0] = - torch.rand((b,)) * 0.2
            box_xyz[:, 1] = torch.rand((b,)) * 0.2 - 0.1
            box_xyz[:, 2] = 0.02 + 0.83
            
            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, 0] = -0.4
            goal_xyz[:, 1] = 0
            goal_xyz[:, 2] = 1e-3 + 0.83
            
            theta_by_2 = (torch.rand(b))*np.pi/6-np.pi/12
            # theta_by_2 = 0
            self.box.set_pose(Pose.create_from_pq(p=box_xyz, q=torch.tensor([float(np.cos(theta_by_2)),0,0,float(np.sin(theta_by_2))])))

            qs = random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=True,
            )

            self.goal_region.set_pose(Pose.create_from_pq(p=goal_xyz, q=[1,0,0,0]))
        self._initialize_agent()

    def _initialize_agent(self):
        qpos = np.array([0.84, 2.307, 0.154, 0.17, -0.27, 0.273, -2.496, -2.473, 0.086, -0.096, 2.642, 2.633, -0.288, 1.875, 0.04, 0.04, 0.04, 0.04])
        self.agent.reset(qpos)
            
    def evaluate(self):
        # Get box center
        box_pos = self.box.pose.sp.p
        if hasattr(box_pos, 'cpu'):
            box_center = box_pos.cpu().numpy()
        else:
            box_center = np.array(box_pos)
        
        # Get goal region center
        goal_pos = self.goal_region.pose.sp.p
        if hasattr(goal_pos, 'cpu'):
            goal_center = goal_pos.cpu().numpy()
        else:
            goal_center = np.array(goal_pos)
        
        # Calculate euclidean distance between box and goal centers (x-y plane only)
        distance = np.sqrt((box_center[0] - goal_center[0])**2 + (box_center[1] - goal_center[1])**2)
        
        # Success if distance is below threshold (0.15 is approximately the goal region size)
        distance_threshold = 0.06
        if distance < distance_threshold:
            return {"success": torch.tensor(True)}
        
        return {"success": torch.tensor(False)}
        
    def _get_obs_extra(self, info: dict):
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
            obs["box_pose"] = self.box.pose.raw_pose
            obs["goal_region_pose"] = self.goal_region.pose.raw_pose
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return 0.0

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return 0.0

if __name__ == "__main__":
    # Now you can load this safe environment
    env = gym.make(
        "DualArmPushBox-v1",
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