from typing import Any, Tuple

import numpy as np
import sapien
import torch
import gymnasium as gym

from mani_skill.agents.robots.panda.dual_panda import DualPanda 
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.envs.tasks.tabletop.colosseum_v2.colosseum_v2_core import ColosseumV2Env

@register_env("DualArmPushBox-v1", max_episode_steps=100)
class DualPandaPushBoxEnv(ColosseumV2Env):
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
    IGNORED_VARIATION_FACTORS = [
        "table_color",
        "table_texture",
    ]

    def __init__(self, *args, robot_uids="dual_panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, ignored_variation_factors=self.IGNORED_VARIATION_FACTORS, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.6, 0.0, 0.8 + 0.83], target=[-0.2, 0, 0.1 + 0.83]) # 0.83: height of the table
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

        self.cube_half_size = [0.08, 0.12, 0.04]
        box_builder = lambda: self.get_box_asset_builder(
            half_size=self.cube_half_size,
            color=np.array([0, 0.5, 0, 1]),
            object_type="MO",
            initial_pose=sapien.Pose(p=[1, 0, 0.02], q=[1,0,0,0]),
        )
        goal_region_builder = lambda: actors.build_box_target(
            self.scene,
            half_sizes=self.cube_half_size[:2], # x and y
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(),
            return_builder=True,
        )
        self.goal_region = self.add_asset_to_scene(goal_region_builder, name="goal_region", physics_type="kinematic", object_type="RO")
        self.box = self.add_asset_to_scene(box_builder, name="box", physics_type="dynamic", object_type="MO")
        self.load_scene_hook(manipulation_objects=[self.box], receiving_objects=[self.goal_region])

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            box_xyz = torch.zeros((b, 3), device=self.device)
            box_xyz[:, 0] = - torch.rand((b,), device=self.device) * 0.2
            box_xyz[:, 1] = torch.rand((b,), device=self.device) * 0.2 - 0.1
            box_xyz[:, 2] = 0.02 + 0.83
            
            goal_xyz = torch.zeros((b, 3), device=self.device)
            goal_xyz[:, 0] = -0.4
            goal_xyz[:, 1] = 0
            goal_xyz[:, 2] = 1e-3 + 0.83
            
            theta_by_2 = (torch.rand(b, device=self.device))*np.pi/6-np.pi/12
            cos_vals = torch.cos(theta_by_2)
            sin_vals = torch.sin(theta_by_2)
            box_q = torch.zeros((b, 4), device=self.device)
            box_q[:, 0] = cos_vals
            box_q[:, 3] = sin_vals
            self.box.set_pose(Pose.create_from_pq(p=box_xyz, q=box_q))

            self.goal_region.set_pose(Pose.create_from_pq(p=goal_xyz, q=torch.tensor([1,0,0,0], device=self.device).repeat(b, 1)))
            self.initialize_episode_hook(env_idx, mo_pose=self.box.pose)
        self._initialize_agent()

    def _initialize_agent(self):
        qpos = np.array([0.84, 2.307, 0.154, 0.17, -0.27, 0.273, -2.496, -2.473, 0.086, -0.096, 2.642, 2.633, -0.288, 1.875, 0.04, 0.04, 0.04, 0.04])
        self.agent.reset(qpos)
            
    def evaluate(self):
        box_pos = self.box.pose.p
        goal_pos = self.goal_region.pose.p

        # Calculate euclidean distance between box and goal centers (x-y plane only)
        distance = torch.linalg.norm(
            box_pos[:, :2] - goal_pos[:, :2],
            dim=1
        )

        # Success if distance is below threshold (0.15 is approximately the goal region size)
        distance_threshold = 0.06
        success = distance < distance_threshold
        return {"success": success}



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