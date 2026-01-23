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

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/TwoRobotStackCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["dual_panda"]
    agent: DualPanda

    goal_radius = 0.06

    def __init__(self, *args, robot_uids="dual_panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.75, 0.0, 0.75 + 0.83], target=[-0.2, 0, 0.3 + 0.83]) # 0.83: height of the table
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

        self.cube_half_size = [0.06,0.04,0.02]

        self.box = actors.build_box(
            self.scene,
            half_sizes=self.cube_half_size,
            color=[0, 1, 0, 1],
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
            box_xyz[:, 0] = - torch.rand((b,)) * 0.1
            box_xyz[:, 1] = torch.rand((b,)) * 0.8 - 0.4
            box_xyz[:, 2] = 0.02 + 0.83
            
            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, 0] = -0.4
            goal_xyz[:, 1] = 0
            goal_xyz[:, 2] = 1e-3 + 0.83
            
            qs = random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=True,
            )

            self.box.set_pose(Pose.create_from_pq(p=box_xyz, q=[1,0,0,0]))

            qs = random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=True,
            )

            self.goal_region.set_pose(Pose.create_from_pq(p=goal_xyz, q=[1,0,0,0]))

    def _initialize_agent(self):
        qpos = np.zeros(self.agent.robot.dof)
        self.agent.reset(qpos)

    def calculate_rectangle_overlap_percentage(self, rect1, rect2):
        """
        Calculate the percentage of overlap between two rectangles.
        
        Based on: https://math.stackexchange.com/questions/2449221/
        
        Args:
            rect1: dict with keys 'left', 'right', 'top', 'bottom' (or use tuple (l, r, t, b))
            rect2: dict with keys 'left', 'right', 'top', 'bottom' (or use tuple (l, r, t, b))
        
        Returns:
            float: Percentage of overlap relative to rect1's area (0-100). Returns 0 if no overlap.
        """
        # Handle both dict and tuple inputs
        if isinstance(rect1, dict):
            l0, r0, t0, b0 = rect1['left'], rect1['right'], rect1['top'], rect1['bottom']
        else:
            l0, r0, t0, b0 = rect1
        
        if isinstance(rect2, dict):
            l1, r1, t1, b1 = rect2['left'], rect2['right'], rect2['top'], rect2['bottom']
        else:
            l1, r1, t1, b1 = rect2
        
        # Calculate overlap area using the formula from Math.SE
        # A_overlap = (max(l0, l1) - min(r0, r1)) * (max(t0, t1) - min(b0, b1))
        overlap_width = min(r0, r1) - max(l0, l1)
        overlap_height = min(t0, t1) - max(b0, b1)
        
        # If either dimension is negative or zero, there's no overlap
        if overlap_width <= 0 or overlap_height <= 0:
            return 0.0
        
        overlap_area = overlap_width * overlap_height
        
        # Calculate area of rect1
        rect1_area = (r0 - l0) * (t0 - b0)
        
        if rect1_area <= 0:
            return 0.0
        
        # Return percentage relative to rect1
        percentage = (overlap_area / rect1_area) * 100
        return percentage

    def check_overlap_and_stop(self, rect1, rect2, threshold=50.0):
        """
        Check if two rectangles overlap by more than a threshold percentage.
        If overlap exceeds threshold, return True (indicating we should stop).
        
        Args:
            rect1: First rectangle
            rect2: Second rectangle
            threshold: Overlap percentage threshold (default 50%)
        
        Returns:
            bool: True if overlap > threshold (should stop), False otherwise
        """
        overlap_pct = self.calculate_rectangle_overlap_percentage(rect1, rect2)
        
        if overlap_pct > threshold:
            # print(f"⚠ Overlap detected: {overlap_pct:.2f}% > {threshold}% threshold - STOPPING")
            return True
        
        return False

    
    def evaluate(self):
        box_obb = get_actor_obb(self.box)
        # Extract 2D rectangle bounds (x-y plane projection)
        # OBB is a trimesh.primitives.Box object with primitive.extents and primitive.transform
        box_transform = np.array(box_obb.primitive.transform)
        box_extents = np.array(box_obb.primitive.extents)
        box_center = box_transform[:3, 3]  # Get center from transformation matrix
        # print(box_extents)
        box_rect = (
            box_center[0] - box_extents[2]/2,  # left
            box_center[0] + box_extents[2]/2,  # right
            box_center[1] + box_extents[1]/2,  # top
            box_center[1] - box_extents[1]/2,  # bottom
        )
        
        # For goal_region, use its pose and known half_sizes from the environment
        # cube_half_size = [0.06, 0.04, 0.02], goal uses first two: [0.06, 0.04]
        goal_pos = self.goal_region.pose.sp.p
        if hasattr(goal_pos, 'cpu'):
            goal_center = goal_pos.cpu().numpy()
        else:
            goal_center = np.array(goal_pos)
        
        goal_half_sizes = np.array([0.06, 0.04])  # half_sizes from cube_half_size[:2]
        goal_rect = (
            goal_center[0] - goal_half_sizes[0],  # left
            goal_center[0] + goal_half_sizes[0],  # right
            goal_center[1] + goal_half_sizes[1],  # top
            goal_center[1] - goal_half_sizes[1],  # bottom
        )
        
        # print(f"Box rectangle (left, right, top, bottom): {box_rect}")
        # print(f"Goal rectangle (left, right, top, bottom): {goal_rect}")
        
        # Calculate and display overlap
        box_goal_overlap = self.calculate_rectangle_overlap_percentage(box_rect, goal_rect)
        # print(f"Box-Goal overlap: {box_goal_overlap:.2f}%")
        
        # Check if overlap exceeds 50% threshold and stop if needed
        if self.check_overlap_and_stop(box_rect, goal_rect, threshold=50.0):
            # print(True)
            return {"success": torch.tensor(True)}
        # print(False)
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
            obs["tcp_pose_left"] = pose_to_vec(self.agent.tcp_1_pose)
            obs["tcp_pose_right"] = pose_to_vec(self.agent.tcp_2_pose)
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