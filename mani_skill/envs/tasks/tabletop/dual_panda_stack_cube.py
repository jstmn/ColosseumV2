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


@register_env("DualArmStackCube-v1", max_episode_steps=100)
class TwoRobotStackCube(BaseEnv):
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
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**19,
                max_rigid_contact_count=2**21,
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[-1.3, -0.4, 0.7+0.83], target=[0.1, 0, 0.1+0.83])
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

    # def _load_agent(self, options: dict):
    #     super()._load_agent(
    #         options, [sapien.Pose(p=[0, -1, 0]), sapien.Pose(p=[0, 1, 0])]
    #     )

    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([0.02] * 3, device=self.device)
        # self.table_scene = TableSceneBuilder(
        #     env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        # )
        # self.table_scene.build()
        self.cubeA = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=np.array([12, 42, 160, 255]) / 255,
            name="cubeA",
            initial_pose=sapien.Pose(p=[1, 0, 0.02]),
        )
        self.cubeB = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[0, 1, 0, 1],
            name="cubeB",
            initial_pose=sapien.Pose(p=[-1, 0, 0.02]),
        )
        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            # self.scene.initialize(env_idx)
            # the table scene initializes two robots. the first one self.agents[0] is on the left and the second one is on the right
            cubeA_xyz = torch.zeros((b, 3))
            cubeA_xyz[:, 0] = torch.rand((b,)) * 0.2 - 0.05
            cubeA_xyz[:, 1] = -0.15 - torch.rand((b,)) * 0.1 + 0.05
            cubeB_xyz = torch.zeros((b, 3))
            cubeB_xyz[:, 0] = torch.rand((b,)) * 0.1 - 0.05
            cubeB_xyz[:, 1] = 0.15 + torch.rand((b,)) * 0.1 - 0.05
            cubeA_xyz[:, 2] = 0.02 + 0.83 
            cubeB_xyz[:, 2] = 0.02 + 0.83

            qs = random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeA.set_pose(Pose.create_from_pq(p=cubeA_xyz, q=qs))

            qs = random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeB.set_pose(Pose.create_from_pq(p=cubeB_xyz, q=qs))

            target_region_xyz = torch.zeros((b, 3))
            target_region_xyz[:, 0] = torch.rand((b,)) * 0.1 - 0.05
            target_region_xyz[:, 1] = 0
            # set a little bit above 0 so the target is sitting on the table
            target_region_xyz[..., 2] = 1e-3 + 0.83
            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p=target_region_xyz,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )

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
        obs["cubeA_pose"] = self.cubeA.pose.raw_pose
        obs["cubeB_pose"] = self.cubeB.pose.raw_pose
        obs["goal_region_pos"] = self.goal_region.pose.p
        return obs


    def evaluate(self):
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        offset = pos_B - pos_A

        xy_flag = (
            torch.linalg.norm(offset[..., :2], axis=1)
            <= 0.02+0.005
        )
        
        z_flag = torch.abs(offset[..., 2] - 0.02 * 2) <= 0.005
        
        # is_cubeB_on_cubeA = torch.logical_and(xy_flag, z_flag)
        is_cubeB_on_cubeA = z_flag
        cubeB_to_goal_dist = torch.linalg.norm(
            self.cubeB.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
        )
        
        cubeB_placed = cubeB_to_goal_dist < self.goal_radius

        success = (
            is_cubeB_on_cubeA * cubeB_placed
        )
        # print(success)
        return {
            "is_cubeB_on_cubeA": is_cubeB_on_cubeA,
            "cubeB_placed": cubeB_placed,
            "success": success,
        }


    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return 0.0

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return 0.0

if __name__ == "__main__":
    # Now you can load this safe environment
    env = gym.make(
        "DualArmStackCube-v1", 
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
    