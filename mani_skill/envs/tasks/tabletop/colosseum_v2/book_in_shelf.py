from typing import Any, Dict, Union
import numpy as np
import sapien
import torch
import trimesh
from mani_skill import PACKAGE_ASSET_DIR, PACKAGE_DIR
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from math import fabs
from mani_skill.utils.geometry import rotation_conversions
import os
import gymnasium as gym
from mani_skill.envs.distraction_set import DistractionSet

@register_env("PlaceBookInShelf-v1", max_episode_steps=50)
class PlaceBookEnv(BaseEnv):
    """
    **Task Description:**
    The goal is to pick up a book and place it inside a shelf with other books already in it.

    **Randomizations:**
    - books on the table have their z-axis rotation randomized.
    - books have their xy positions on top of the table scene randomized. The positions are sampled such that the books do not collide with each other.

    **Success Conditions:**
    - the book is inside the shelf. (to within half of the book size)
    - the book is static
    - the book is not being grasped by the robot (robot must let go of the cube)

    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/StackCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs
    ):
        distraction_set: DistractionSet | dict | None = kwargs.pop("distraction_set", None)
        self._distraction_set: DistractionSet | None = DistractionSet(**distraction_set) if isinstance(distraction_set, dict) else distraction_set
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        # sim_backend="physx_cuda:0", render_backend="sapien_cuda:0"
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

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0])) # Loads the panda arm

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.shelf = self.load_glb_as_actor(
            self.scene, 
            os.path.join(PACKAGE_ASSET_DIR, 'book_in_shelf/BookShelf.glb'),
            sapien.Pose(p=[0.293, -0.1, 0], q=[-0.5, -0.5, 0.5, 0.5]), 
            name="custom_glb_shelf",
            type="kinematic",
            is_shelf=True
        )

        self.book_A = self.load_glb_as_actor(self.scene, 
            os.path.join(PACKAGE_ASSET_DIR ,'book_in_shelf/simple_book_1.glb'),
            sapien.Pose(p=[0.055, -0.158, 0.1], q=[0.854,0.471,0.212,0.068]),
            name="book_A",
            type="dynamic")


    @staticmethod
    def load_glb_as_actor(scene, glb_file_path, pose, name, type="static", color=None, is_shelf=False):
        """Load GLB file as a static actor in the scene"""
        builder = scene.create_actor_builder()
        if color is not None:
            custom_material = sapien.render.RenderMaterial()
            custom_material.base_color = color  # Green [R, G, B, A]
            custom_material.roughness = 0.8
            custom_material.metallic = 0.0
            builder.add_visual_from_file(glb_file_path, material=custom_material)
        else:
            builder.add_visual_from_file(glb_file_path)

        builder.add_multiple_convex_collisions_from_file(glb_file_path, decomposition="coacd")

        builder.set_initial_pose(pose)
        if type=="dynamic":
            actor = builder.build_dynamic(name)
        elif type=="kinematic":
            actor = builder.build_kinematic(name)
        else:
            actor = builder.build_static(name)
        return actor

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.089
            region = [[0.03, -0.25],[0.09, 0]] 
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001
            bookA_xy = sampler.sample(radius, 100)

            xyz[:, :2] = bookA_xy
            self.book_A.set_pose(Pose.create_from_pq(p=xyz.clone(), q=torch.tensor([0.06, -0.162, -0.296, 0.940]).repeat(b,1)))

            xyz[..., 0] = 0.293 + torch.rand(b, device=self.device)*0.05
            xyz[..., 1] = -0.1 + torch.rand(b, device=self.device)*0.1            
            xyz[..., 2] = 0
            self.shelf.set_pose(Pose.create_from_pq(p=xyz, q=[-0.5, -0.5, 0.5, 0.5]))
        self._initialize_agent()
        
    def _initialize_agent(self):
        qpos = np.array([-0.816, 0.109, 0.437, -3.005, 2.678, 1.626, -2.312, 0.04, 0.04])
        self.agent.reset(qpos)
        
    def evaluate(self):
        pos_shelf = self.shelf.pose.p
        pos_book = self.book_A.pose.p
        offset = pos_book
        offset[..., 0] = offset[..., 0] - pos_shelf[..., 0] + 0.293
        offset[..., 1] = offset[..., 1] - pos_shelf[..., 1] - 0.1
        x_flag = torch.logical_and((offset[..., 0]) <= 0.36, (offset[..., 0] >= 0.21))
        y_flag = torch.logical_and(-0.08 >= (offset[..., 1]), (offset[..., 1]) >= -0.17)
        z_flag = torch.logical_and(offset[..., 2] <= 0.16 + 0.005, offset[..., 2] >= 0.14)
        is_book_in_shelf = torch.logical_and(torch.logical_and(x_flag, y_flag),  z_flag)

        # NOTE (stao): GPU sim can be fast but unstable. Angular velocity is rather high despite it not really rotating
        is_book_static = self.book_A.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_book_grasped = self.agent.is_grasping(self.book_A)
        success = is_book_in_shelf * is_book_static * (~is_book_grasped)
        return {
            "is_book_grasped": is_book_grasped,
            "is_book_in_shelf": is_book_in_shelf,
            "is_book_static": is_book_static,
            "success": success
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                shelf_pose=self.shelf.pose.raw_pose,
                book_pose=self.book_A.pose.raw_pose,
                tcp_to_shelf_pos=self.shelf.pose.p - self.agent.tcp.pose.p,
                tcp_to_book_pos=self.book_A.pose.p - self.agent.tcp.pose.p,
                book_to_shelf_pos=self.shelf.pose.p - self.book_A.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # rotation reward as cosine similarity between peg direction vectors
        # peg center of mass to end of peg, (1,0,0), rotated by peg pose rotation
        # dot product with its goal orientation: (0,0,1) or (0,0,-1)
        qmats = rotation_conversions.quaternion_to_matrix(self.book_A.pose.q)
        vec = torch.tensor([-1.0, 0, 0], device=self.device)
        goal_vec = torch.tensor([0, 0, 1.0], device=self.device)
        rot_vec = (qmats @ vec).view(-1, 3)
        # abs since (0,0,-1) is also valid, values in [0,1]
        rot_rew = (rot_vec @ goal_vec).view(-1).abs()
        reward = rot_rew

        # position reward using common maniskill distance reward pattern
        # giving reward in [0,1] for moving center of mass toward half length above table
        z_dist = torch.abs(self.book_A.pose.p[:, 2] - 0.16)
        reward += 1 - torch.tanh(5 * z_dist)

        # small reward to motivate initial reaching
        # initially, we want to reach and grip peg
        to_grip_vec = self.book_A.pose.p - self.agent.tcp.pose.p
        to_grip_dist = torch.linalg.norm(to_grip_vec, axis=1)
        reaching_rew = 1 - torch.tanh(5 * to_grip_dist)
        # reaching reward granted if gripping block
        reaching_rew[self.agent.is_grasping(self.book_A)] = 1
        # weight reaching reward less
        reaching_rew = reaching_rew / 5
        reward += reaching_rew

        reward[info["success"]] = 3

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8
    

if __name__ == "__main__":
    # Now you can load this safe environment
    env = gym.make(
        "PlaceBookInShelf-v1", 
        # robot_uids="dual_panda", # Force the dual panda
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
    
