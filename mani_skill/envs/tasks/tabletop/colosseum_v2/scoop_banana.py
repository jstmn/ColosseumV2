from typing import Any, Dict, Union
import os
import numpy as np
import sapien
import torch
from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.envs.tasks.tabletop.colosseum_v2.colosseum_v2_core import ColosseumV2Env

@register_env("ScoopBanana-v1", max_episode_steps=100)
class ScoopBananaEnv(ColosseumV2Env):
    """
    **Task Description**
    Take a dustpan and scoop a banana onto it.

    **Randomizations**
    - The banana's (x,y) positions are randomized on top of a table.


    **Success Conditions**
    - The banana is inside the dustpan, lifted above the table height.
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PullCubeTool-v1_rt.mp4"

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    SUPPORTED_REWARD_MODES = ("normalized_dense", "dense", "sparse", "none")
    agent: Union[Panda, Fetch]

    goal_radius = 0.3
    banana_radius: float = 0.035  # radius of the banana
    handle_length = 0.2
    hook_length = 0.05
    width = 0.24
    height = 0.15
    cube_size = 0.02
    arm_reach = 0.35

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[-0.25, 0.25, 0.5], target=[0.0, 0.0, 0.1])
        return self.update_camera_configs([
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ])

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([-0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return [
            CameraConfig(
                "render_camera",
                pose=pose,
                width=512,
                height=512,
                fov=1,
                near=0.01,
                far=100,
            )
        ]

    def _load_scene(self, options: dict):
        banana_builder = lambda: self.get_ycb_asset_builder(ycb_id="011_banana", object_type="MO")
        dustpan_builder = lambda: self.get_glb_asset_builder(
            glb_filepath=os.path.join(PACKAGE_ASSET_DIR, 'scoop_particles/dustpan.glb'),
            object_type="MO",
            initial_pose=sapien.Pose(p=[0, 0, 0.015]),
        )

        wall_builder = lambda: actors.build_box(
            self.scene, half_sizes=[0.2, 0.02, 0.1], 
            color=[201/255, 204/255, 182/255, 1], 
            name="wall", 
            body_type="kinematic", 
            add_collision=True, 
            initial_pose=sapien.Pose(p=[0.2, -0.25, 0.2], q=[0.7071, 0, 0, 0.7071]),
            return_builder=True,
        )
        self.banana = self.add_asset_to_scene(banana_builder, name="banana", physics_type="dynamic", object_type="RO")
        self.dustpan = self.add_asset_to_scene(dustpan_builder, name="dustpan", physics_type="dynamic", object_type="MO")
        self.wall = self.add_asset_to_scene(wall_builder, name="wall", physics_type="kinematic", object_type="BACKGROUND")
        self.load_scene_hook(manipulation_objects=[self.dustpan], receiving_objects=[self.banana])



    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)

            tool_xyz = torch.zeros((b, 3), device=self.device)
            tool_xyz[..., :2] = -torch.rand((b, 2), device=self.device) * 0.2 - 0.1
            tool_xyz[..., 2] = 0.015
            tool_q = torch.tensor([0.559, 0.464, 0.439, 0.529], device=self.device).expand(b, 4)

            tool_pose = Pose.create_from_pq(p=tool_xyz, q=tool_q)
            self.dustpan.set_pose(tool_pose)

            banana_xyz = torch.zeros((b, 3), device=self.device)
            banana_xyz[..., 0] = (
                self.arm_reach
                + torch.rand(b, device=self.device) * (self.handle_length)
                - 0.4
            )
            banana_xyz[..., 1] = torch.rand(b, device=self.device) * 0.3 - 0.25
            banana_xyz[..., 2] = self.banana_radius + 0.01

            banana_pose = Pose.create_from_pq(p=banana_xyz, q=torch.tensor([1, 0, 0, 0], dtype=torch.float32))
            self.banana.set_pose(banana_pose)
            wall_xyz = torch.zeros((b, 3), device=self.device)
            wall_xyz[..., 0] = banana_xyz[..., 0] + 0.15
            wall_xyz[..., 1] = banana_xyz[..., 1]
            wall_xyz[..., 2] = 0.1
            self.wall.set_pose(Pose.create_from_pq(p=wall_xyz, q=torch.tensor([0.7071, 0, 0, 0.7071], dtype=torch.float32)))

            self.initialize_episode_hook(env_idx, mo_pose=self.dustpan.pose)

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )

        if self.obs_mode_struct.use_state:
            obs.update(
                banana_pose=self.banana.pose.raw_pose,
                dustpan_pose=self.dustpan.pose.raw_pose,
            )

        return obs

    def evaluate(self):
        banana_pos = self.banana.pose.p

        dustpan_pos = self.dustpan.pose.p

        z_dist = banana_pos[..., 2] - dustpan_pos[..., 2]
        xy_dist = torch.linalg.norm(banana_pos[..., :2] - dustpan_pos[..., :2], dim=1)
        # Success condition - cube is pulled close enough
        banana_z_close_flag = torch.logical_and(z_dist < ScoopBananaEnv.banana_radius + 0.02, z_dist > 0.0)
        banana_xy_close_flag = xy_dist < ScoopBananaEnv.width * 1.414 / 2
        banana_pulled_close = torch.logical_and(banana_z_close_flag, banana_xy_close_flag)
        # is_banana_static = self.banana.is_static(lin_thresh=1e-1, ang_thresh=0.5)
        # print(is_banana_static)
        success = banana_pulled_close

        return {
            "success": success,
            "banana_pulled_close": banana_pulled_close,
            "banana_z_close_flag": banana_z_close_flag,
            "banana_xy_close_flag": banana_xy_close_flag,
            # "is_banana_static": is_banana_static,
        }
