from typing import Any, Dict, Union
import numpy as np
import sapien
import torch
import trimesh
import os
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.robocasa.scene_builder import RoboCasaSceneBuilder
from mani_skill.utils.structs.pose import Pose
from math import fabs
from mani_skill.envs.tasks.tabletop.colosseum_v2.distraction_set import DistractionSet

@register_env("RobocasaDemo-v1", max_episode_steps=50)
class RoboCasaDemoEnv(BaseEnv):
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
        return self.update_camera_configs([CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)])

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([-0.6, -0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0])) # Loads the panda arm

    def _load_scene(self, options: dict):
        # self.table_scene = TableSceneBuilder(
        #     env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        # )
        # self.table_scene.build()
        
        self.cabinet_scene = RoboCasaSceneBuilder(
            env=self, init_robot_base_pos=sapien.Pose(p=[4, -0.6, 0.94], q=[ 0.7071, 0, 0, 0.7071]) )
        self.cabinet_scene.build(build_config_idxs=[1])
        
    @staticmethod
    def add_glb_asset_to_scene(scene, glb_filepath, pose, name, scale, type="static"):
        """Load GLB file as a static actor in the scene"""
        builder = scene.create_actor_builder()
        builder.add_visual_from_file(glb_filepath, scale=scale)
        builder.add_multiple_convex_collisions_from_file(glb_filepath, decomposition="coacd")
        builder.set_initial_pose(pose)
        if type=="dynamic":
            actor = builder.build_dynamic(name)
        else:
            actor = builder.build_static(name)
        return actor

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            # b = len(env_idx)
            # self.table_scene.initialize(env_idx)

            # xyz = torch.zeros((b, 3))
            # xyz[:, 2] = 0.422
            # # xy = torch.rand((b, 2)) * 0.2 - 0.1
            # region = [[0.111, 0.368],[0.395, -0.14]]
            # sampler = randomization.UniformPlacementSampler(
            #     bounds=region, batch_size=b, device=self.device
            # )
            # radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001
            # soda_xy = sampler.sample(radius, 100)
            # # # cubeB_xy = xy + sampler.sample(radius, 100, verbose=False)

            # xyz[:, :2] = soda_xy
            # # qs = randomization.random_quaternions(
            # #     b,
            # #     lock_x=True,
            # #     lock_y=True,
            # #     lock_z=True,
            # # )
            # # [0.854,0.471,0.212,0.068] - q for sleeping book
            # # [0.748, 0.279, -0.464, 0.384] - q for other side facing book
            # self.soda.set_pose(Pose.create_from_pq(p=xyz.clone(), q=torch.tensor([0.0, 0, 0.7071, 0.7071]).repeat(b,1)))

            # xyz[:, :2] = cubeB_xy
            # qs = randomization.random_quaternions(
            #     b,
            #     lock_x=True,
            #     lock_y=True,
            #     lock_z=False,
            # )
            # self.cubeB.set_pose(Pose.create_from_pq(p=xyz, q=qs))
            return

    def evaluate(self):
        return {
            "success": True
        }
