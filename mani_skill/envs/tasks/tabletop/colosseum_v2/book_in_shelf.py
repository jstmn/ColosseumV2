import os
from typing import Dict, Union

import numpy as np
import sapien
import torch
import gymnasium as gym

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.envs.tasks.tabletop.colosseum_v2.colosseum_v2_core import ColosseumV2Env, DisabledVariationFactors, PlacementRegion


@register_env("PlaceBookInShelf-v1", max_episode_steps=50)
class PlaceBookEnv(ColosseumV2Env):
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

    DISABLED_VARIATION_FACTORS = DisabledVariationFactors(
        MO_size=True,
        RO_size=True,
    )
    DEFAULT_BOOK_REGION = PlacementRegion(x_lims=(-0.2, 0.0), y_lims=(-0.4, -0.2))
    DEFAULT_SHELF_REGION = PlacementRegion(x_lims=(-0.4, -0.2), y_lims=(-0.4, -0.2))

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs
    ):
        self._shelf_dist_origin_to_furtherst_negative_x = 0.2
        self._book_to_shelf_padding = 0.0
        # ^ this is the distance from the origin of the shelf to the furthest negative x-axis of the shelf
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose1 = sapien_utils.look_at(eye=[-0.3, 0, 0.6], target=[-0.1, 0, -0.1])
        pose2 = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[0.1, 0, -0.1])
        return self.update_camera_configs(
            [
                CameraConfig("external1_camera", pose1, 224, 224, np.pi / 2, 0.01, 100),
                CameraConfig("external2_camera", pose2, 224, 224, np.pi / 2, 0.01, 100),
            ]
        )

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([-0.6, -0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0])) # Loads the panda arm


    def _load_scene(self, options: dict):

        self._book_region = self.update_placement_region(self.DEFAULT_BOOK_REGION)
        self._shelf_region = self.update_placement_region(self.DEFAULT_SHELF_REGION)

        def get_shelf_builder():
            return self.get_glb_asset_builder(
                glb_filepath=os.path.join(PACKAGE_ASSET_DIR, 'book_in_shelf/BookShelf.glb'),
                object_type="RO",
            )

        def get_book_builder():
            return self.get_glb_asset_builder(
                glb_filepath=os.path.join(PACKAGE_ASSET_DIR ,'book_in_shelf/simple_book_1.glb'),
                object_type="MO",
            )

        self.shelf = self.add_asset_to_scene(get_shelf_builder, name="shelf", physics_type="kinematic", object_type="RO")
        self.book_A = self.add_asset_to_scene(get_book_builder, name="book_A", physics_type="dynamic", object_type="MO")
        self.load_scene_hook(manipulation_objects=[self.book_A], receiving_objects=[self.shelf])


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)

            book_xyz = torch.zeros((b, 3))
            book_xyz[:, 2] = 0.089
            sampler = randomization.UniformPlacementSampler(bounds=self._book_region.to_bounds(), batch_size=b, device=self.device)
            # radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001
            radius = 0.125
            bookA_xy = sampler.sample(radius, max_trials=100)
            # ^ I don't think this does anything, because 'fixture_positions' is not set

            book_xyz[:, :2] = bookA_xy
            self.book_A.set_pose(Pose.create_from_pq(p=book_xyz.clone(), q=torch.tensor([0.06, -0.162, -0.296, 0.940]).repeat(b,1)))

            shelf_xyz = torch.zeros((b, 3))
            shelf_xyz[..., 0] = self._shelf_region.x_lims[1] + self._shelf_dist_origin_to_furtherst_negative_x + \
                (torch.rand(b, device=self.device) * 0.1) + self._book_to_shelf_padding
            shelf_xyz[..., 1] = self._shelf_region.y_lims[0] + torch.rand(b, device=self.device) * self._shelf_region.width_y
            shelf_xyz[..., 2] = 0
            self.shelf.set_pose(Pose.create_from_pq(p=shelf_xyz, q=[-0.5, -0.5, 0.5, 0.5]))

            self.initialize_episode_hook(env_idx, mo_pose=book_xyz)
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
