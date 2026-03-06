from typing import Union
import numpy as np
import sapien
import torch
import os
from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.envs.tasks.tabletop.colosseum_v2.colosseum_v2_core import ColosseumV2Env, PlacementRegion


@register_env("HangClothingFrameOnPole-v1", max_episode_steps=50)
class HangClothingFrameOnPoleEnv(ColosseumV2Env):
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
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)


    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[-0.3, 0, 0.6], target=[-0.1, 0, -0.1])
        return self.update_camera_configs([CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)])

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([-0.6, -0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0]))


    def _load_scene(self, options: dict):
        def clothing_frame_builder():
            return self.get_glb_asset_builder(
                glb_filepath=os.path.join(PACKAGE_ASSET_DIR, 'ClothHangingFrameTransfer/Chrome_Metal_Hanger.glb'),
                object_type="MO",
                initial_pose=sapien.Pose(p=[-0.3, -0.4, 0.431], q=[0.548,0.5,0.5,0.453]),
            )
        def rack1_builder():
            return self.get_glb_asset_builder(
                glb_filepath=os.path.join(PACKAGE_ASSET_DIR, 'ClothHangingFrameTransfer/clothes_rack.glb'),
                object_type="RO",
                initial_pose=sapien.Pose(p=[-0.1, 0.2, 0.08], q=[0,0,0.7071,0.7071]),
                scale=(0.01,0.004,0.005),
            )

        def rack2_builder():
            return self.get_glb_asset_builder(
                glb_filepath=os.path.join(PACKAGE_ASSET_DIR, 'ClothHangingFrameTransfer/clothes_rack.glb'),
                object_type="RO",
                initial_pose=sapien.Pose(p=[-0.116, -0.4, 0.08], q=[0,0,0.7071,0.7071]),
                scale=(0.01,0.004,0.005),
            )
        
        self.clothing_frame = self.add_asset_to_scene(clothing_frame_builder, name="clothing_frame", physics_type="dynamic", object_type="MO")
        self.rack1 = self.add_asset_to_scene(rack1_builder, name="rack1", physics_type="static", object_type="RO")
        self.rack2 = self.add_asset_to_scene(rack2_builder, name="rack2", physics_type="static", object_type="RO")
        self.load_scene_hook(manipulation_objects=[self.clothing_frame], receiving_objects=[self.rack1, self.rack2])

        self._frame_region = self.update_placement_region(
            # Ground-truth from legacy sampling:
            # x = -torch.rand((b, 1))*0.3-0.1
            # xyz[:, 0] = x
            # xyz[:, 1] = -0.4
            PlacementRegion.from_center_and_width(
                # x in [-0.4, -0.1], y fixed at -0.4 (epsilon width for PlacementRegion)
                center=(-0.25, -0.4),
                width=(0.3, 0.0),
            )
        )


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            xyz = torch.zeros((b, 3))
            # x = -torch.rand((b, 1))*0.3-0.1
            # xyz[:, 0] = x
            # xyz[:, 1] = -0.4
            xyz[:, 0:2] = self._frame_region.sample_xy(b, device=self.device)
            xyz[:, 2] = 0.431
            self.clothing_frame.set_pose(Pose.create_from_pq(p=xyz, q=torch.tensor([0.548,0.5,0.5,0.453]).repeat(b,1)))
            self.initialize_episode_hook(env_idx, mo_pose=xyz)


    def evaluate(self):
        # # NOTE (stao): GPU sim can be fast but unstable. Angular velocity is rather high despite it not really rotating
        is_soda_static = self.clothing_frame.is_static(lin_thresh=1e-1, ang_thresh=1) # Not working well
        is_soda_on_table = torch.logical_and(self.clothing_frame.pose.p[:,2] < 0.43,self.clothing_frame.pose.p[:,2] > 0.4)
        success = (is_soda_on_table)
        return {
            "is_frame_on_pole": is_soda_on_table,
            "is_frame_static": is_soda_static,
            "success": success
        }