from typing import Union

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.registration import register_env
from mani_skill.utils.sapien_utils import look_at
from mani_skill.utils.structs.pose import Pose
from mani_skill.envs.tasks.tabletop.colosseum_v2.colosseum_v2_core import ColosseumV2Env, DisabledVariationFactors, PlacementRegion

@register_env("LiftPegUprightColosseumV2-v1", max_episode_steps=50)
class LiftPegUprightColosseumV2Env(ColosseumV2Env):
    r"""
    **Task Description:**
    A simple task where the objective is to move a peg laying on the table to any upright position on the table

    **Randomizations:**
    - the peg's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat along it's length on the table

    **Success Conditions:**
    - the absolute value of the peg's y euler angle is within 0.08 of $\pi$/2 and the z position of the peg is within 0.005 of its half-length (0.12).
    """

    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    agent: Union[Panda, Fetch]

    peg_half_width = 0.025
    peg_half_length = 0.12

    DISABLED_VARIATION_FACTORS = DisabledVariationFactors(
        RO_color=True,
        RO_texture=True,
        RO_size=True,
    )

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose1 = look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        pose2 = look_at(eye=[0.0, 0.3, 0.5], target=[-0.1, 0, 0.1])
        return self.update_camera_configs(
            [
                CameraConfig("external1_camera", pose1, 224, 224, np.pi / 2, 0.01, 100),
                CameraConfig("external2_camera", pose2, 224, 224, np.pi / 2, 0.01, 100),
            ]
        )

    @property
    def _default_human_render_camera_configs(self):
        pose = look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):


        def peg_builder():
            MO_scale = self.get_MO_scale()
            # the peg that we want to manipulate
            return actors.build_twocolor_peg(
                self.scene,
                length=self.peg_half_length * MO_scale[0],
                width=self.peg_half_width * MO_scale[1],
                color_1=np.array([176, 14, 14, 255]) / 255,
                color_2=np.array([12, 42, 160, 255]) / 255,
                name="peg",
                body_type="dynamic",
                initial_pose=sapien.Pose(p=[0, 0, 0.1]),
                return_builder=True,
            )
        self.peg = self.add_asset_to_scene(peg_builder, name="peg", physics_type="dynamic", object_type="MO")
        self.load_scene_hook(manipulation_objects=[self.peg])

        self._peg_region = self.update_placement_region(
            PlacementRegion.from_center_and_width(center=(0.0, 0.0), width=(0.2, 0.2))
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)

            xyz = torch.zeros((b, 3))
            # xyz[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[..., :2] = self._peg_region.sample_xy(b, device=self.device)
            xyz[..., 2] = self.peg_half_width
            q = euler2quat(np.pi / 2, 0, 0)

            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.peg.set_pose(obj_pose)

            self.initialize_episode_hook(env_idx, mo_pose=obj_pose)

    def evaluate(self):
        q = self.peg.pose.q
        qmat = rotation_conversions.quaternion_to_matrix(q)
        euler = rotation_conversions.matrix_to_euler_angles(qmat, "XYZ")
        is_peg_upright = (
            torch.abs(torch.abs(euler[:, 2]) - np.pi / 2) < 0.08
        )  # 0.08 radians of difference permitted
        close_to_table = torch.abs(self.peg.pose.p[:, 2] - self.peg_half_length) < 0.005
        return {
            "success": is_peg_upright & close_to_table,
        }
