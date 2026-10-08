import numpy as np
import sapien
import torch
from mani_skill.envs.mpcm.camera_configs import get_camera_configs
from mani_skill.envs.mpcm.constants import (
    FORMER_TABLE_BASE_X,
    LIFT_PEG_UPRIGHT_MAX_N_STEPS,
    MANISKILL_CAMERA_HEIGHT,
    MANISKILL_CAMERA_WIDTH,
)
from mani_skill.envs.mpcm.panda_wrist_cam import (
    PandaWristCam,
    reset_wristcam_robot,
    shift_table_to_origin_base,
    wrist_camera_config,
)
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.registration import register_env
from mani_skill.utils.sapien_utils import look_at
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from transforms3d.euler import euler2quat


@register_env("LiftPegUpright-v2", max_episode_steps=LIFT_PEG_UPRIGHT_MAX_N_STEPS)
class LiftPegUprightEnv(BaseEnv):
    r"""
    **Task Description:**
    A simple task where the objective is to move a peg laying on the table to any upright position on the table

    **Randomizations:**
    - the peg's xy position is uniform in a 0.2 m × 0.2 m square, x ∈ [0.36, 0.56], y ∈ [-0.1, 0.1]. It is placed flat along its length on the table

    **Success Conditions:**
    - the absolute value of the peg's y euler angle is within 0.12 rad (6.9°) of $\pi$/2 and the z position of the peg is within 0.01 of its half-length (0.12).
    """

    _sample_video_link = (
        "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/LiftPegUpright-v1_rt.mp4"
    )
    SUPPORTED_ROBOTS = ("panda_wristcam2",)
    agent: PandaWristCam

    peg_half_width = 0.025
    peg_half_length = 0.12
    # Same 0.2 m × 0.2 m square as before. Only the center moved, from x=0.615 to x=0.46.
    peg_spawn_center_x = 0.46
    peg_spawn_center_y = 0.0
    peg_spawn_half_x = 0.1
    peg_spawn_half_y = 0.1

    def __init__(
        self, *args, robot_uids="panda_wristcam2", robot_init_qpos_noise=0.02, included_cameras=None, **kwargs
    ):
        self._included_cameras = list(included_cameras) if included_cameras else None
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        cube_spawn_center = np.array([self.peg_spawn_center_x + 0.1, self.peg_spawn_center_y, 0.0])
        offset_to_panda = 0.1
        target = (float(cube_spawn_center[0] - offset_to_panda), float(cube_spawn_center[1]), 0.2)
        cameras_origin = (target[0], target[1], 0.0)
        configs = [
            *get_camera_configs(
                cameras_origin=cameras_origin,
                cameras_origin_xy_offset=0.6,
                cameras_origin_z_offset=0.15,
                target=target,
                is_pointcloud=False,
                cube_spawn_center=cube_spawn_center,
            ),
            wrist_camera_config(self.agent, MANISKILL_CAMERA_WIDTH, MANISKILL_CAMERA_HEIGHT),
        ]
        if self._included_cameras is not None:
            configs = [cfg for cfg in configs if cfg.uid in self._included_cameras]
        return configs

    @property
    def _default_human_render_camera_configs(self):
        spawn_dx = self.peg_spawn_center_x + FORMER_TABLE_BASE_X
        pose = look_at(
            [0.6 - FORMER_TABLE_BASE_X + spawn_dx, 0.7, 0.6],
            [self.peg_spawn_center_x, self.peg_spawn_center_y, 0.35],
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()

        self.peg = actors.build_twocolor_peg(
            self.scene,
            length=self.peg_half_length,
            width=self.peg_half_width,
            color_1=np.array([176, 14, 14, 255]) / 255,
            color_2=np.array([12, 42, 160, 255]) / 255,
            name="peg",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[self.peg_spawn_center_x, self.peg_spawn_center_y, 0.1]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            reset_wristcam_robot(self, b, flip_joint_6=False)
            shift_table_to_origin_base(self.table_scene)

            xyz = torch.zeros((b, 3))
            xy = torch.rand((b, 2))
            xyz[..., 0] = xy[..., 0] * (2 * self.peg_spawn_half_x) - self.peg_spawn_half_x + self.peg_spawn_center_x
            xyz[..., 1] = xy[..., 1] * (2 * self.peg_spawn_half_y) - self.peg_spawn_half_y + self.peg_spawn_center_y
            xyz[..., 2] = self.peg_half_width
            q = euler2quat(np.pi / 2, 0, 0)

            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.peg.set_pose(obj_pose)

    def evaluate(self):
        q = self.peg.pose.q
        qmat = rotation_conversions.quaternion_to_matrix(q)
        euler = rotation_conversions.matrix_to_euler_angles(qmat, "XYZ")
        is_peg_upright = torch.abs(torch.abs(euler[:, 2]) - np.pi / 2) < 0.12
        close_to_table = torch.abs(self.peg.pose.p[:, 2] - self.peg_half_length) < 0.01
        return {
            "success": is_peg_upright & close_to_table,
        }

    def _get_obs_extra(self, info: dict):
        obs = {
            "tcp_pose": self.agent.tcp.pose.raw_pose,
        }
        if self.obs_mode_struct.use_state:
            obs.update(
                obj_pose=self.peg.pose.raw_pose,
            )
        return obs
