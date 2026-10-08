import numpy as np
import sapien
import torch
from mani_skill.envs.mpcm.camera_configs import get_camera_configs, get_human_render_camera_config
from mani_skill.envs.mpcm.constants import (
    ALL_CAMERA_NAMES,
    MANISKILL_CAMERA_HEIGHT,
    MANISKILL_CAMERA_WIDTH,
    PICK_CUBE_MAX_N_STEPS,
    POINTCLOUD_CAM_HEIGHT,
    POINTCLOUD_CAM_WIDTH,
)
from mani_skill.envs.mpcm.panda_wrist_cam import PandaWristCam, reset_wristcam_robot, wrist_camera_config
from mani_skill.envs.tasks.tabletop.colosseum_v2.colosseum_v2_core import (
    ColosseumV2Env,
    DisabledPerturbationFactors,
)
from mani_skill.envs.utils import randomization
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from transforms3d.euler import euler2quat

CUBE_SPAWN_CENTER = (0.615, 0.0)
# 1cm offset from the table to account for the vention plate on the real hardware
TABLE_POSE = sapien.Pose(p=[-0.12, 0, -0.9196429 - 0.01], q=euler2quat(0, 0, np.pi))


class PickCubeBaseEnv(ColosseumV2Env):
    """
    Stores the common evaluation code for the PickCube-v2 environment.
    """

    ENV_ID: str
    SUPPORTED_ROBOTS = ["panda_wristcam2"]
    agent: PandaWristCam
    DISABLED_PERTURBATION_FACTORS = DisabledPerturbationFactors(
        RO_color=True,
        RO_texture=True,
        RO_size=True,
    )

    def __init__(
        self,
        *args,
        included_cameras: list[str] | None = None,
        camera_width: int | None = None,
        camera_height: int | None = None,
        goal_height: float = 0.2,
        robot_init_qpos_noise=0.02,
        is_pointcloud: bool = False,
        **kwargs,
    ):
        if included_cameras is not None:
            assert isinstance(included_cameras, list), "included_cameras must be a list"
            assert len(included_cameras) > 0, "included_cameras must be non-empty"
            for camera in included_cameras:
                assert isinstance(camera, str), "included_cameras must be a list of strings"
                assert camera in ALL_CAMERA_NAMES, f"Camera {camera} not found. Valid cameras: {ALL_CAMERA_NAMES}"
            included_cameras = list(included_cameras)

        self._camera_poses: dict[str, sapien.Pose] = {}
        assert isinstance(is_pointcloud, bool), f"is_pointcloud must be bool, got {type(is_pointcloud).__name__}"
        assert not is_pointcloud, "is_pointcloud=True is not supported for these ColosseumV2 envs"
        self._is_pointcloud = is_pointcloud
        if camera_width is None:
            camera_width = POINTCLOUD_CAM_WIDTH if self._is_pointcloud else MANISKILL_CAMERA_WIDTH
        if camera_height is None:
            camera_height = POINTCLOUD_CAM_HEIGHT if self._is_pointcloud else MANISKILL_CAMERA_HEIGHT
        assert isinstance(camera_width, int), f"camera_width must be an int, got {type(camera_width).__name__}"
        assert isinstance(camera_height, int), f"camera_height must be an int, got {type(camera_height).__name__}"
        assert camera_width > 0, f"camera_width must be positive, got {camera_width}"
        assert camera_height > 0, f"camera_height must be positive, got {camera_height}"
        self._camera_width = camera_width
        self._camera_height = camera_height
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.cube_half_size = 0.02
        assert (
            isinstance(goal_height, float) and goal_height > self.cube_half_size
        ), f"goal_height={goal_height}, expected float > cube half-height {self.cube_half_size}"
        self._goal_height = goal_height
        self.goal_thresh = 0.025
        self.cube_spawn_half_size = 0.1
        self.cube_spawn_center = CUBE_SPAWN_CENTER
        kwargs.setdefault("_env_id", type(self).ENV_ID)
        super().__init__(
            *args,
            robot_uids="panda_wristcam2",
            robot_init_qpos_noise=robot_init_qpos_noise,
            reconfiguration_freq=kwargs.pop("reconfiguration_freq", 0),
            included_cameras=included_cameras,
            **kwargs,
        )

    @property
    def goal_height(self):
        return self._goal_height

    @property
    def _default_sensor_configs(self):
        # Note: BaseEnv._setup_sensors will filter the configs to only include the included_cameras
        offset_to_panda = 0.0
        # offset_to_panda = 0.1
        target = [self.cube_spawn_center[0] - offset_to_panda, self.cube_spawn_center[1], 0.1]
        cameras_origin = [self.cube_spawn_center[0] - offset_to_panda, self.cube_spawn_center[1], 0.0]
        # cameras_origin_xy_offset = 0.75
        # cameras_origin_z_offset = 0.25
        # ^ n_successes=27/44
        cameras_origin_xy_offset = 0.4
        cameras_origin_z_offset = 0.2
        # ^ n_successes=_/_
        configs = [
            *get_camera_configs(
                cameras_origin=cameras_origin,
                cameras_origin_xy_offset=cameras_origin_xy_offset,
                cameras_origin_z_offset=cameras_origin_z_offset,
                target=target,
                is_pointcloud=self._is_pointcloud,
                cube_spawn_center=np.array([self.cube_spawn_center[0], self.cube_spawn_center[1], 0.0]),
                camera_width=self._camera_width,
                camera_height=self._camera_height,
            ),
            wrist_camera_config(self.agent, self._camera_width, self._camera_height),
        ]
        configs = self.update_camera_configs(configs)
        included = self._included_cameras
        self._camera_poses = {
            cfg.uid: cfg.pose for cfg in configs if included is None or cfg.uid in included
        }
        return configs

    @property
    def _default_human_render_camera_configs(self):
        return get_human_render_camera_config(
            eye=[
                self.cube_spawn_center[0] + 0.45,
                self.cube_spawn_center[1] + 0.45,
                0.4,
            ],
            target=[self.cube_spawn_center[0] - 0.1, self.cube_spawn_center[1], 0.1],
            shader="rt-fast",
        )

    def _load_scene(self, options: dict):
        with torch.device(self.device):
            self.cube = actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=[1, 0, 0, 1],
                name="cube",
                initial_pose=sapien.Pose(
                    p=[
                        self.cube_spawn_center[0],
                        self.cube_spawn_center[1],
                        self.cube_half_size,
                    ]
                ),
            )
            self.load_scene_hook(manipulation_objects=[self.cube])
            # self.goal_site = actors.build_sphere(
            #     self.scene,
            #     radius=self.goal_thresh,
            #     color=[0, 1, 0, 0.5],
            #     name="goal_site",
            #     body_type="kinematic",
            #     add_collision=False,
            #     initial_pose=sapien.Pose(p=[self.cube_spawn_center[0], self.cube_spawn_center[1], 0.2]),
            # )
            # self._hidden_objects.append(self.goal_site)
            # ^ NOTE: The goal site will be visible in the depth image as well. Best to leave it hidden. This will more
            # closely match the real world.

    def _initialize_table_robot_and_perturbations(self, env_idx: torch.Tensor, b: int, cube_xyz: torch.Tensor) -> None:
        self.initialize_episode_hook(env_idx, mo_pose=cube_xyz)
        self.table.set_pose(TABLE_POSE)
        reset_wristcam_robot(self, b, flip_joint_6=True)

    def _get_obs_extra(self, info: dict):
        with torch.device(self.device):
            # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
            obs = dict(
                is_grasped=info["is_grasped"],
                tcp_xyz=self.agent.tcp_pose.p,
                tcp_quat=self.agent.tcp_pose.q,
                tcp_pose=self.agent.tcp_pose.raw_pose,
                # goal_pos=self.goal_site.pose.p,
                # NOTE: Camera poses are not in the opencv frame format, there is a unique frame format for ManiSkill.
                # Convert to opencv format with sapien_pose_to_opencv_extrinsic()
                camera_poses={},
            )
            # This is needed for pointcloud procesessing, but breaks obs_mode=state
            # for uid, pose in self._camera_poses.items():
            #     obs["camera_poses"][uid] = pose.to_transformation_matrix()
            if "state" in self.obs_mode:
                obs.update(
                    obj_pose=self.cube.pose.raw_pose,
                    tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp_pose.p,
                    # obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
                )
            return obs

    def evaluate(self):
        with torch.device(self.device):
            # Updated so that only the height of the cube is considered
            is_obj_raised = self.cube.pose.p[:, 2] > self._goal_height
            # is_obj_placed = (
            #     torch.linalg.norm(self.goal_site.pose.p[:, 2].view(-1, 1) - self.cube.pose.p[:, 2].view(-1, 1), axis=1)
            #     <= self.goal_thresh
            # )
            # is_obj_placed = torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1) <= self.goal_thresh
            is_grasped = self.agent.is_grasping(self.cube)
            is_robot_static = self.agent.is_static(0.2)
            return {
                "success": is_obj_raised,
                # "success": is_obj_placed & is_robot_static,
                "is_obj_raised": is_obj_raised,
                "is_robot_static": is_robot_static,
                "is_grasped": is_grasped,
                "tcp_pose": self.agent.tcp_pose.raw_pose,
            }


@register_env("PickCube-v2-wrist", max_episode_steps=PICK_CUBE_MAX_N_STEPS)
class PickCubeWristMountedEnv(PickCubeBaseEnv):
    ENV_ID = "PickCube-v2-wrist"

    def __init__(self, *args, included_cameras: list[str] | None = None, robot_init_qpos_noise=0.02, **kwargs):
        super().__init__(
            *args,
            included_cameras=included_cameras,
            robot_init_qpos_noise=robot_init_qpos_noise,
            **kwargs,
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            cube_xyz = torch.zeros((b, 3))
            cube_xyz[:, :2] = torch.rand((b, 2)) * self.cube_spawn_half_size * 2 - self.cube_spawn_half_size
            cube_xyz[:, 0] += self.cube_spawn_center[0]
            cube_xyz[:, 1] += self.cube_spawn_center[1]
            cube_xyz[:, 2] = self.cube_half_size

            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(cube_xyz, qs))
            self._initialize_table_robot_and_perturbations(env_idx, b, cube_xyz)


# @register_env("PickCube-v2-fixed", max_episode_steps=100)
# class PickCubeMultiViewFixedEnv(PickCubeBaseEnv):
#     """
#     Exact replica of PickCube-v2, but with the cube's xy position fixed.
#     """
#
#     ENV_ID = "PickCube-v2-fixed"
#
#     def __init__(self, included_cameras: list[str], *args, robot_init_qpos_noise=0.02, **kwargs):
#         super().__init__(
#             *args,
#             included_cameras=included_cameras,
#             robot_init_qpos_noise=robot_init_qpos_noise,
#             **kwargs,
#         )
#
#     def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
#         with torch.device(self.device):
#             b = len(env_idx)
#             cube_xyz = torch.zeros((b, 3))
#             cube_xyz[:, 0] = self.cube_spawn_center[0]
#             cube_xyz[:, 1] = self.cube_spawn_center[1]
#             cube_xyz[:, 2] = self.cube_half_size
#
#             qs = None
#             self.cube.set_pose(Pose.create_from_pq(cube_xyz, qs))
#             self._initialize_table_robot_and_perturbations(env_idx, b, cube_xyz)
#
#
# @register_env("PickCube-v2", max_episode_steps=100)
# class PickCubeMultiViewEnv(PickCubeBaseEnv):
#     """
#     Changes from the original PickCube-v1:
#         - Added a new argument `dont_randomize_cube_q` to control whether the cube's z-axis rotation is randomized
#         - Updated the success condition to only consider the height of the cube
#         - Added a new argument `cube_xy` to control the cube's xy position
#         - Added a new argument `n_cams` to control the number of cameras
#         - Only the height of the cube is considered for the success condition
#     """
#
#     ENV_ID = "PickCube-v2"
#
#     def __init__(self, included_cameras: list[str], *args, robot_init_qpos_noise=0.02, **kwargs):
#         super().__init__(
#             *args,
#             included_cameras=included_cameras,
#             robot_init_qpos_noise=robot_init_qpos_noise,
#             **kwargs,
#         )
#
#     def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
#         with torch.device(self.device):
#             b = len(env_idx)
#             cube_xyz = torch.zeros((b, 3))
#             cube_xyz[:, :2] = torch.rand((b, 2)) * self.cube_spawn_half_size * 2 - self.cube_spawn_half_size
#
#             cube_xyz[:, 0] += self.cube_spawn_center[0]
#             cube_xyz[:, 1] += self.cube_spawn_center[1]
#             cube_xyz[:, 2] = self.cube_half_size
#
#             qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
#             self.cube.set_pose(Pose.create_from_pq(cube_xyz, qs))
#             self._initialize_table_robot_and_perturbations(env_idx, b, cube_xyz)
