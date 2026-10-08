import numpy as np
import sapien
import torch
from mani_skill.envs.mpcm.camera_configs import get_camera_configs, get_human_render_camera_config
from mani_skill.envs.mpcm.constants import (
    ALL_CAMERA_NAMES,
    MANISKILL_CAMERA_HEIGHT,
    MANISKILL_CAMERA_WIDTH,
    POINTCLOUD_CAM_HEIGHT,
    POINTCLOUD_CAM_WIDTH,
    PULL_CUBE_TOOL_MAX_N_STEPS,
)
from mani_skill.envs.mpcm.panda_wrist_cam import PandaWristCam, reset_wristcam_robot, wrist_camera_config
from mani_skill.envs.mpcm.pick_cube_envs import TABLE_POSE
from mani_skill.envs.tasks.tabletop.colosseum_v2.colosseum_v2_core import ColosseumV2Env, DisabledPerturbationFactors
from mani_skill.envs.utils import randomization
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

TOOL_SPAWN_CENTER = (0.35, 0.0)
CUBE_SPAWN_CENTER = (0.75, 0.0)
# SCENE_LOOKAT_CENTER = (0.55, 0.0)
SCENE_LOOKAT_CENTER = (0.65, 0.0)


class PullCubeToolBaseEnv(ColosseumV2Env):
    """PickCube-v2 cameras/table/wrist, official PullCubeTool cube+hook and success."""

    ENV_ID: str
    SUPPORTED_ROBOTS = ["panda_wristcam2"]
    agent: PandaWristCam
    DISABLED_PERTURBATION_FACTORS = DisabledPerturbationFactors()

    goal_radius = 0.3
    cube_half_size = 0.02
    handle_length = 0.2
    hook_length = 0.05
    width = 0.05
    height = 0.05
    cube_size = 0.02
    arm_reach = 0.35
    # Tool must move this far from spawn so pushing the cube without the hook is not success.
    # Larger than typical spawn-settle motion (~1 cm drop).
    tool_min_displacement = 0.04

    def __init__(
        self,
        *args,
        included_cameras: list[str] | None = None,
        randomize_spawn: bool,
        camera_width: int | None = None,
        camera_height: int | None = None,
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

        self._randomize_spawn = randomize_spawn
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
        self.tool_spawn_center = TOOL_SPAWN_CENTER
        self.cube_spawn_center = CUBE_SPAWN_CENTER
        self.tool_spawn_half_size = 0.08
        self.cube_spawn_half_size_x = 0.08
        self.cube_spawn_half_size_y = 0.12
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
    def _default_sensor_configs(self):
        offset_to_panda = 0.1
        target = [SCENE_LOOKAT_CENTER[0] - offset_to_panda, SCENE_LOOKAT_CENTER[1], 0.1]
        cameras_origin = [SCENE_LOOKAT_CENTER[0] - offset_to_panda, SCENE_LOOKAT_CENTER[1], 0.0]
        # cameras_origin_xy_offset = 0.75
        cameras_origin_xy_offset = 0.45
        cameras_origin_z_offset = 0.2
        configs = [
            *get_camera_configs(
                cameras_origin=cameras_origin,
                cameras_origin_xy_offset=cameras_origin_xy_offset,
                cameras_origin_z_offset=cameras_origin_z_offset,
                target=target,
                is_pointcloud=self._is_pointcloud,
                cube_spawn_center=np.array([SCENE_LOOKAT_CENTER[0], SCENE_LOOKAT_CENTER[1], 0.0]),
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
            eye=[SCENE_LOOKAT_CENTER[0] + 0.45, SCENE_LOOKAT_CENTER[1] + 0.45, 0.4],
            target=[SCENE_LOOKAT_CENTER[0] - 0.1, SCENE_LOOKAT_CENTER[1], 0.1],
            shader="rt-fast",
        )

    def _build_l_shaped_tool(self, handle_length, hook_length, width, height):
        builder = self.scene.create_actor_builder()

        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 0, 0, 1])
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0

        builder.add_box_collision(
            sapien.Pose([handle_length / 2, 0, 0]),
            [handle_length / 2, width / 2, height / 2],
            density=500,
        )
        builder.add_box_visual(
            sapien.Pose([handle_length / 2, 0, 0]),
            [handle_length / 2, width / 2, height / 2],
            material=mat,
        )

        builder.add_box_collision(
            sapien.Pose([handle_length - hook_length / 2, width, 0]),
            [hook_length / 2, width, height / 2],
        )
        builder.add_box_visual(
            sapien.Pose([handle_length - hook_length / 2, width, 0]),
            [hook_length / 2, width, height / 2],
            material=mat,
        )

        builder.initial_pose = sapien.Pose(p=[self.tool_spawn_center[0], self.tool_spawn_center[1], height / 2])
        return builder.build(name="l_shape_tool")

    def _load_scene(self, options: dict):
        with torch.device(self.device):
            self.cube = actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=np.array([12, 42, 160, 255]) / 255,
                name="cube",
                body_type="dynamic",
                initial_pose=sapien.Pose(p=[self.cube_spawn_center[0], self.cube_spawn_center[1], self.cube_half_size]),
            )
            self.l_shape_tool = self._build_l_shaped_tool(
                handle_length=self.handle_length,
                hook_length=self.hook_length,
                width=self.width,
                height=self.height,
            )
            self.load_scene_hook(manipulation_objects=[self.cube], receiving_objects=[self.l_shape_tool])
            self._initial_tool_pos = torch.zeros((self.num_envs, 3), device=self.device)

    def _initialize_table_and_robot(self, env_idx: torch.Tensor, b: int) -> None:
        self.initialize_episode_hook(
            env_idx,
            mo_pose=self.cube.pose.p,
            ro_pose=self.l_shape_tool.pose.p,
        )
        self.table.set_pose(TABLE_POSE)
        reset_wristcam_robot(self, b, flip_joint_6=False)

    def _spawn_tool_and_cube(self, b: int) -> None:
        tool_xyz = torch.zeros((b, 3))
        cube_xyz = torch.zeros((b, 3))
        if self._randomize_spawn:
            tool_xyz[:, 0] = self.tool_spawn_center[0] + (
                torch.rand(b) * self.tool_spawn_half_size * 2 - self.tool_spawn_half_size
            )
            tool_xyz[:, 1] = self.tool_spawn_center[1] + (
                torch.rand(b) * self.tool_spawn_half_size * 2 - self.tool_spawn_half_size
            )
            cube_xyz[:, 0] = self.cube_spawn_center[0] + (
                torch.rand(b) * self.cube_spawn_half_size_x * 2 - self.cube_spawn_half_size_x
            )
            cube_xyz[:, 1] = self.cube_spawn_center[1] + (
                torch.rand(b) * self.cube_spawn_half_size_y * 2 - self.cube_spawn_half_size_y
            )
            cube_q = randomization.random_quaternions(
                b, lock_x=True, lock_y=True, lock_z=False, bounds=(-np.pi / 6, np.pi / 6)
            )
        else:
            tool_xyz[:, 0] = self.tool_spawn_center[0]
            tool_xyz[:, 1] = self.tool_spawn_center[1]
            cube_xyz[:, 0] = self.cube_spawn_center[0]
            cube_xyz[:, 1] = self.cube_spawn_center[1]
            cube_q = None
        tool_xyz[:, 2] = self.height / 2
        cube_xyz[:, 2] = self.cube_size / 2 + 0.015
        tool_q = torch.tensor([1, 0, 0, 0]).expand(b, 4)
        self.l_shape_tool.set_pose(Pose.create_from_pq(p=tool_xyz, q=tool_q))
        self.cube.set_pose(Pose.create_from_pq(p=cube_xyz, q=cube_q))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self._spawn_tool_and_cube(b)
            self._initialize_table_and_robot(env_idx, b)
            self._initial_tool_pos[env_idx] = self.l_shape_tool.pose.p[env_idx]

    def _get_obs_extra(self, info: dict):
        with torch.device(self.device):
            obs = dict(
                is_grasped=info["is_grasped"],
                tcp_xyz=self.agent.tcp_pose.p,
                tcp_quat=self.agent.tcp_pose.q,
                tcp_pose=self.agent.tcp_pose.raw_pose,
                camera_poses={},
            )
            if "state" in self.obs_mode:
                obs.update(
                    obj_pose=self.cube.pose.raw_pose,
                    tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp_pose.p,
                )
            return obs

    def evaluate(self):
        with torch.device(self.device):
            cube_pos = self.cube.pose.p
            robot_base_pos = self.agent.robot.get_links()[0].pose.p
            cube_to_base_dist = torch.linalg.norm(cube_pos[:, :2] - robot_base_pos[:, :2], dim=1)
            cube_pulled_close = cube_to_base_dist < 0.6
            tool_displacement = torch.linalg.norm(self.l_shape_tool.pose.p - self._initial_tool_pos, dim=1)
            tool_moved = tool_displacement > self.tool_min_displacement
            is_grasped = self.agent.is_grasping(self.l_shape_tool, max_angle=20)
            return {
                "success": cube_pulled_close & tool_moved,
                "is_grasped": is_grasped,
                "tool_moved": tool_moved,
                "tcp_pose": self.agent.tcp_pose.raw_pose,
            }


@register_env("PullCubeTool-v2", max_episode_steps=PULL_CUBE_TOOL_MAX_N_STEPS)
class PullCubeToolWristMountedEnv(PullCubeToolBaseEnv):
    ENV_ID = "PullCubeTool-v2"

    def __init__(self, *args, included_cameras: list[str] | None = None, robot_init_qpos_noise=0.02, **kwargs):
        super().__init__(
            *args,
            included_cameras=included_cameras,
            randomize_spawn=True,
            robot_init_qpos_noise=robot_init_qpos_noise,
            **kwargs,
        )
