import numpy as np

from mani_skill.envs.mpcm.constants import (
    MANISKILL_CAMERA_HEIGHT,
    MANISKILL_CAMERA_WIDTH,
    POINTCLOUD_CAM_HEIGHT,
    POINTCLOUD_CAM_WIDTH,
)
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

REALSENSE_DEPTH_FOV_VERTICAL_RAD = 58.0 * np.pi / 180
REALSENSE_DEPTH_FOV_HORIZONTAL_RAD = 87.0 * np.pi / 180

SHADER = "default"


def get_camera_configs(
    cameras_origin: tuple[float, float, float],
    cameras_origin_xy_offset: float,
    cameras_origin_z_offset: float,
    target: tuple[float, float, float],
    is_pointcloud: bool,
    cube_spawn_center: np.array,
    camera_width: int | None = None,
    camera_height: int | None = None,
):
    """This uses the following convention. There is a 'camera_origin' for which all cameras are spawned around. Specifically:
        - camera_center: camera_origin + (cameras_origin_xy_offset, 0, cameras_origin_z_offset)
        - camera_left: camera_origin + (0, -cameras_origin_xy_offset, cameras_origin_z_offset)
        - camera_right: camera_origin + (0, cameras_origin_xy_offset, cameras_origin_z_offset)

    Args:
        cameras_origin (tuple[float, float, float]): The origin of the camera_origin
        cameras_origin_xy_offset (float): The xy offset of the camera_origin
        cameras_origin_z_offset (float): The z offset of the camera_origin
        target (tuple[float, float, float]): The target position
        n_cams (int): The number of cameras
    """
    if camera_width is None:
        camera_width = POINTCLOUD_CAM_WIDTH if is_pointcloud else MANISKILL_CAMERA_WIDTH
    if camera_height is None:
        camera_height = POINTCLOUD_CAM_HEIGHT if is_pointcloud else MANISKILL_CAMERA_HEIGHT
    cam_origin = np.array(cameras_origin)
    left_offset = np.array([0, -cameras_origin_xy_offset, 0])
    right_offset = np.array([0, cameras_origin_xy_offset, 0])
    center_offset = np.array([cameras_origin_xy_offset, 0, 0])
    z_offset = np.array([0, 0, cameras_origin_z_offset])
    pose_center = sapien_utils.look_at(eye=cam_origin + center_offset + z_offset, target=target)
    pose_left = sapien_utils.look_at(eye=cam_origin + left_offset + z_offset, target=target)
    pose_right = sapien_utils.look_at(eye=cam_origin + right_offset + z_offset, target=target)

    cams = [
        CameraConfig(
            uid="camera_center",
            pose=pose_center,
            width=camera_width,
            height=camera_height,
            fov=REALSENSE_DEPTH_FOV_VERTICAL_RAD,
            near=0.01,
            far=100,
            shader_pack=SHADER,
        ),
        CameraConfig(
            uid="camera_left",
            pose=pose_left,
            width=camera_width,
            height=camera_height,
            fov=REALSENSE_DEPTH_FOV_VERTICAL_RAD,
            near=0.01,
            far=100,
            shader_pack=SHADER,
        ),
        CameraConfig(
            uid="camera_right",
            pose=pose_right,
            width=camera_width,
            height=camera_height,
            fov=REALSENSE_DEPTH_FOV_VERTICAL_RAD,
            near=0.01,
            far=100,
            shader_pack=SHADER,
        ),
    ]

    pose_base = sapien_utils.look_at(
        eye=cube_spawn_center + np.array([0.3, 0.5, 0.6]), target=cube_spawn_center + np.array([-0.1, 0, 0.1])
    )
    # This is the same pose offset as 'camera_base', see:https://github.com/mani-skill/ManiSkill/blob/main/mani_skill/envs/tasks/tabletop/pick_cube.py#L66-L71
    cams += [
        CameraConfig(
            uid="camera_base",
            pose=pose_base,
            width=camera_width,
            height=camera_height,
            fov=REALSENSE_DEPTH_FOV_VERTICAL_RAD,
            near=0.01,
            far=100,
            shader_pack=SHADER,
        )
    ]
    cams += [CameraConfig("base_camera", pose_base, 128, 128, np.pi / 2, 0.01, 100)]
    cams += low_y_camera_configs(np.asarray(cube_spawn_center), camera_width, camera_height)
    return cams


def low_y_camera_configs(
    cube_spawn_center: np.ndarray,
    camera_width: int,
    camera_height: int,
    x_offset: float = 0.2,
    y_offset: float = 0.3,
) -> list[CameraConfig]:
    """Table-height cameras on +y and -y, aimed at the spawn center."""
    assert x_offset > 0, f"x_offset={x_offset}"
    assert y_offset > 0, f"y_offset={y_offset}"
    center = np.asarray(cube_spawn_center)
    target = center + np.array([-0.1, 0, 0.1])
    configs = []
    for uid, signed_y_offset in (("camera_low_pos_y", y_offset), ("camera_low_neg_y", -y_offset)):
        configs.append(
            CameraConfig(
                uid=uid,
                pose=sapien_utils.look_at(eye=center + np.array([x_offset, signed_y_offset, 0.2]), target=target),
                width=camera_width,
                height=camera_height,
                fov=REALSENSE_DEPTH_FOV_VERTICAL_RAD,
                near=0.01,
                far=100,
                shader_pack=SHADER,
            )
        )
    return configs


def get_camera_configs_random(xy_radius, z_offset, target: tuple[float, float, float], n_cams: int = 1):
    raise NotImplementedError("No longer used")
    # randomly generate n_cams poses at a constant distance from the target, the z is z_offset,
    # the xy is randomly generated so that the distance from the target is xy_radius
    # so we will sample 3 random angles and then compute the x and y
    width = MANISKILL_CAMERA_WIDTH
    height = MANISKILL_CAMERA_HEIGHT
    angles = np.random.uniform(0, 2 * np.pi, n_cams)
    x = xy_radius * np.cos(angles)
    y = xy_radius * np.sin(angles)
    cams = []
    for i in range(n_cams):
        pose = sapien_utils.look_at(eye=[x[i], y[i], z_offset], target=target)
        cams.append(
            CameraConfig(
                uid=f"camera_pose{i+1}",
                pose=pose,
                width=width,
                height=height,
                fov=REALSENSE_DEPTH_FOV_VERTICAL_RAD,
                near=0.01,
                far=100,
                shader_pack=SHADER,
            )
        )

    return cams


def get_human_render_camera_config(
    eye: tuple[float, float, float], target: tuple[float, float, float], shader: str | None = None
):
    """Configures the human render camera. Shader options:
    - minimal: The fastest shader with minimal GPU memory usage. Note that the background will always be black (normally it is the color of the ambient light)
    - default: A balance between speed and texture availability
    - rt: A shader optimized for photo-realistic rendering via ray-tracing
    - rt-med: Same as rt but runs faster with slightly lower quality
    - rt-fast: Same as rt-med but runs faster with slightly lower quality
    -> https://maniskill.readthedocs.io/en/latest/user_guide/concepts/sensors.html#shaders-and-textures
    """
    SHADER = "default" if shader is None else shader
    pose = sapien_utils.look_at(eye=eye, target=target)
    return CameraConfig(
        "render_camera", pose=pose, width=1264, height=1264, fov=np.pi / 3, near=0.01, far=100, shader_pack=SHADER
    )
