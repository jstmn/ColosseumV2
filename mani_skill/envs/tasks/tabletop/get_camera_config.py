import numpy as np
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
import sapien

REALSENSE_DEPTH_FOV_VERTICAL_RAD = 58.0 * np.pi / 180
REALSENSE_DEPTH_FOV_HORIZONTAL_RAD = 87.0 * np.pi / 180

SHADER = "default"

def get_camera_configs(xy_offset, z_offset, target: tuple[float, float, float], camera_width: int, camera_height: int, is_act: bool=False) -> list[CameraConfig]:
    pose_center = sapien_utils.look_at(eye=[xy_offset, 0,  z_offset], target=target)
    pose_left = sapien_utils.look_at(eye=[0.0, -xy_offset, z_offset], target=target)
    pose_right = sapien_utils.look_at(eye=[0.0, xy_offset, z_offset], target=target)
    return [
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
        )]

def get_act_camera_configs(xy_offset, z_offset, target: tuple[float, float, float], robot) -> list[CameraConfig]:
    pose_center = sapien_utils.look_at(eye=[xy_offset, 0,  z_offset], target=target)
    pose_left = sapien_utils.look_at(eye=[0.0, -xy_offset, z_offset], target=target)
    camera_width = 640
    camera_height = 480
    LOGITEK_FOV = REALSENSE_DEPTH_FOV_VERTICAL_RAD # 78.0 * np.pi / 180
    return [
        CameraConfig(
            uid="camera_center",
            pose=pose_center,
            width=camera_width,
            height=camera_height,
            fov=LOGITEK_FOV,
            near=0.01,
            far=100,
            shader_pack=SHADER,
        ),
        CameraConfig(
            uid="camera_left",
            pose=pose_left,
            width=camera_width,
            height=camera_height,
            fov=LOGITEK_FOV,
            near=0.01,
            far=100,
            shader_pack=SHADER,
        ),
        CameraConfig(
            uid="camera_wrist",
            pose=sapien.Pose(p=[0.1, 0.0, -0.1], q=[0.0,0.5,0.0,0.866]),
            mount=robot.links_map["panda_hand_tcp"],
            width=camera_width,
            height=camera_height,
            fov=LOGITEK_FOV,
            near=0.01,
            far=100,
            shader_pack=SHADER,
        )]

def get_human_render_camera_config(eye: tuple[float, float, float], target: tuple[float, float, float]):
    """ Configures the human render camera. Shader options:
        - minimal: The fastest shader with minimal GPU memory usage. Note that the background will always be black (normally it is the color of the ambient light)
        - default: A balance between speed and texture availability
        - rt: A shader optimized for photo-realistic rendering via ray-tracing
        - rt-med: Same as rt but runs faster with slightly lower quality
        - rt-fast: Same as rt-med but runs faster with slightly lower quality
        -> https://maniskill.readthedocs.io/en/latest/user_guide/concepts/sensors.html#shaders-and-textures
    """
    SHADER = "default"
    pose = sapien_utils.look_at(eye=eye, target=target)
    return CameraConfig("render_camera", pose=pose, width=1264, height=1264, fov=np.pi / 3, near=0.01, far=100, shader_pack=SHADER)