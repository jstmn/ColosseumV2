import numpy as np
import sapien

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

from .dual_panda import DualPanda


@register_agent()
class DualPandaWristCam(DualPanda):
    """Panda arm robot with the real sense camera attached to gripper"""

    uid = "dual_panda_wristcam"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/dual_panda_table_wristcam.urdf"

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="panda1_hand_camera",
                pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["panda1_camera_link"],
            ),
            CameraConfig(
                uid="panda2_hand_camera",
                pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["panda2_camera_link"],
            )
        ]


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)