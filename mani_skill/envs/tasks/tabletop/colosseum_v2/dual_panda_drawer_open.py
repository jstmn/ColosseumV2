import gymnasium as gym
import numpy as np
import sapien.core as sapien
from mani_skill.utils.registration import register_env
from mani_skill.agents.robots.panda.dual_panda_wristcam import DualPandaWristCam
from mani_skill.utils.building import articulations
import torch
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.envs.tasks.tabletop.colosseum_v2.colosseum_v2_core import ColosseumV2Env, DisabledVariationFactors, PlacementRegion


@register_env("DualArmDrawerOpen-v1", max_episode_steps=1000, asset_download_ids=["partnet_mobility_cabinet"])
class DualArmDrawerOpenEnv(ColosseumV2Env):
    """
    Two hold the handles of drawer and open the doors.
    Uses PartNet-Mobility dataset (ID 1005).
    """
    cube_half_size = 0.02
    # Explicitly tell ManiSkill to use the DualPanda agent
    SUPPORTED_ROBOTS = ["dual_panda_wristcam"]
    agent: DualPandaWristCam # Type hinting for IDE support

    DISABLED_VARIATION_FACTORS = DisabledVariationFactors(
        MO_color=True,
        MO_texture=True,
        MO_size=True,
        RO_color=True,
        RO_texture=True,
        RO_size=True,
    )

    def __init__(self, *args, robot_uids="dual_panda_wristcam", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose1 = sapien_utils.look_at(eye=[-0.3, 0.5, 1.0+0.83], target=[-0.1, 0, 0.2+0.83])
        pose2 = sapien_utils.look_at(eye=[-0.3, -0.5, 1.0+0.83], target=[-0.1, 0, 0.2+0.83])
        return self.update_camera_configs([
            CameraConfig(
                "external1_camera",
                pose=pose1,
                width=224,
                height=224,
                fov=np.pi / 3,
                near=0.01,
                far=10,
            ),
            CameraConfig(
                "external2_camera",
                pose=pose2,
                width=224,
                height=224,
                fov=np.pi / 3,
                near=0.01,
                far=10,
            )
        ])

    @property
    def _default_human_render_camera_configs(self):
        """Configure camera for rendering videos and visualization"""
        pose = sapien_utils.look_at(eye=[-0.6, -0.2, 0.4+0.83], target=[0.1, 0, 0.1+0.83])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)
        

    def _load_scene(self, options: dict):
        def get_builder_fn():
            # Load PartNet-Mobility Drawer (ID 1005 is a standard table with drawer)
            model_id = "1005"
            builder = articulations.get_articulation_builder(
                self.scene, f"partnet-mobility:{model_id}"
            )
            # Set initial pose to match the previous drawer's location
            # Position: [0.25, 0, 1.256] (0.456 + 0.8)
            # Orientation: 90 degree rotation around X (q=[0.7071, 0, 0, -0.7071])
            builder.initial_pose = sapien.Pose(
                p=[0.25, 0, 0.456+0.8], 
                q=[0.7071, 0, 0, -0.7071]
            )
            return builder

        self.open_cabinet = self.add_asset_to_scene(get_builder_fn, name="drawer", physics_type="articulation", object_type="MO")
        assert isinstance(self.open_cabinet, Articulation), "open_cabinet must be an articulation"

        self.load_scene_hook(manipulation_objects=[self.open_cabinet])

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            # xyz = torch.zeros((b, 3), device=self.device)
            # xyz[..., :2] = -torch.rand((b, 2), device=self.device) * 0.2 + 0.1
            # xyz[..., 0] += 0.2
            xyz = torch.zeros((b, 3), device=self.device)
            drawer_region = self.update_placement_region(PlacementRegion.from_center_and_width(center=(0.2, 0.0), width=(0.2, 0.2)))
            xyz[..., :2] = drawer_region.sample_xy(b, device=self.device)
            xyz[..., 2] = 0.456 + 0.8
            theta_by_2 = torch.rand(b, device=self.device) * np.pi / 16 - np.pi / 32
            dof_tensor = self.open_cabinet.dof
            if isinstance(dof_tensor, torch.Tensor):
                dof = int(dof_tensor.flatten()[0].cpu().item())
            else:
                dof = int(dof_tensor)

            # Vectorized pose setting
            cos_vals = torch.cos(theta_by_2)
            sin_vals = torch.sin(theta_by_2)
            qs = torch.zeros((b, 4), device=self.device)
            qs[:, 0] = cos_vals
            qs[:, 3] = sin_vals
            cabinet_pose = Pose.create_from_pq(p=xyz, q=qs)
            self.open_cabinet.set_pose(cabinet_pose)

            # Close the drawer (reset joint positions to 0)
            self.open_cabinet.set_qpos(torch.zeros((b, dof), device=self.device))
            self.open_cabinet.set_qvel(torch.zeros((b, dof), device=self.device))

            self.initialize_episode_hook(env_idx)
        self._initialize_agent()

    def _initialize_agent(self):
        # Reset the robot to a neutral position
        qpos = np.array([1.683, 1.357, 0.284, 0.393, -0.103, 0.249, -1.529, -2.074, -1.497, 1.647, 1.409, 1.758, -2.106, -0.114, 0.04, 0.04, 0.04, 0.04])
        self.agent.reset(qpos)

    def evaluate(self):

        active_joints = self.open_cabinet.active_joints
        assert len(active_joints) == 4, "There should be 4 active joints, got %d" % len(active_joints)

        qpos_1 = active_joints[0].qpos
        qpos_2 = active_joints[2].qpos
        qpos_min_1, qpos_max_1 = active_joints[0].limits[..., 0], active_joints[0].limits[..., 1]
        qpos_min_2, qpos_max_2 = active_joints[1].limits[..., 0], active_joints[1].limits[..., 1]

        drawer_open_pct_1 = (qpos_1 - qpos_min_1) / (qpos_max_1 - qpos_min_1 + 1e-6)
        drawer_open_pct_2 = (qpos_2 - qpos_min_2) / (qpos_max_2 - qpos_min_2 + 1e-6)
        is_drawer_open = torch.logical_and(drawer_open_pct_1 > 0.20, drawer_open_pct_2 > 0.20)
        success = is_drawer_open
        return {"is_drawer_open_1": drawer_open_pct_1 > 0.30, "is_drawer_open_2": drawer_open_pct_2 > 0.30, "success": success}