from typing import Any, Dict, List, Optional, Union

import numpy as np
import sapien
import torch
import trimesh

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors, articulations
from mani_skill.utils.geometry.geometry import transform_points
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Articulation, Link, Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.scene_builder.table import TableSceneBuilder

CABINET_COLLISION_BIT = 29


@register_env(
    "OpenCabinet-v1",
    asset_download_ids=["partnet_mobility_cabinet"],
    max_episode_steps=100,
)
class OpenCabinetEnv(BaseEnv):
    """
    **Task Description:**
    Open a cabinet door using the Panda robot.

    **Randomizations:**
    - Cabinet position is slightly randomized on the table.

    **Success Conditions:**
    - Cabinet door is opened beyond the target threshold.
    - Door is static (not moving).
    """

    SUPPORTED_ROBOTS = ["panda"]
    agent: Union[Panda]
    handle_types = ["revolute", "revolute_unwrapped"]  # Cabinet doors use revolute joints

    TRAIN_JSON = (
        PACKAGE_ASSET_DIR / "partnet_mobility/meta/info_cabinet_door_train.json"
    )

    CABINET_X_LIMS = [0.15, 0.25]
    CABINET_Y_LIMS = [-0.05, 0.05]

    min_open_frac = 0.5

    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        # Cabinet door model - 1027 has revolute door joints (from info_cabinet_door_train.json)
        self._model_id = 1027

        super().__init__(
            *args,
            robot_uids=robot_uids,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq=100,
            control_freq=20,
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**23, max_rigid_patch_count=2**17
            ),
        )

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.5, -0.5, 0.8], target=[0.0, 0.0, 0.3])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.3])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        sapien.set_log_level("off")
        self._load_cabinets(self.handle_types)
        sapien.set_log_level("warn")
        self._hidden_objects.append(self.handle_link_goal)

    def _load_cabinets(self, joint_types: List[str]):
        link_ids = [0]

        self._cabinets = []
        handle_links: List[List[Link]] = []
        handle_links_meshes: List[List[trimesh.Trimesh]] = []

        cabinet_builder = articulations.get_articulation_builder(
            self.scene, f"partnet-mobility:{self._model_id}"
        )
        cabinet_builder.initial_pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
        cabinet = cabinet_builder.build(name=f"cabinet-{self._model_id}")
        self.remove_from_state_dict_registry(cabinet)

        for link in cabinet.links:
            link.set_collision_group_bit(
                group=2, bit_idx=CABINET_COLLISION_BIT, bit=1
            )
        self._cabinets.append(cabinet)
        handle_links.append([])
        handle_links_meshes.append([])

        for link, joint in zip(cabinet.links, cabinet.joints):
            # Check for revolute joints (doors)
            if joint.type[0] in joint_types:
                handle_links[-1].append(link)
                meshes = link.generate_mesh(
                    filter=lambda _, render_shape: "handle" in render_shape.name,
                    mesh_name="handle",
                )
                if meshes:
                    handle_links_meshes[-1].append(meshes[0])
                else:
                    # Fallback: create a dummy mesh at origin
                    handle_links_meshes[-1].append(None)

        # If no revolute joints found, raise error
        if not handle_links[0]:
            raise ValueError(
                f"No {joint_types} joints found in cabinet model {self._model_id}. "
                f"Available joint types: {[j.type[0] for j in cabinet.joints]}"
            )

        self.cabinet = Articulation.merge(self._cabinets, name="cabinet")
        self.add_to_state_dict_registry(self.cabinet)

        # Merge handle links
        self.handle_link = Link.merge(
            [links[link_ids[i] % len(links)] for i, links in enumerate(handle_links)],
            name="handle_link",
        )

        # Store handle position relative to link
        self.handle_link_pos = common.to_tensor(
            np.array(
                [
                    meshes[link_ids[i] % len(meshes)].bounding_box.center_mass
                    if meshes[link_ids[i] % len(meshes)] is not None
                    else [0, 0, 0]
                    for i, meshes in enumerate(handle_links_meshes)
                ]
            ),
            device=self.device,
        )

        self.handle_link_goal = actors.build_sphere(
            self.scene,
            radius=0.02,
            color=[0, 1, 0, 1],
            name="handle_link_goal",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
        )

    def _after_reconfigure(self, options):
        self.cabinet_zs = []
        for cabinet in self._cabinets:
            collision_mesh = cabinet.get_first_collision_mesh()
            self.cabinet_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.cabinet_zs = common.to_tensor(self.cabinet_zs, device=self.device)

        target_qlimits = self.handle_link.joint.limits
        qmin, qmax = target_qlimits[..., 0], target_qlimits[..., 1]
        self.target_qpos = qmin + (qmax - qmin) * self.min_open_frac

    def handle_link_positions(self, env_idx: Optional[torch.Tensor] = None):
        if env_idx is None:
            return transform_points(
                self.handle_link.pose.to_transformation_matrix().clone(),
                common.to_tensor(self.handle_link_pos, device=self.device),
            )
        return transform_points(
            self.handle_link.pose[env_idx].to_transformation_matrix().clone(),
            common.to_tensor(self.handle_link_pos[env_idx], device=self.device),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            xy = torch.zeros((b, 3))
            cabinet_x_range = self.CABINET_X_LIMS[1] - self.CABINET_X_LIMS[0]
            cabinet_y_range = self.CABINET_Y_LIMS[1] - self.CABINET_Y_LIMS[0]
            xy[:, 0] = torch.rand(b) * cabinet_x_range + self.CABINET_X_LIMS[0]
            xy[:, 1] = torch.rand(b) * cabinet_y_range + self.CABINET_Y_LIMS[0]
            xy[:, 2] = self.cabinet_zs[env_idx]

            self.cabinet.set_pose(Pose.create_from_pq(p=xy))

            # Initialize robot
            qpos_0 = np.array(
                [
                    -0.13595445,
                    -1.2611351,
                    0.24094589,
                    -2.9000182,
                    2.5728698,
                    3.0259767,
                    0.029944034,
                    0.039999813,
                    0.03999985,
                ]
            )
            self.table_scene.initialize(env_idx, table_z_rotation_angle=np.pi, qpos_0=qpos_0)

            # Close all cabinet doors
            qlimits = self.cabinet.get_qlimits()
            self.cabinet.set_qpos(qlimits[env_idx, :, 0])
            self.cabinet.set_qvel(self.cabinet.qpos[env_idx] * 0)

            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene.px.step()
                self.scene._gpu_fetch_all()

            self.handle_link_goal.set_pose(
                Pose.create_from_pq(p=self.handle_link_positions(env_idx))
            )

    def _after_control_step(self):
        if self.gpu_sim_enabled:
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        self.handle_link_goal.set_pose(
            Pose.create_from_pq(p=self.handle_link_positions())
        )
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()

    def evaluate(self):
        open_enough = self.handle_link.joint.qpos >= self.target_qpos
        if open_enough.ndim > 1:
            open_enough = open_enough.squeeze(-1)

        link_is_static = (
            torch.linalg.norm(self.handle_link.angular_velocity, axis=1) <= 1
        ) & (torch.linalg.norm(self.handle_link.linear_velocity, axis=1) <= 0.1)

        return {
            "success": open_enough & link_is_static,
            "handle_link_pos": self.handle_link_positions(),
            "open_enough": open_enough,
            "link_is_static": link_is_static,
        }

    def _get_obs_extra(self, info: Dict):
        obs = {
            "tcp_pose": self.agent.tcp_pose.raw_pose,
            "handle_pos": self.handle_link_positions(),
        }
        if "state" in self.obs_mode:
            obs["cabinet_qpos"] = self.cabinet.qpos
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_pos = self.agent.tcp_pose.p
        handle_pos = self.handle_link_positions()

        # Distance to handle
        tcp_to_handle = torch.linalg.norm(tcp_pos - handle_pos, axis=1)
        reaching_reward = 1 - torch.tanh(5.0 * tcp_to_handle)

        # Opening reward
        qpos = self.handle_link.joint.qpos
        if qpos.ndim > 1:
            qpos = qpos.squeeze(-1)
        qlimits = self.handle_link.joint.limits
        qmin, qmax = qlimits[..., 0].squeeze(-1), qlimits[..., 1].squeeze(-1)
        open_frac = (qpos - qmin) / (qmax - qmin + 1e-6)
        open_reward = open_frac

        # Success bonus
        success = info["success"].float()

        reward = reaching_reward + open_reward * 2 + success * 5
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs, action, info) / 8.0
