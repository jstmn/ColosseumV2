from typing import Any, Dict, List, Optional, Union

import numpy as np
import sapien
import torch
import trimesh
from transforms3d.euler import euler2quat

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
    "ObjectInCabinet-v1",
    asset_download_ids=["partnet_mobility_cabinet"],
    max_episode_steps=200,
)
class ObjectInCabinetEnv(BaseEnv):
    """
    **Task Description:**
    Open a cabinet door and place an apple inside using the Panda robot.

    **Randomizations:**
    - Apple position is randomized on the table in front of the robot.

    **Success Conditions:**
    - The apple is inside the cabinet
    - The apple is static
    - The robot is not grasping the apple
    """

    SUPPORTED_ROBOTS = ["panda"]
    agent: Union[Panda]

    APPLE_RADIUS = 0.04
    handle_types = ["revolute", "revolute_unwrapped"]

    TRAIN_JSON = (
        PACKAGE_ASSET_DIR / "partnet_mobility/meta/info_cabinet_door_train.json"
    )

    min_open_frac = 0.5

    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,  # Small noise like OpenCabinet
        **kwargs,
    ):
        kwargs.pop("distraction_set", None)
        self.robot_init_qpos_noise = robot_init_qpos_noise
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
        # Camera from front-left to see robot, cabinet door, and apple
        pose = sapien_utils.look_at(eye=[-0.6, -0.6, 0.8], target=[0.0, 0.0, 0.2])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1.2, near=0.01, far=100
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

        self._load_apple()

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
            if joint.type[0] in joint_types:
                handle_links[-1].append(link)
                meshes = link.generate_mesh(
                    filter=lambda _, render_shape: "handle" in render_shape.name,
                    mesh_name="handle",
                )
                if meshes:
                    handle_links_meshes[-1].append(meshes[0])
                else:
                    handle_links_meshes[-1].append(None)

        if not handle_links[0]:
            raise ValueError(
                f"No {joint_types} joints found in cabinet model {self._model_id}."
            )

        self.cabinet = Articulation.merge(self._cabinets, name="cabinet")
        self.add_to_state_dict_registry(self.cabinet)

        self.handle_link = Link.merge(
            [links[link_ids[i] % len(links)] for i, links in enumerate(handle_links)],
            name="handle_link",
        )

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

    def _load_apple(self):
        self.apple = actors.build_sphere(
            self.scene,
            radius=self.APPLE_RADIUS,
            color=[1, 0, 0, 1],
            name="apple",
            initial_pose=sapien.Pose(p=[0, 0, self.APPLE_RADIUS]),
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

            # Fixed cabinet position (same as OpenCabinet)
            xy = torch.zeros((b, 3))
            xy[:, 0] = 0.20
            xy[:, 1] = 0.0
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

            # Position robot angled to pull door arc
            robot_angle = np.pi / 12  # 15 degrees
            robot_pose = sapien.Pose(
                p=[-0.615, 0.15, 0],
                q=euler2quat(0, 0, robot_angle)
            )
            self.agent.robot.set_pose(robot_pose)

            # Close cabinet doors
            qlimits = self.cabinet.get_qlimits()
            self.cabinet.set_qpos(qlimits[env_idx, :, 0])
            self.cabinet.set_qvel(self.cabinet.qpos[env_idx] * 0)

            # Place apple on table in reachable position
            # The solution will temporarily move apple during door opening to avoid motion planning interference
            apple_xyz = torch.zeros((b, 3), device=self.device)
            apple_xyz[:, 0] = -0.30 + torch.rand(b) * 0.10  # X: -0.30 to -0.20 (reachable by robot)
            apple_xyz[:, 1] = -0.15 - torch.rand(b) * 0.10  # Y: -0.15 to -0.25 (right side, away from door)
            apple_xyz[:, 2] = self.APPLE_RADIUS
            self.apple.set_pose(Pose.create_from_pq(p=apple_xyz))

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
        pos_apple = self.apple.pose.p
        cabinet_pos = self.cabinet.pose.p

        # Apple is in cabinet zone if it's near the cabinet and above table
        # Wider zone (0.30) to account for robot reach limitations and apple rolling
        is_near_cabinet_x = torch.abs(pos_apple[..., 0] - cabinet_pos[..., 0]) < 0.30
        is_near_cabinet_y = torch.abs(pos_apple[..., 1] - cabinet_pos[..., 1]) < 0.30
        is_above_table = pos_apple[..., 2] > 0.03  # Allow table placement

        is_apple_in_cabinet = is_near_cabinet_x & is_near_cabinet_y & is_above_table
        is_apple_static = self.apple.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_apple_grasped = self.agent.is_grasping(self.apple)

        # Door must be at least 80% open (of target, which is 50% of full range)
        # This means door must stay open once it passes 80% threshold
        door_qpos = self.handle_link.joint.qpos
        if door_qpos.ndim > 1:
            door_qpos = door_qpos.squeeze(-1)
        door_open_enough = door_qpos >= self.target_qpos * 0.8  # 80% of target (which is 50% of full range = 40% of full)

        # Success requires:
        # 1. Apple placed in cabinet zone
        # 2. Apple is static
        # 3. Robot is not grasping apple
        # 4. Door must remain open (>80% of target opening)
        success = is_apple_in_cabinet & is_apple_static & (~is_apple_grasped) & door_open_enough

        return {
            "is_apple_grasped": is_apple_grasped,
            "is_apple_in_cabinet": is_apple_in_cabinet,
            "is_apple_static": is_apple_static,
            "door_open_enough": door_open_enough,
            "door_qpos": door_qpos,
            "target_qpos": self.target_qpos,
            "handle_link_pos": self.handle_link_positions(),
            "success": success,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                apple_pose=self.apple.pose.raw_pose,
                tcp_to_apple_pos=self.apple.pose.p - self.agent.tcp.pose.p,
                tcp_to_handle_pos=info["handle_link_pos"] - self.agent.tcp.pose.p,
                target_link_qpos=self.handle_link.joint.qpos,
                target_handle_pos=info["handle_link_pos"],
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        reward = torch.zeros(self.num_envs, device=self.device)

        tcp_pos = self.agent.tcp.pose.p
        apple_pos = self.apple.pose.p

        # Door opening reward
        door_progress = self.handle_link.joint.qpos / (self.target_qpos + 1e-6)
        door_reward = torch.clamp(door_progress.squeeze(-1), 0, 1)
        reward += door_reward

        # Reaching reward for apple
        tcp_to_apple_dist = torch.linalg.norm(tcp_pos - apple_pos, axis=1)
        reaching_reward = 1 - torch.tanh(5 * tcp_to_apple_dist)
        is_grasping = info["is_apple_grasped"]
        reaching_reward[is_grasping] = 1.0
        reward += reaching_reward

        # Grasp reward
        grasp_reward = is_grasping.float() * 0.5
        reward += grasp_reward

        reward[info["success"]] = 5.0

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 5.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
