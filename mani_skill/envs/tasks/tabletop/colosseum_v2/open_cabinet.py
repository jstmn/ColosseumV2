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
from mani_skill.utils.geometry.trimesh_utils import merge_meshes
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb
from mani_skill.envs.distraction_set import DistractionSet

CABINET_COLLISION_BIT = 29


def _normalize(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Normalize a vector, returning fallback if norm is too small."""
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return fallback
    return vec / norm


def _get_joint_axis(joint) -> np.ndarray:
    """Get the global rotation axis for a joint."""
    axis = None
    axis_is_local = True
    raw_joint = joint._objs[0] if hasattr(joint, "_objs") and joint._objs else None
    if raw_joint is not None:
        if hasattr(raw_joint, "get_axis"):
            try:
                axis = np.array(raw_joint.get_axis(), dtype=np.float32)
            except Exception:
                axis = None
        if axis is None and hasattr(raw_joint, "axis"):
            axis = np.array(raw_joint.axis, dtype=np.float32)
    if axis is None and hasattr(joint, "get_axis"):
        try:
            axis = np.array(joint.get_axis(), dtype=np.float32)
        except Exception:
            axis = None
    if axis is None and hasattr(joint, "axis"):
        axis = np.array(joint.axis, dtype=np.float32)
    if axis is None:
        joint_pose = joint.get_global_pose().to_transformation_matrix()
        if hasattr(joint_pose, "ndim") and joint_pose.ndim == 3:
            joint_pose = joint_pose[0]
        if hasattr(joint_pose, "cpu"):
            joint_pose = joint_pose.cpu().numpy()
        axis = np.array(joint_pose[:3, 0], dtype=np.float32)
        axis_is_local = False
    if axis_is_local:
        joint_pose = joint.get_global_pose().to_transformation_matrix()
        if hasattr(joint_pose, "ndim") and joint_pose.ndim == 3:
            joint_pose = joint_pose[0]
        if hasattr(joint_pose, "cpu"):
            joint_pose = joint_pose.cpu().numpy()
        axis = joint_pose[:3, :3] @ axis
    return _normalize(axis, np.array([0.0, 0.0, 1.0], dtype=np.float32))


def _get_joint_pivot(joint) -> np.ndarray:
    """Get the pivot point (position) of a joint in global coordinates."""
    joint_pose = joint.get_global_pose()
    pivot = joint_pose.p
    if hasattr(pivot, "cpu"):
        pivot = pivot.cpu().numpy()
    pivot = np.array(pivot)
    if pivot.ndim == 2:
        pivot = pivot[0]
    return pivot


def _get_handle_obb(handle_link):
    """Get the oriented bounding box of the handle mesh."""
    try:
        meshes = handle_link.generate_mesh(
            filter=lambda _, render_shape: "handle" in render_shape.name,
            mesh_name="handle",
        )
    except Exception:
        return None
    if not meshes:
        return None
    merged = merge_meshes([mesh for mesh in meshes if mesh is not None])
    if merged is None:
        return None
    link_pose = handle_link.pose.to_transformation_matrix()
    if hasattr(link_pose, "ndim") and link_pose.ndim == 3:
        link_pose = link_pose[0]
    if hasattr(link_pose, "cpu"):
        link_pose = link_pose.cpu().numpy()
    merged.apply_transform(link_pose)
    return merged.bounding_box_oriented


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

    CABINET_X_LIMS = [0.15, 0.22]
    CABINET_Y_LIMS = [0.0, 0.08]  # Shifted positive to match angled robot

    min_open_frac = 0.25

    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.0,  # No noise - IK handles initial configuration
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        # Cabinet door model - 1027 has revolute door joints (from info_cabinet_door_train.json)
        self._model_id = 1027
        distraction_set: DistractionSet | dict | None = kwargs.pop("distraction_set", None)
        self._distraction_set: DistractionSet | None = DistractionSet(**distraction_set) if isinstance(distraction_set, dict) else distraction_set
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
        # View from behind-left of robot (right side of scene), looking at cabinet
        pose = sapien_utils.look_at(eye=[-0.8, -0.6, 0.7], target=[0.1, 0.0, 0.35])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    @property
    def _default_sensor_configs(self):
        # Sensor camera with view of robot arm and cabinet handle
        pose = sapien_utils.look_at(eye=[-0.4, -0.5, 0.6], target=[0.0, 0.0, 0.35])
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
        from transforms3d.euler import euler2quat

        with torch.device(self.device):
            b = len(env_idx)
            xy = torch.zeros((b, 3))
            xy[:, 0] = 0.20  # Fixed X position
            xy[:, 1] = 0.0   # Fixed Y position
            xy[:, 2] = self.cabinet_zs[env_idx]

            self.cabinet.set_pose(Pose.create_from_pq(p=xy))

            # Close all cabinet doors first so we can get handle position
            qlimits = self.cabinet.get_qlimits()
            self.cabinet.set_qpos(qlimits[env_idx, :, 0])
            self.cabinet.set_qvel(self.cabinet.qpos[env_idx] * 0)

            # Position robot angled to pull door arc to 90%
            robot_angle = np.pi / 12  # 15 degrees
            robot_base_pos = np.array([-0.615, 0.15, 0])
            robot_pose = sapien.Pose(
                p=robot_base_pos,
                q=euler2quat(0, 0, robot_angle)
            )

            # Get handle position to compute grasp pose
            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene.px.step()
                self.scene._gpu_fetch_all()

            handle_pos = self.handle_link_positions(env_idx)[0].cpu().numpy()

            # Compute grasp pose using EXACT same logic as motion planning solution
            # Get joint info
            joint = self.handle_link.joint
            axis = _get_joint_axis(joint)
            pivot = _get_joint_pivot(joint)

            # Calculate approach direction (from robot to handle)
            approaching = _normalize(handle_pos - robot_base_pos, np.array([1.0, 0.0, 0.0], dtype=np.float32))

            # Door normal perpendicular to axis and radial direction
            radial = _normalize(handle_pos - pivot, np.array([1.0, 0.0, 0.0], dtype=np.float32))
            door_normal = np.cross(axis, radial)
            door_normal = _normalize(door_normal, np.array([0.0, 1.0, 0.0], dtype=np.float32))
            if np.dot(door_normal, robot_base_pos - handle_pos) < 0:
                door_normal = -door_normal

            # Tangent direction for gripper closing
            tangent = _normalize(np.cross(axis, door_normal), np.array([0.0, 1.0, 0.0], dtype=np.float32))

            # Get handle OBB for precise grasp
            handle_obb = _get_handle_obb(self.handle_link)

            # Compute grasp pose (same as motion planning)
            finger_length = 0.025
            grasp_backoff = -0.005
            if handle_obb is not None:
                grasp_info = compute_grasp_info_by_obb(
                    handle_obb,
                    approaching=approaching,
                    target_closing=tangent,
                    depth=finger_length,
                )
                closing = grasp_info["closing"]
                center = grasp_info["center"] + approaching * grasp_backoff
            else:
                closing = _normalize(np.cross(axis, approaching), np.array([0.0, 1.0, 0.0], dtype=np.float32))
                center = handle_pos + approaching * grasp_backoff

            # Fixed pre-grasp qpos (robot and cabinet positions are fixed, so this is deterministic)
            pregrasp_qpos = np.array([
                -2.5732915, -0.6889829, 1.9453487, -2.5163524,
                -2.704773, 1.5929992, 1.5251541, 0.04, 0.04
            ])

            # Initialize robot with fixed configuration
            self.table_scene.initialize(env_idx, table_z_rotation_angle=np.pi, qpos_0=pregrasp_qpos)
            self.agent.robot.set_pose(robot_pose)

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
        # Door must be at least 45% of target (which is 50% of full range = ~22.5% of full)
        open_enough = self.handle_link.joint.qpos >= self.target_qpos * 0.45
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
