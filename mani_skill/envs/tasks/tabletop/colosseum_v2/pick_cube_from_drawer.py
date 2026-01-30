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
from mani_skill.envs.tasks.tabletop.colosseum_v2.distraction_set import DistractionSet

CABINET_COLLISION_BIT = 29


@register_env(
    "PickCubeFromDrawer-v1",
    asset_download_ids=["partnet_mobility_cabinet"],
    max_episode_steps=200,
)
class PickCubeFromDrawerEnv(BaseEnv):
    """
    **Task Description:**
    Open a drawer and pick up a cube from inside it.

    **Success Conditions:**
    - Cube is in the robot's gripper and in the air.
    """

    SUPPORTED_ROBOTS = ["panda"]
    agent: Union[Panda]
    handle_types = ["prismatic"]  # Drawer joints

    LIFT_HEIGHT = 0.15  # Cube must be lifted above this absolute z height
    CUBE_HALF_SIZE = 0.035

    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.0,
        **kwargs,
    ):
        distraction_set: DistractionSet | dict | None = kwargs.pop("distraction_set", None)
        self._distraction_set: DistractionSet | None = DistractionSet(**distraction_set) if isinstance(distraction_set, dict) else distraction_set
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self._model_id = 45427  # Cabinet with drawers

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
        pose = sapien_utils.look_at(eye=[0.5, 0.9, 0.6], target=[0.0, -0.2, 0.3])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    @property
    def _default_sensor_configs(self):
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
        # Robot positioned perpendicular to cabinet (at -Y, facing +Y)
        super()._load_agent(options, sapien.Pose(p=[0, -0.615, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        sapien.set_log_level("off")
        self._load_cabinets(self.handle_types)
        sapien.set_log_level("warn")
        self._hidden_objects.append(self.handle_link_goal)

        self._load_cube()

    def _load_cube(self):
        # Build cube with high friction so it stays in drawer
        builder = self.scene.create_actor_builder()
        material = sapien.physx.PhysxMaterial(
            static_friction=2.0,
            dynamic_friction=2.0,
            restitution=0.0,
        )
        builder.add_box_collision(
            half_size=[self.CUBE_HALF_SIZE] * 3,
            material=material,
        )
        builder.add_box_visual(
            half_size=[self.CUBE_HALF_SIZE] * 3,
            material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 1]),
        )
        builder.set_initial_pose(sapien.Pose(p=[0, 0, self.CUBE_HALF_SIZE]))
        self.cube = builder.build(name="cube")

    def _load_cabinets(self, joint_types: List[str]):
        # Use the bottom drawer
        link_ids = [2]

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

        handle_pos_list = [
            meshes[link_ids[i] % len(meshes)].bounding_box.center_mass
            if meshes[link_ids[i] % len(meshes)] is not None
            else [0, 0, 0]
            for i, meshes in enumerate(handle_links_meshes)
        ]
        # Expand to num_envs (single model replicated across all parallel envs)
        self.handle_link_pos = common.to_tensor(
            np.array(handle_pos_list * self.num_envs),
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

        # Add drive properties to allow drawer to be driven open
        for cabinet in self._cabinets:
            for joint in cabinet.joints:
                if joint.type[0] == "prismatic":
                    joint.set_drive_properties(stiffness=100.0, damping=50.0)

    def _after_reconfigure(self, options):
        cabinet_zs = []
        for cabinet in self._cabinets:
            collision_mesh = cabinet.get_first_collision_mesh()
            cabinet_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        # Expand to num_envs (single model replicated across all parallel envs)
        self.cabinet_zs = common.to_tensor(
            cabinet_zs * self.num_envs, device=self.device
        )

        target_qlimits = self.handle_link.joint.limits
        qmin, qmax = target_qlimits[..., 0], target_qlimits[..., 1]
        self.target_qpos = qmin + (qmax - qmin) * 0.5

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

            # Position cabinet so robot approaches perpendicular
            # Robot is at Y=-0.615, cabinet rotated so drawer faces -X
            cabinet_pos = torch.zeros((b, 3))
            cabinet_pos[:, 0] = -0.20    # X position (20cm to right of robot)
            cabinet_pos[:, 1] = -0.55   # Y position (20cm towards robot)
            cabinet_pos[:, 2] = self.cabinet_zs[env_idx]

            # Rotate 90° clockwise around Z so drawer faces -X
            cabinet_quat = torch.zeros((b, 4))
            cabinet_quat[:, 0] = 0.7071  # cos(-45°)
            cabinet_quat[:, 3] = -0.7071  # sin(-45°)

            self.cabinet.set_pose(Pose.create_from_pq(p=cabinet_pos, q=cabinet_quat))

            # Close all drawers
            qlimits = self.cabinet.get_qlimits()
            self.cabinet.set_qpos(qlimits[env_idx, :, 0])
            self.cabinet.set_qvel(self.cabinet.qpos[env_idx] * 0)

            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene.px.step()
                self.scene._gpu_fetch_all()

            # Initialize robot with default pose
            self.table_scene.initialize(env_idx)

            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene.px.step()
                self.scene._gpu_fetch_all()

            self.handle_link_goal.set_pose(
                Pose.create_from_pq(p=self.handle_link_positions(env_idx))
            )

            # Place cube inside the drawer
            self._place_cube_in_drawer(env_idx)

    def _place_cube_in_drawer(self, env_idx: torch.Tensor):
        """Place cube in the drawer with random position offset."""
        with torch.device(self.device):
            b = len(env_idx)

            handle_pos = self.handle_link_positions(env_idx)

            # Random offsets for cube position within drawer
            # X offset: +/- 0.03m (side to side in drawer)
            # Y offset: -0.08 to -0.16m (depth into drawer)
            x_offset = (torch.rand(b) - 0.5) * 0.06  # [-0.03, 0.03]
            y_offset = -0.08 - torch.rand(b) * 0.08  # [-0.08, -0.16]

            cube_xyz = torch.zeros((b, 3))
            # Place cube in drawer with random offset
            cube_xyz[:, 0] = handle_pos[:, 0] + x_offset
            cube_xyz[:, 1] = handle_pos[:, 1] + y_offset
            cube_xyz[:, 2] = handle_pos[:, 2] + self.CUBE_HALF_SIZE

            self.cube.set_pose(Pose.create_from_pq(p=cube_xyz))
            self.cube_initial_z = cube_xyz[:, 2].clone()

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

        # Success: cube is lifted above initial z position AND drawer >20% open
        cube_height = self.cube.pose.p[..., 2]
        if cube_height.ndim > 1:
            cube_height = cube_height.squeeze(-1)
        initial_z = self.cube_initial_z
        if initial_z.ndim > 1:
            initial_z = initial_z.squeeze(-1)
        is_lifted = cube_height > initial_z

        # Check drawer is more than 20% open
        qlimits = self.handle_link.joint.limits
        qmin, qmax = qlimits[..., 0], qlimits[..., 1]
        qpos = self.handle_link.joint.qpos
        if qpos.ndim > 1:
            qpos = qpos.squeeze(-1)
        if qmin.ndim > 1:
            qmin = qmin.squeeze(-1)
        if qmax.ndim > 1:
            qmax = qmax.squeeze(-1)
        drawer_open_pct = (qpos - qmin) / (qmax - qmin + 1e-6)
        drawer_open_enough = drawer_open_pct > 0.20

        return {
            "success": is_lifted & drawer_open_enough,
            "handle_link_pos": self.handle_link_positions(),
            "open_enough": open_enough,
            "is_lifted": is_lifted,
            "drawer_open_enough": drawer_open_enough,
            "cube_height": cube_height,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                cube_pose=self.cube.pose.raw_pose,
                cabinet_qpos=self.cabinet.qpos,
                handle_pos=info["handle_link_pos"],
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_pos = self.agent.tcp.pose.p
        handle_pos = info["handle_link_pos"]
        cube_pos = self.cube.pose.p

        # Reach handle
        tcp_to_handle = torch.linalg.norm(tcp_pos - handle_pos, axis=1)
        reach_reward = 1 - torch.tanh(5.0 * tcp_to_handle)

        # Open drawer
        open_enough = info["open_enough"]
        open_reward = open_enough.float() * 2.0

        # Reach cube (after drawer is open)
        tcp_to_cube = torch.linalg.norm(tcp_pos - cube_pos, axis=1)
        cube_reach_reward = (1 - torch.tanh(5.0 * tcp_to_cube)) * open_enough.float()

        # Lift reward
        lift_reward = info["is_lifted"].float() * 3.0

        reward = reach_reward + open_reward + cube_reach_reward + lift_reward
        reward[info["success"]] = 10.0

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs, action, info) / 10.0
