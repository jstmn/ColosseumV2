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
    "PlaceCubeInDrawer-v1",
    asset_download_ids=["partnet_mobility_cabinet"],
    max_episode_steps=200,
)
class PlaceCubeInDrawerEnv(BaseEnv):
    """
    **Task Description:**
    Open a drawer, pick up a cube from the table, and place it inside the drawer.

    **Randomizations:**
    - Cube spawns on the table with small xy noise.

    **Success Conditions:**
    - The drawer is open (>30% of range).
    - The cube is inside the drawer.
    - The cube is static.
    - The robot is not grasping the cube.
    """

    SUPPORTED_ROBOTS = ["panda"]
    agent: Union[Panda]
    handle_types = ["prismatic"]  # Drawer joints

    CUBE_HALF_SIZE = 0.035

    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.0,
        **kwargs,
    ):
        kwargs.pop("distraction_set", None)
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self._model_id = 45427  # Same cabinet as PickCubeFromDrawer

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
        # Use the bottom drawer (same as PickCubeFromDrawer)
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

        # Add drive properties to allow drawer to be driven open
        for cabinet in self._cabinets:
            for joint in cabinet.joints:
                if joint.type[0] == "prismatic":
                    joint.set_drive_properties(stiffness=0.0, damping=50.0)

    def _after_reconfigure(self, options):
        self.cabinet_zs = []
        for cabinet in self._cabinets:
            collision_mesh = cabinet.get_first_collision_mesh()
            self.cabinet_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.cabinet_zs = common.to_tensor(self.cabinet_zs, device=self.device)

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
            # Robot is at Y=-0.615, cabinet rotated so drawer faces -Y (towards robot)
            # Swapped: cabinet now on the right side
            cabinet_pos = torch.zeros((b, 3))
            cabinet_pos[:, 0] = 0.0     # X position (right side)
            cabinet_pos[:, 1] = -0.45    # Y position
            cabinet_pos[:, 2] = self.cabinet_zs[env_idx]

            # Rotate 90° clockwise around Z so drawer faces -Y
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

            # Initialize robot at pre-grasp pose (ready to grasp handle)
            pregrasp_qpos = np.array([
                1.1327756643295288, 1.102452039718628, -1.1890534162521362, -1.8933054208755493,
                -0.6512154340744019, 2.07772159576416, -1.7972638607025146,
                0.04, 0.04  # gripper open
            ])
            self.table_scene.initialize(env_idx, qpos_0=pregrasp_qpos)

            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene.px.step()
                self.scene._gpu_fetch_all()

            self.handle_link_goal.set_pose(
                Pose.create_from_pq(p=self.handle_link_positions(env_idx))
            )

            # Place cube on table (NOT in drawer - this is the key difference)
            self._place_cube_on_table(env_idx)

    def _place_cube_on_table(self, env_idx: torch.Tensor):
        """Place cube on the table with random position offset."""
        with torch.device(self.device):
            b = len(env_idx)

            # Place cube on table, to the left of the cabinet
            # Robot is at Y=-0.615, cabinet at X=0.10
            # Swapped: cube now on the left side
            cube_xyz = torch.zeros((b, 3))
            cube_xyz[:, 0] = -0.30 + (torch.rand(b) - 0.5) * 0.08  # X: to the left
            cube_xyz[:, 1] = 0.30 + (torch.rand(b) - 0.5) * 0.08  # Y: between robot and cabinet
            cube_xyz[:, 2] = self.CUBE_HALF_SIZE

            self.cube.set_pose(Pose.create_from_pq(p=cube_xyz))

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
        cube_pos = self.cube.pose.p
        handle_pos = self.handle_link_positions()

        # Check if drawer is open enough (>30% of range)
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
        is_drawer_open = drawer_open_pct > 0.30

        # Check if cube is inside the drawer
        # Drawer interior is behind the handle (in -Y direction since drawer faces -Y)
        drawer_center = handle_pos.clone()
        drawer_center[..., 1] -= 0.12  # Behind handle into drawer

        cube_to_drawer_dist_xy = torch.linalg.norm(
            cube_pos[..., :2] - drawer_center[..., :2], dim=-1
        )
        cube_z_ok = torch.abs(cube_pos[..., 2] - drawer_center[..., 2]) < 0.08
        is_cube_in_drawer = (cube_to_drawer_dist_xy < 0.10) & cube_z_ok

        # Check if cube is static
        is_cube_static = self.cube.is_static(lin_thresh=0.1, ang_thresh=0.5)

        # Check if robot is not grasping cube
        is_cube_grasped = self.agent.is_grasping(self.cube)

        success = is_drawer_open & is_cube_in_drawer & is_cube_static & (~is_cube_grasped)

        return {
            "success": success,
            "is_drawer_open": is_drawer_open,
            "is_cube_in_drawer": is_cube_in_drawer,
            "is_cube_static": is_cube_static,
            "is_cube_grasped": is_cube_grasped,
            "drawer_open_pct": drawer_open_pct,
            "handle_link_pos": handle_pos,
        }

    def _get_obs_extra(self, info: Dict):
        obs = {"tcp_pose": self.agent.tcp.pose.raw_pose}
        if "state" in self.obs_mode:
            obs.update(
                cube_pose=self.cube.pose.raw_pose,
                cabinet_qpos=self.cabinet.qpos,
                handle_pos=info["handle_link_pos"],
                tcp_to_cube_pos=self.cube.pose.p - self.agent.tcp.pose.p,
                tcp_to_handle_pos=info["handle_link_pos"] - self.agent.tcp.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_pos = self.agent.tcp.pose.p
        handle_pos = info["handle_link_pos"]
        cube_pos = self.cube.pose.p

        reward = torch.zeros(self.num_envs, device=self.device)

        # Phase 1: Reach and open drawer
        tcp_to_handle = torch.linalg.norm(tcp_pos - handle_pos, axis=1)
        reach_handle_reward = 1 - torch.tanh(5.0 * tcp_to_handle)
        reward += reach_handle_reward

        # Drawer opening reward
        drawer_reward = info["drawer_open_pct"] * 2.0
        reward += drawer_reward

        # Phase 2: After drawer is open, reach cube
        is_open = info["is_drawer_open"]
        tcp_to_cube = torch.linalg.norm(tcp_pos - cube_pos, axis=1)
        reach_cube_reward = (1 - torch.tanh(5.0 * tcp_to_cube)) * is_open.float()
        reward += reach_cube_reward

        # Grasp reward
        is_grasping = info["is_cube_grasped"]
        grasp_reward = is_grasping.float() * is_open.float()
        reward += grasp_reward

        # Phase 3: Move cube to drawer
        drawer_center = handle_pos.clone()
        drawer_center[..., 1] -= 0.12
        cube_to_drawer = torch.linalg.norm(cube_pos - drawer_center, axis=1)
        place_reward = (1 - torch.tanh(5.0 * cube_to_drawer)) * is_grasping.float()
        reward += place_reward

        # Cube in drawer bonus
        reward += info["is_cube_in_drawer"].float() * 2.0

        # Success bonus
        reward[info["success"]] = 10.0

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs, action, info) / 10.0
