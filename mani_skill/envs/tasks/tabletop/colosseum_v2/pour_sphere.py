from typing import Any, Dict, Union

import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.envs.tasks.tabletop.colosseum_v2.distraction_set import DistractionSet


@register_env("PourSphere-v1", max_episode_steps=200)
class PourSphereEnv(BaseEnv):
    """
    **Task Description:**
    Pick up the cup containing a sphere and pour the sphere into the second cup.

    **Randomizations:**
    - The cups start at fixed positions on the table.
    - The sphere starts in the first cup.

    **Success Conditions:**
    - The sphere is within the bounding box of the second cup.
    """

    agent: Union[Panda, Fetch]

    # Cup parameters - cup diameter ~0.10m, slightly larger than gripper opening (~0.08m)
    # Gripper can grip the rim edge but can't fit inside
    _cup_height = 0.10
    _cup_radius = 0.05  # diameter 0.10m - gripper grips rim edge
    _cup_thickness = 0.006  # thicker rim for better grip

    # Sphere parameters
    _sphere_radius = 0.02

    # Cup positions - default (used if no randomization)
    _cup1_position = np.array([0.0, -0.10, 0.0])
    _cup2_position = np.array([0.0, 0.15, 0.0])

    # Randomization bounding box for cups (relative to table center)
    # Cup1 (source cup with sphere)
    _cup1_x_range = (-0.02, 0.06)  # x range
    _cup1_y_range = (-0.15, -0.10)  # y range (right side of robot)
    # Cup2 (target cup)
    _cup2_x_range = (-0.02, 0.06)  # x range
    _cup2_y_range = (0.10, 0.15)  # y range (left side of robot)

    # Robot positioning
    _robot_angle = 0
    _robot_base_pos = np.array([-0.5, 0.0, 0])
    _cup_mesh_path = PACKAGE_ASSET_DIR / "pour_sphere/hollow_cylinder_with_floor.stl"

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        distraction_set: DistractionSet | dict | None = kwargs.pop("distraction_set", None)
        self._distraction_set: DistractionSet | None = DistractionSet(**distraction_set) if isinstance(distraction_set, dict) else distraction_set
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        from mani_skill.utils.structs.types import SceneConfig
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**23, max_rigid_patch_count=2**17
            ),
            scene_config=SceneConfig(
                solver_position_iterations=30,
                solver_velocity_iterations=3,
                contact_offset=0.005,
                rest_offset=0.0,
            )
        )

    @property
    def _default_sensor_configs(self):
        # Sensor camera with view of robot arm and cups (matching OpenCabinet setup)
        pose = sapien_utils.look_at(eye=[-0.4, -0.5, 0.6], target=[0.0, 0.0, 0.35])
        return self.update_camera_configs([
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ])

    @property
    def _default_human_render_camera_configs(self):
        # View from behind-left of robot (matching OpenCabinet setup)
        pose = sapien_utils.look_at(eye=[-0.8, -0.6, 0.7], target=[0.1, 0.0, 0.35])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_agent(self, options: Dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _get_obs_agent(self):
        obs = super()._get_obs_agent()
        for key in ("world__T__ee", "world__T__root"):
            value = obs.get(key, None)
            if isinstance(value, torch.Tensor) and value.ndim == 3:
                obs[key] = value.reshape(value.shape[0], -1)
        return obs

    def _load_scene(self, options: Dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        
        # Build hollow cups
        self.cup1 = self._build_hollow_cup("cup1", body_type="dynamic")
        self.cup2 = self._build_hollow_cup("cup2", body_type="kinematic")
        self.cup1.set_mass(0.15)
        self.cup1.set_linear_damping(5.0)
        self.cup1.set_angular_damping(10.0)

        # Build sphere that will be poured from cup1 to cup2
        builder = self.scene.create_actor_builder()
        sphere_material = physx.PhysxMaterial(
            static_friction=1.5, dynamic_friction=1.0, restitution=0.0
        )
        builder.add_sphere_collision(
            radius=self._sphere_radius, material=sphere_material, density=1000
        )
        builder.add_sphere_visual(
            radius=self._sphere_radius,
            material=sapien.render.RenderMaterial(base_color=[0.2, 0.6, 0.8, 1])
        )
        builder.set_initial_pose(sapien.Pose(p=[0, 0, 0]))
        self.sphere = builder.build_dynamic(name="sphere")
        self.sphere.set_linear_damping(1.0)
        self.sphere.set_angular_damping(1.0)
        if hasattr(self.sphere, "set_sleep_threshold"):
            self.sphere.set_sleep_threshold(0.02)


    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        with torch.device(self.device):
            b = len(env_idx)

            # Position robot
            robot_pose = sapien.Pose(p=self._robot_base_pos)

            # Default Panda qpos - arm pointing down, ready position
            default_qpos = np.array([
                0.0, -0.785, 0.0, -2.356,
                0.0, 1.571, 0.785, 0.04, 0.04
            ])

            # Initialize table scene with default qpos
            self.table_scene.initialize(env_idx, table_z_rotation_angle=np.pi, qpos_0=default_qpos)
            self.agent.robot.set_pose(robot_pose)

            # Randomize cup1 position within defined ranges
            cup1_x = np.random.uniform(*self._cup1_x_range)
            cup1_y = np.random.uniform(*self._cup1_y_range)
            cup1_xyz = np.array([cup1_x, cup1_y, 0.001])
            cup1_q = np.array([1, 0, 0, 0])
            self.cup1.set_pose(Pose.create_from_pq(p=cup1_xyz, q=cup1_q))

            # Randomize cup2 position within defined ranges
            cup2_x = np.random.uniform(*self._cup2_x_range)
            cup2_y = np.random.uniform(*self._cup2_y_range)
            cup2_xyz = np.array([cup2_x, cup2_y, 0.001])
            cup2_q = np.array([1, 0, 0, 0])
            self.cup2.set_pose(Pose.create_from_pq(p=cup2_xyz, q=cup2_q))

            # Position sphere inside cup1 (at bottom of cup)
            sphere_xyz = cup1_xyz.copy()
            sphere_xyz[2] = (
                cup1_xyz[2] + self._cup_thickness + self._sphere_radius + 0.001
            )
            self.sphere.set_pose(Pose.create_from_pq(p=sphere_xyz, q=np.array([1, 0, 0, 0])))
            zero_velocity = torch.zeros((b, 3), device=self.device)
            self.sphere.set_linear_velocity(zero_velocity)
            self.sphere.set_angular_velocity(zero_velocity)

            # GPU sync
            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene.px.step()
                self.scene._gpu_fetch_all()

    def _build_hollow_cup(self, name: str, body_type: str = "kinematic"):
        """Build a hollow cylindrical cup from STL file"""
        builder = self.scene.create_actor_builder()
        cup_material = physx.PhysxMaterial(
            static_friction=1.5, dynamic_friction=1.0, restitution=0.01
        )

        cup_stl_path = str(self._cup_mesh_path)
        stl_scale = 0.10 / 35.0

        builder.add_visual_from_file(
            filename=cup_stl_path,
            scale=[stl_scale, stl_scale, stl_scale],
            material=sapien.render.RenderMaterial(base_color=[0.8, 0.6, 0.4, 1]),
        )
        builder.add_multiple_convex_collisions_from_file(
            filename=cup_stl_path,
            scale=[stl_scale, stl_scale, stl_scale],
            material=cup_material,
            density=2000,
            decomposition="coacd",
        )
        builder.set_initial_pose(sapien.Pose(p=[0, 0, 0]))
        if body_type == "dynamic":
            return builder.build_dynamic(name=name)
        elif body_type == "kinematic":
            return builder.build_kinematic(name=name)
        elif body_type == "static":
            return builder.build_static(name=name)
        raise ValueError(f"Unsupported body type: {body_type}")

    def _add_cup_wall_collisions(
        self, builder: sapien.ActorBuilder, material: physx.PhysxMaterial, density: float
    ):
        wall_thickness = float(self._cup_thickness)
        outer_radius = float(self._cup_radius)
        inner_radius = max(outer_radius - wall_thickness, wall_thickness)
        wall_half_thickness = wall_thickness / 2.0
        wall_half_height = float(self._cup_height) / 2.0
        wall_center_z = wall_half_height

        builder.add_box_collision(
            half_size=[wall_half_thickness, inner_radius, wall_half_height],
            pose=sapien.Pose(p=[inner_radius + wall_half_thickness, 0.0, wall_center_z]),
            material=material,
            density=density,
        )
        builder.add_box_collision(
            half_size=[wall_half_thickness, inner_radius, wall_half_height],
            pose=sapien.Pose(p=[-(inner_radius + wall_half_thickness), 0.0, wall_center_z]),
            material=material,
            density=density,
        )
        builder.add_box_collision(
            half_size=[inner_radius, wall_half_thickness, wall_half_height],
            pose=sapien.Pose(p=[0.0, inner_radius + wall_half_thickness, wall_center_z]),
            material=material,
            density=density,
        )
        builder.add_box_collision(
            half_size=[inner_radius, wall_half_thickness, wall_half_height],
            pose=sapien.Pose(p=[0.0, -(inner_radius + wall_half_thickness), wall_center_z]),
            material=material,
            density=density,
        )
        bottom_half_height = wall_half_thickness
        builder.add_box_collision(
            half_size=[inner_radius, inner_radius, bottom_half_height],
            pose=sapien.Pose(p=[0.0, 0.0, bottom_half_height]),
            material=material,
            density=density,
        )


    def evaluate(self):
        """Evaluate task success - sphere must be within cup2's bounding box."""
        sphere_pos = self.sphere.pose.p
        cup2_pos = self.cup2.pose.p

        within_x = torch.abs(sphere_pos[:, 0] - cup2_pos[:, 0]) <= self._cup_radius
        within_y = torch.abs(sphere_pos[:, 1] - cup2_pos[:, 1]) <= self._cup_radius
        within_z = sphere_pos[:, 2] >= (cup2_pos[:, 2] - self._cup_height / 2)
        within_z &= sphere_pos[:, 2] <= (cup2_pos[:, 2] + self._cup_height / 2)

        success = within_x & within_y & within_z

        return {
            "success": success,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)

        if "state" in self.obs_mode:
            cup1_pose = self.cup1.pose
            cup2_pose = self.cup2.pose
            sphere_pose = self.sphere.pose
            obs.update(
                cup1_pos=cup1_pose.p,
                cup1_quat=cup1_pose.q,
                cup2_pos=cup2_pose.p,
                cup2_quat=cup2_pose.q,
                sphere_pos=sphere_pose.p,
                tcp_to_cup1=cup1_pose.p - self.agent.tcp_pose.p,
                sphere_to_cup2=cup2_pose.p - sphere_pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict) -> torch.Tensor:
        """Compute dense reward for the task."""
        cup1_pos = self.cup1.pose.p
        cup2_pos = self.cup2.pose.p
        sphere_pos = self.sphere.pose.p
        tcp_pos = self.agent.tcp_pose.p

        # Reward for reaching cup1
        tcp_to_cup1_dist = torch.linalg.norm(tcp_pos - cup1_pos, dim=1)
        reach_cup1_reward = 1.0 - torch.tanh(3.0 * tcp_to_cup1_dist)

        # Reward for grasping cup1
        is_grasping_cup1 = self.agent.is_grasping(self.cup1)
        grasp_reward = is_grasping_cup1.float() * 2.0

        # Reward for getting sphere close to cup2
        sphere_to_cup2_dist = torch.linalg.norm(sphere_pos - cup2_pos, dim=1)
        sphere_to_cup2_reward = (1.0 - torch.tanh(2.0 * sphere_to_cup2_dist)) * 3.0

        # Bonus for sphere in cup2
        success_reward = info["success"].float() * 5.0

        return reach_cup1_reward + grasp_reward + sphere_to_cup2_reward + success_reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict) -> torch.Tensor:
        """Compute normalized dense reward."""
        return self.compute_dense_reward(obs, action, info) / 11.0
