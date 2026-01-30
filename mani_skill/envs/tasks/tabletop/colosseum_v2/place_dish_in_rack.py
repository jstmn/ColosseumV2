import logging
from typing import Any, Dict, Union

import numpy as np
import sapien
import sapien.render
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from sapien.physx import PhysxMaterial
from mani_skill.agents.controllers import PDEEPoseControllerConfig
from mani_skill.envs.tasks.tabletop.colosseum_v2.distraction_set import DistractionSet

logger = logging.getLogger(__name__)


@register_env("PlaceDishInRack-v1", max_episode_steps=100)
class PlaceDishInRackEnv(BaseEnv):
    """
    **Task Description:**
    Pick up the plate and place it vertically into the upright slots of the dish rack.

    **Randomizations:**
    - The plate starts randomly on the table near the robot.
    - The dish rack pose is randomized slightly on the tabletop.

    **Success Conditions:**
    - The plate is upright and centered inside the dish rack while the robot releases it.
    """

    agent: Union[Panda, Fetch]

    _rack_mesh_path = PACKAGE_ASSET_DIR / "dish_into_rack/dish_rack_with_connectors.stl"
    _plate_visual_mesh_path = (
        PACKAGE_ASSET_DIR / "dish_into_rack/white_ceramic_serving_bowl.glb"
    )
    _plate_mesh_source_radius = 0.5  # Radius of the raw OBJ (measured once offline)
    _plate_mesh_source_height = 0.2494586706161499  # OBJ height once flattened
    _plate_mesh_flat_quat = [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]  # Rotate mesh so Z is the plate normal
    _rack_scale = 0.0015  # Rack to match

    # Plate geometry parameters (meters)
    _plate_outer_radius = 0.09  # Desired radius after scaling the OBJ
    _plate_inner_radius = 0.07  # Left for planners/controllers that rely on this value
    _plate_density = 300.0  # Lower density = lighter = easier to hold
    _plate_total_height = (
        _plate_mesh_source_height * (_plate_outer_radius / _plate_mesh_source_radius)
    )
    # Legacy attributes kept for compatibility with existing planners/utilities.
    _plate_base_thickness = _plate_total_height * 0.25
    _plate_rim_height = _plate_total_height - _plate_base_thickness
    _plate_extent = np.array(
        [_plate_outer_radius * 2, _plate_outer_radius * 2, _plate_total_height]
    )
    _plate_spawn_buffer = 0.002  # Small buffer to prevent initial interpenetration

    _rack_extent = np.array([0.12060600281, 0.16782440567, 0.085])  # Normal rack size
    # STL is now centered at origin, no offset needed

    _plate_goal_offset = np.array([0.0, 0.0, 0.15])  # Above rack slots
    _rack_position = np.array([-0.1, 0.1, 0])  # Rack position closer to robot workspace
    _plate_support_radius = 0.015
    _plate_support_height = 0.0  # No pedestal - plate flush with table

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        # Use the default robot joint configuration with light noise.
        distraction_set: DistractionSet | dict | None = kwargs.pop("distraction_set", None)
        self._distraction_set: DistractionSet | None = DistractionSet(**distraction_set) if isinstance(distraction_set, dict) else distraction_set
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq=200,  # Moderate increase for better physics (default is 100)
            control_freq=20,  # Keep control frequency the same
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**23, max_rigid_patch_count=2**17
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.2, -0.2, 0.4], target=[-0.3, 0.0, 0.0])
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

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.65, -0.35, 0.35], [0.05, 0.0, 0.1])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_agent(self, options: Dict):
        # Keep robot at normal position
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

        # Keep table at default position
        self.table_height_offset = 0.0

        self.plate = self._build_plate()
        # Add heavy damping so the plate stays still on the table until the robot touches it.
        # Without this, tiny numerical vibrations from the robot settling would slowly tip it upright.
        self.plate.set_linear_damping(5.0)
        self.plate.set_angular_damping(8.0)
        self.dish_rack = self._build_rack()
        self.plate_support = self._build_plate_support()

    def _build_plate(self):
        """Build the plate directly from the high-fidelity ceramic bowl mesh."""
        builder = self.scene.create_actor_builder()

        physical_material = PhysxMaterial(
            static_friction=20.0,
            dynamic_friction=20.0,
            restitution=0.0,
        )

        collision_scale = float(
            self._plate_outer_radius / self._plate_mesh_source_radius
        )
        mesh_pose = sapien.Pose(q=self._plate_mesh_flat_quat)
        builder.add_multiple_convex_collisions_from_file(
            filename=str(self._plate_visual_mesh_path),
            scale=[collision_scale, collision_scale, collision_scale],
            pose=mesh_pose,
            material=physical_material,
            density=self._plate_density,
            decomposition="coacd",
        )
        plate_visual_material = sapien.render.RenderMaterial(
            base_color=[1.0, 1.0, 1.0, 1.0],
            specular=0.4,
            roughness=0.2,
            metallic=0.0,
        )

        builder.add_visual_from_file(
            filename=str(self._plate_visual_mesh_path),
            scale=[collision_scale, collision_scale, collision_scale],
            pose=mesh_pose,
            material=plate_visual_material,
        )

        builder.initial_pose = sapien.Pose()
        return builder.build(name="plate")

    
    def _build_rack(self):
        builder = self.scene.create_actor_builder()

        # Collision geometry matching the exact STL mesh: base + 4 vertical dividers
        # Extracted from dish_rack_with_connectors.stl after 0.0015 scaling

        # Base plate dimensions (extracted from STL)
        base_width = 0.180906  # X
        base_depth = 0.251737  # Y
        base_thickness = 0.009999  # Z (about 1cm)
        base_center = [0.004323, 0.007770, -0.057710]

        builder.add_box_collision(
            half_size=[base_width / 2, base_depth / 2, base_thickness / 2],
            pose=sapien.Pose(p=base_center)
        )

        # Vertical divider positions and heights (extracted from STL)
        rack_width = self._rack_extent[0]  # Width for dividers to span across
        divider_y_positions = [-0.105254, -0.046585, 0.015046, 0.074831]  # 4 dividers
        divider_heights = [0.125304, 0.122871, 0.122871, 0.125304]
        divider_z_centers = [-0.001098, -0.002315, -0.002315, -0.001098]
        divider_thickness = 0.003  # 3mm thick

        # Add vertical dividers on top of base
        # Dividers run in X direction (left-right) so plates slide in from the front (Y direction)
        for y_pos, height, z_center in zip(divider_y_positions, divider_heights, divider_z_centers):
            builder.add_box_collision(
                half_size=[rack_width / 2, divider_thickness, height / 2],
                pose=sapien.Pose(p=[0, y_pos, z_center])
            )

        # Keep the visual mesh (now centered at origin)
        builder.add_visual_from_file(
            filename=str(self._rack_mesh_path),
            scale=[self._rack_scale] * 3,
        )
        builder.initial_pose = sapien.Pose()
        return builder.build_kinematic(name="dish_rack")

    def _build_plate_support(self):
        if self._plate_support_height <= 0.0:
            return None
        builder = self.scene.create_actor_builder()
        builder.add_cylinder_collision(
            radius=self._plate_support_radius,
            half_length=self._plate_support_height / 2,
        )
        builder.add_cylinder_visual(
            radius=self._plate_support_radius,
            half_length=self._plate_support_height / 2,
            material=sapien.render.RenderMaterial(base_color=[0.8, 0.8, 0.8, 1.0]),
        )
        builder.initial_pose = sapien.Pose()
        support = builder.build_kinematic(name="plate_support")
        return support

    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        device = self.device
        with torch.device(device):
            b = len(env_idx)

            # First compute plate position so we can position robot above it
            # Randomize plate position within reachable zone - 20cm range (±0.1m)
            plate_x = -0.3 + (torch.rand(b, device=device) - 0.5) * 0.2  # ±0.1m in X
            plate_y = -0.2 + (torch.rand(b, device=device) - 0.5) * 0.2  # ±0.1m in Y

            # Use heuristic-based qpos with base rotation toward plate
            # plate_x_val = float(plate_x[0].cpu()) if b == 1 else -0.45
            # plate_y_val = float(plate_y[0].cpu()) if b == 1 else -0.25

            # Compute angle from robot base to plate
            # Robot base is at (-0.615, 0), plate is at (plate_x_val, plate_y_val)
            # angle_to_plate = np.arctan2(plate_y_val, plate_x_val + 0.615)

            # Use the default working arm configuration from TableSceneBuilder
            # but with joint 0 rotated to point toward the plate
            # Default: [0.0, π/8, 0, -π*5/8, 0, π*3/4, π/4, 0.04, 0.04]
            # panda_qpos_above_plate = np.array([
            #     angle_to_plate,        # j0: base rotation toward plate
            #     np.pi / 8,             # j1: shoulder slightly forward
            #     0,                     # j2: upper arm
            #     -np.pi * 5 / 8,        # j3: elbow bent
            #     0,                     # j4: forearm
            #     np.pi * 3 / 4,         # j5: wrist 1
            #     np.pi / 4,             # j6: wrist 2
            #     0.04,                  # gripper left
            #     0.04,                  # gripper right
            # ])
            panda_qpos_above_plate = np.array(
                [-1.08, 0, 0.68, -2.64, 0.07, 2.6, -1.25, 0.04, 0.04]
            )


            self.table_scene.initialize(env_idx, qpos_0=panda_qpos_above_plate)

            # Compute table top Z for placing objects
            table_p = self.table_scene.table.pose.p
            if table_p.ndim == 2:
                table_z = float(table_p[0, 2].cpu())
            else:
                table_z = float(table_p[2].cpu())
            table_top_z = table_z + float(self.table_scene.table_height)

            # Set plate position (using pre-computed random values)
            xyz = torch.zeros((b, 3), device=device)
            xyz[:, 0] = plate_x
            xyz[:, 1] = plate_y
            plate_half_height = self._plate_total_height / 2.0
            xyz[:, 2] = table_top_z

            # Randomize plate yaw (rotation around Z axis)
            plate_yaw = torch.rand(b, device=device) * 2 * np.pi  # Random yaw 0 to 2π
            # Convert yaw to quaternion (rotation around Z): [cos(θ/2), 0, 0, sin(θ/2)]
            flat_quat = torch.zeros((b, 4), device=device)
            flat_quat[:, 0] = torch.cos(plate_yaw / 2)  # w
            flat_quat[:, 3] = torch.sin(plate_yaw / 2)  # z

            plate_pose = Pose.create_from_pq(p=xyz, q=flat_quat)
            self.plate.set_pose(plate_pose)

            # Randomize rack position - 10cm range (±0.05m)
            # Note: rack yaw is NOT randomized because motion planning uses fixed offsets
            rack_pos = torch.zeros((b, 3), device=device)
            rack_pos[:] = torch.tensor(self._rack_position, device=device)
            rack_pos[:, 0] += (torch.rand(b, device=device) - 0.5) * 0.1  # ±0.05m in X
            rack_pos[:, 1] += (torch.rand(b, device=device) - 0.5) * 0.1  # ±0.05m in Y
            rack_pos[:, 2] = table_top_z + float(self._rack_extent[2])
            rack_pose = Pose.create_from_pq(p=rack_pos)
            self.dish_rack.set_pose(rack_pose)

            # Position plate support pedestal under plate if it exists
            if self.plate_support is not None:
                support_pos = xyz.clone()
                support_pos[:, 2] = self._plate_support_height / 2  # Center of cylinder
                support_pose = Pose.create_from_pq(p=support_pos)
                self.plate_support.set_pose(support_pose)

            # Let the plate settle on the table for stable physics
            for _ in range(50):
                self.scene.step()

            # Force the plate back to its intended flat pose in case it rolled during settling.
            xyz[:, 2] = table_top_z + plate_half_height + self._plate_spawn_buffer
            plate_pose = Pose.create_from_pq(p=xyz, q=flat_quat)
            self.plate.set_pose(plate_pose)

            # Zero velocities after settling (but keep the settled pose to avoid interpenetration)
            zero_velocity = torch.zeros((b, 3), device=device)
            self.plate.set_linear_velocity(zero_velocity)
            self.plate.set_angular_velocity(zero_velocity)

            # Reset robot to target position after settling (controller may have drifted during steps)
            self.agent.reset(panda_qpos_above_plate)

    def evaluate(self):
        plate_pos = self.plate.pose.p
        rack_pos = self.dish_rack.pose.p
        rack_half = torch.tensor(
            self._rack_extent / 2.0, device=self.device, dtype=plate_pos.dtype
        )
        within_x = torch.abs(plate_pos[:, 0] - rack_pos[:, 0]) <= rack_half[0]
        within_y = torch.abs(plate_pos[:, 1] - rack_pos[:, 1]) <= rack_half[1]
        plate_within_rack = within_x & within_y

        rot_mats = quaternion_to_matrix(self.plate.pose.q)
        plate_norm = rot_mats[..., 2]
        plate_vertical = torch.abs(plate_norm[..., 2]) <= 0.35

        is_grasped = self.agent.is_grasping(self.plate)
        is_static = self.plate.is_static(lin_thresh=0.02, ang_thresh=0.4)

        # Check that plate is above table surface (not clipping through)
        # Table surface is at z=0, plate bottom should be at least at z > -0.01
        above_table = plate_pos[:, 2] > -0.01

        # Success requires: plate within rack bounds, vertical orientation, static, and released
        # This prevents counting plates resting ON TOP of the rack as success
        success = plate_within_rack & plate_vertical & is_static & ~is_grasped & above_table

        return {
            "success": success,
            "plate_close_to_goal": plate_within_rack,
            "plate_vertical": plate_vertical,
            "is_static": is_static,
            "is_grasped": is_grasped,
            "above_table": above_table,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            plate_pose = self.plate.pose
            rack_pose = self.dish_rack.pose
            obs.update(
                plate_pos=plate_pose.p,
                plate_quat=plate_pose.q,
                rack_pos=rack_pose.p,
                rack_quat=rack_pose.q,
                plate_to_goal=plate_pose.p - rack_pose.p,
                tcp_to_plate=plate_pose.p - self.agent.tcp_pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict) -> torch.Tensor:
        """Compute dense reward for the task."""
        plate_pos = self.plate.pose.p
        rack_pos = self.dish_rack.pose.p
        rack_half = torch.tensor(
            self._rack_extent / 2.0, device=self.device, dtype=plate_pos.dtype
        )
        within_x = torch.abs(plate_pos[:, 0] - rack_pos[:, 0]) <= rack_half[0]
        within_y = torch.abs(plate_pos[:, 1] - rack_pos[:, 1]) <= rack_half[1]
        plate_within_rack = within_x & within_y

        # Distance reward
        reaching_reward = plate_within_rack.float()

        # Orientation reward (plate should be vertical)
        rot_mats = quaternion_to_matrix(self.plate.pose.q)
        plate_norm = rot_mats[..., 2]
        vertical_alignment = torch.abs(plate_norm[..., 2])
        orientation_reward = 1.0 - vertical_alignment

        # Gripper release reward
        is_grasped = self.agent.is_grasping(self.plate)
        release_reward = torch.where(
            plate_within_rack,
            torch.where(is_grasped, torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device)),
            torch.tensor(0.0, device=self.device)
        )

        # Success bonus
        success = info["success"].float()
        success_reward = success * 5.0

        reward = reaching_reward + orientation_reward + release_reward + success_reward
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict) -> torch.Tensor:
        """Compute normalized dense reward."""
        return self.compute_dense_reward(obs, action, info) / 8.0
