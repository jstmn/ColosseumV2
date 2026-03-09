from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Union

import numpy as np
import sapien
import torch
import sapien.physx as physx
import sapien.render
from scipy.spatial.transform import Rotation
from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.envs.tasks.tabletop.colosseum_v2.colosseum_v2_core import ColosseumV2Env, DisabledVariationFactors, PlacementRegion

YCB_HAMMER_ID = "048_hammer"
NAIL_HEIGHT = 0.086


@dataclass
class NailSpec:
    name: str
    rest_center: np.ndarray
    success_center_y: float  # Changed from z to y for sideways hammering
    raise_range: float


@register_env("HammerNail-v1", max_episode_steps=200)
class HammerNailEnv(ColosseumV2Env):
    """A tabletop hammering task with a single nail and a simple wooden block.

    **Task**: Pick up the hammer and drive the nail horizontally (sideways) into the
    vertical block until the nail center passes a target depth.

    Assets:
    - YCB hammer model (048_hammer). Falls back to a simple procedural hammer if YCB assets
      are not available.
    - Nail and block are procedurally built primitives."""

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    DISABLED_VARIATION_FACTORS = DisabledVariationFactors(RO_size=True)

    def __init__(
        self,
        *args,
        robot_uids: Union[str, Sequence[str]] = "panda",
        robot_init_qpos_noise: float = 0.0,  # No noise to prevent self-collision
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self._asset_scale = 0.05

        # Block is now vertical (tall in Z, narrow in X and Y)
        # Rotated 90° around Z so the nail can be hammered horizontally into the left side.
        self._block_half_size = torch.tensor([0.015, 0.03, 0.06], dtype=torch.float32)  # Thin in X, normal Y, tall in Z
        self._block_center = torch.tensor([0.15, 0.10, 0.06], dtype=torch.float32)  # Positioned to the left of the robot
        # After Z rotation, local X maps to world Y.
        self._nail_dir = -1.0  # -1 means nail points toward -Y (sticks out on left)
        self._block_entry_y = float(
            self._block_center[1] + self._block_half_size[0]
            if self._nail_dir < 0
            else self._block_center[1] - self._block_half_size[0]
        )
        self._hole_center_z = 0.04  # Raise the nail channel within the block.

        # Nail starts sticking out horizontally (pointing in +Y direction towards block)
        nail_initial_insert = 0.05  # Start mostly out so it needs multiple strikes
        nail_target_insert = 0.06    # Drive it deeper into block (in +Y direction)
        nail_rest_center = np.array(
            [
                self._block_center[0].item(),
                self._block_entry_y
                - self._nail_dir * NAIL_HEIGHT / 2
                + self._nail_dir * nail_initial_insert,
                (self._block_center[2] + self._hole_center_z).item(),  # Hole center height
            ],
            dtype=np.float32,
        )
        self._nail_success_center_y = float(
            self._block_entry_y
            - self._nail_dir * NAIL_HEIGHT / 2
            + self._nail_dir * nail_target_insert
        )

        self._nail_spec = NailSpec(
                name="nail",
                rest_center=nail_rest_center,
                success_center_y=self._nail_success_center_y,
                raise_range=0.005,
            )

    # Simple box hammer design: rotated 270° so handle at -X (left), head at +X (right)
        # Perfect for striking horizontally along the hammer's +X axis (mapped to world +Y after Z rotation)
        handle_size = np.array([0.04, 0.15, 0.04])  # Handle: thicker for better grip (4cm x 15cm x 4cm)
        head_size = np.array([0.06, 0.09, 0.06])    # Head: larger for better collision (6cm x 9cm x 6cm)
        self._hammer_handle_size = handle_size
        self._hammer_head_size = head_size
        self._nail_mesh_path = PACKAGE_ASSET_DIR / "hammer_nail/steel_nail.stl"
        handle_density = 2000  # kg/m³
        head_density = 6173   # kg/m³ (head volume 0.000324 m³ * 6173 ≈ 2.0 kg)

        # Position hammer on table (will be set by _initialize_episode)
        self._hammer_rest_center = torch.tensor(
            [-0.1, 0.35, 0.05], dtype=torch.float32
        )
        # Hammer orientation [w, x, y, z] - updated on load to keep the thinnest axis vertical.
        self._hammer_orientation = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], dtype=torch.float32
        )
        # After 270° rotation: head is at +X (right side), handle at -X (left side)
        self._hammer_head_offset = handle_size[1] / 2 + head_size[1] / 2  # ~0.115m

        # Calculate center of mass for proper grasping
        # Before rotation: handle at origin, head at [0, 0.115, 0]
        # After 270° rotation: handle at origin, head at [0.115, 0, 0]
        handle_mass = np.prod(handle_size) * handle_density
        head_mass = np.prod(head_size) * head_density
        head_offset_x = handle_size[1] / 2 + head_size[1] / 2  # Positive after 270° rotation
        com_x = (handle_mass * 0 + head_mass * head_offset_x) / (handle_mass + head_mass)
        self._hammer_com_offset = torch.tensor([com_x, 0.0, 0.0], dtype=torch.float32)  # ~[0.093, 0, 0]

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        from mani_skill.utils.structs.types import SceneConfig
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**24, max_rigid_patch_count=2**17
            ),
            scene_config=SceneConfig(
                solver_position_iterations=25,  # Increase from default 15 for better collision
                solver_velocity_iterations=3,   # Increase from default 1
                contact_offset=0.01,            # Larger contact offset to prevent penetration
                rest_offset=0.0,                # Zero rest offset for solid contacts
                sleep_threshold=0.02,
                enable_ccd=True,                # Enable CCD globally
            )
        )

    @property
    def _default_sensor_configs(self):
        # View from front-right to see sideways hammering action
        pose = sapien_utils.look_at(eye=[0.5, 0.5, 0.3], target=[-0.1, 0.1, 0.0])
        return self.update_camera_configs([
            CameraConfig(
                "base_camera",
                pose=pose,
                width=224,
                height=224,
                fov=np.pi / 3,
                near=0.01,
                far=10,
            )
        ])

    @property
    def _default_human_render_camera_configs(self):
        # Better view for sideways hammering - from front-right angle
        pose = sapien_utils.look_at([0.4, 0.6, 0.35], [0.05, 0.10, 0.06 + self._hole_center_z])
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=1.0, near=0.01, far=10)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.55, 0.0, 0.0]))

    def _load_scene(self, options: dict):
        self._load_block()
        self._load_nails()
        self._load_hammer()
        
        self.block = self.add_asset_to_scene(self._load_block, name="block", physics_type="kinematic", object_type="BACKGROUND")
        self.nail = self.add_asset_to_scene(self._load_nails, name="nail", physics_type="dynamic", object_type="MO")
        self.hammer = self.add_asset_to_scene(self._load_hammer, name="hammer", physics_type="dynamic", object_type="MO")
        self.load_scene_hook(manipulation_objects=[self.hammer], receiving_objects=[self.block])
        self._lock_table()


        # ========== Nail initialization ==========
        spec = self._nail_spec
        rest_centers: List[torch.Tensor] = []
        success_thresholds: List[torch.Tensor] = []
        self.nail.set_linear_damping(2.0)
        self.nail.set_angular_damping(5.0)
        # Lock X and Z linear movement, allow only Y; lock all rotations.
        # Must be set before GPU sim initialization.
        self.nail.set_locked_motion_axes([True, False, True, True, True, True])
        self.nails = [self.nail]

        rest_center = torch.from_numpy(spec.rest_center).to(torch.float32)
        rest_centers.append(rest_center)
        success_thresholds.append(torch.tensor(spec.success_center_y, dtype=torch.float32))

        self._nail_rest_centers = torch.stack(rest_centers)
        self._nail_success_threshold = torch.stack(success_thresholds)
        self._nail_raise_ranges = torch.tensor([self._nail_spec.raise_range], dtype=torch.float32)

        # ========== Hammer initialization ==========
        # self.hammer.set_mass(2.0)
        # Keep the hammer from drifting due to tiny numerical vibrations.
        self.hammer.set_linear_damping(10.0)
        self.hammer.set_angular_damping(12.0)

        self._choose_hammer_orientation()
        self._apply_hammer_z_rotation(-90.0)
        self._update_hammer_rest_height()

        self._hammer_region = self.update_placement_region(
            # Ground-truth from legacy sampling:
            # hammer_pos = self._hammer_rest_center.to(self.device).unsqueeze(0).repeat(b, 1)
            # # Add randomization: ±0.03m in X, ±0.03m in Y
            # hammer_pos[:, 0] += (torch.rand(b, device=self.device) - 0.5) * 0.06
            # hammer_pos[:, 1] += (torch.rand(b, device=self.device) - 0.5) * 0.06
            # => x,y are centered at hammer_rest_center with width 0.06
            PlacementRegion.from_center_and_width(
                center=(float(self._hammer_rest_center[0]), float(self._hammer_rest_center[1])),
                width=(0.06, 0.06),
            )
        )



    def _get_obs_agent(self):
        obs = super()._get_obs_agent()
        for key in ("world__T__ee", "world__T__root"):
            value = obs.get(key, None)
            if isinstance(value, torch.Tensor) and value.ndim == 3:
                obs[key] = value.reshape(value.shape[0], -1)
        return obs

    def _load_block(self):
        """Create a vertical wooden block with a square hole for the nail and collision mesh"""
        block_half = self._block_half_size.cpu().numpy()
        block_center = self._block_center.cpu().numpy()

        # Hole dimensions - square hole for the nail to sit in
        hole_half_size = 0.003  # 0.9cm x 0.9cm hole
        hole_center_z = float(self._hole_center_z)

        # Block dimensions
        # X range: block_center[0] - block_half[0] to block_center[0] + block_half[0]
        # Y range: block_center[1] - block_half[1] to block_center[1] + block_half[1]
        # Z range: block_center[2] - block_half[2] to block_center[2] + block_half[2]

        block_material = physx.PhysxMaterial(
            static_friction=2.5,
            dynamic_friction=2.0,
            restitution=0.0
        )
        wood_material = sapien.render.RenderMaterial(
            base_color=[0.6, 0.4, 0.2, 1.0],  # Wood color
            metallic=0.0,
            roughness=0.9
        )

        builder = self.scene.create_actor_builder()

        # Create block with hole by building collision boxes around the hole
        # The hole is at center Y and Z of the block, running through X

        # 1. Bottom section (below the hole)
        hole_bottom = hole_center_z - hole_half_size
        bottom_height = hole_bottom + block_half[2]
        bottom_half_z = bottom_height / 2
        bottom_center_z = -block_half[2] + bottom_half_z
        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, bottom_center_z]),
            half_size=[block_half[0], block_half[1], bottom_half_z],
            material=block_material
        )
        builder.add_box_visual(
            pose=sapien.Pose(p=[0, 0, bottom_center_z]),
            half_size=[block_half[0], block_half[1], bottom_half_z],
            material=wood_material
        )

        # 2. Top section (above the hole)
        hole_top = hole_center_z + hole_half_size
        top_height = block_half[2] - hole_top
        top_half_z = top_height / 2
        top_center_z = hole_top + top_half_z
        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, top_center_z]),
            half_size=[block_half[0], block_half[1], top_half_z],
            material=block_material
        )
        builder.add_box_visual(
            pose=sapien.Pose(p=[0, 0, top_center_z]),
            half_size=[block_half[0], block_half[1], top_half_z],
            material=wood_material
        )

        # 3. Front section (in front of the hole, in the hole Z-range)
        front_half_y = (block_half[1] - hole_half_size) / 2
        front_center_y = -block_half[1] + front_half_y
        builder.add_box_collision(
            pose=sapien.Pose(p=[0, front_center_y, hole_center_z]),
            half_size=[block_half[0], front_half_y, hole_half_size],
            material=block_material
        )
        builder.add_box_visual(
            pose=sapien.Pose(p=[0, front_center_y, hole_center_z]),
            half_size=[block_half[0], front_half_y, hole_half_size],
            material=wood_material
        )

        # 4. Back section (behind the hole, in the hole Z-range)
        back_half_y = (block_half[1] - hole_half_size) / 2
        back_center_y = block_half[1] - back_half_y
        builder.add_box_collision(
            pose=sapien.Pose(p=[0, back_center_y, hole_center_z]),
            half_size=[block_half[0], back_half_y, hole_half_size],
            material=block_material
        )
        builder.add_box_visual(
            pose=sapien.Pose(p=[0, back_center_y, hole_center_z]),
            half_size=[block_half[0], back_half_y, hole_half_size],
            material=wood_material
        )

        # Hole runs fully through the block in local X.

        # Rotate block so the nail channel runs along world +Y.
        block_rot = Rotation.from_euler("z", 90, degrees=True).as_quat()  # [x, y, z, w]
        block_quat = [block_rot[3], block_rot[0], block_rot[1], block_rot[2]]  # [w, x, y, z]
        builder.set_initial_pose(sapien.Pose(p=block_center.tolist(), q=block_quat))
        # self.block = builder.build_static(name="wood_block")
        return builder

    def _load_nails(self):
        """Load nails that can only move in Y axis (horizontally)"""
        nail_material = physx.PhysxMaterial(
            static_friction=2.5,
            dynamic_friction=2.0,
            restitution=0.0,
        )
        nail_angle = -90 if self._nail_dir > 0 else 90
        nail_rot = Rotation.from_euler('x', nail_angle, degrees=True).as_quat()  # [x, y, z, w]
        nail_quat = [nail_rot[3], nail_rot[0], nail_rot[1], nail_rot[2]]  # [w, x, y, z]

        nail_visual = sapien.render.RenderMaterial(
            base_color=[0.75, 0.75, 0.75, 1.0],
            metallic=0.2,
            roughness=0.4,
        )
        mesh_scale = NAIL_HEIGHT / 2.0
        builder = self.scene.create_actor_builder()
        builder.add_multiple_convex_collisions_from_file(
            filename=str(self._nail_mesh_path),
            scale=[mesh_scale] * 3,
            material=nail_material,
            density=5000,
            decomposition="coacd",
        )
        builder.add_visual_from_file(
            str(self._nail_mesh_path),
            scale=[mesh_scale] * 3,
            material=nail_visual,
        )
    
        builder.set_initial_pose(sapien.Pose(q=nail_quat))
        return builder

    def _load_hammer(self):

        hammer_material = physx.PhysxMaterial(
            static_friction=3.0,
            dynamic_friction=2.0,
            restitution=0.0
        )
        initial_pose = sapien.Pose(
            p=self._hammer_rest_center.tolist(),
            q=self._hammer_orientation.tolist()
        )
        builder = self.get_ycb_asset_builder(ycb_id=YCB_HAMMER_ID, object_type="MO", physical_material=hammer_material, density=2000, initial_pose=initial_pose)
        return builder

    def _apply_hammer_z_rotation(self, degrees: float):
        angle = np.deg2rad(degrees) * 0.5
        z_quat = torch.tensor(
            [np.cos(angle), 0.0, 0.0, np.sin(angle)], dtype=torch.float32
        )
        w1, x1, y1, z1 = z_quat
        w2, x2, y2, z2 = self._hammer_orientation
        self._hammer_orientation = torch.tensor(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=torch.float32,
        )

    def _apply_hammer_x_rotation(self, degrees: float):
        angle = np.deg2rad(degrees) * 0.5
        x_quat = torch.tensor(
            [np.cos(angle), np.sin(angle), 0.0, 0.0], dtype=torch.float32
        )
        w1, x1, y1, z1 = x_quat
        w2, x2, y2, z2 = self._hammer_orientation
        self._hammer_orientation = torch.tensor(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=torch.float32,
        )

    def _choose_hammer_orientation(self):

        # Use underlying SAPIEN object directly since GPU sim is not yet initialized
        raw_hammer = self.hammer._objs[0]
        render_comp = raw_hammer.find_component_by_type(
            sapien.render.RenderBodyComponent
        )
        if render_comp is None:
            return

        candidates = [
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
            torch.tensor([0.7071068, 0.7071068, 0.0, 0.0], dtype=torch.float32),
            torch.tensor([0.7071068, 0.0, 0.7071068, 0.0], dtype=torch.float32),
            torch.tensor([0.7071068, 0.0, 0.0, 0.7071068], dtype=torch.float32),
        ]

        best_q = candidates[0]
        best_extent = None
        for q in candidates:
            raw_hammer.set_pose(sapien.Pose(p=[0.0, 0.0, 0.0], q=q.tolist()))
            aabb = render_comp.compute_global_aabb_tight()
            z_extent = float(aabb[1, 2] - aabb[0, 2])
            if best_extent is None or z_extent < best_extent:
                best_extent = z_extent
                best_q = q
        self._hammer_orientation = best_q

    def _update_hammer_rest_height(self):

        # Use underlying SAPIEN object directly since GPU sim is not yet initialized
        raw_hammer = self.hammer._objs[0]
        render_comp = raw_hammer.find_component_by_type(
            sapien.render.RenderBodyComponent
        )
        if render_comp is None:
            return

        raw_hammer.set_pose(
            sapien.Pose(p=[0.0, 0.0, 0.0], q=self._hammer_orientation.tolist())
        )
        aabb = render_comp.compute_global_aabb_tight()
        # Align the hammer bottom with the table surface (z=0) with a small clearance.
        self._hammer_rest_center[2] = float(-aabb[0, 2] + 1e-3)

    def _lock_table(self):
        components = getattr(self.table, "components", None)
        if components is None:
            getter = getattr(self.table, "get_components", None)
            components = getter() if getter is not None else []
        for comp in components:
            if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
                comp.kinematic = True
                comp.disable_gravity = True
                comp.linear_velocity = np.zeros(3, dtype=np.float32)
                comp.angular_velocity = np.zeros(3, dtype=np.float32)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)

            base_centers = self._nail_rest_centers.to(self.device)
            raise_ranges = self._nail_raise_ranges.to(self.device)

            centers = base_centers.unsqueeze(0).repeat(b, 1, 1).clone()
            # Randomize slightly in Y direction (pull nail back a bit from initial position)
            lift = torch.rand((b, 1), device=self.device)
            centers[..., 1] = centers[..., 1] - self._nail_dir * lift * raise_ranges.unsqueeze(0)

            # Horizontal nail orientation along the selected Y direction
            nail_angle = -90 if self._nail_dir > 0 else 90
            nail_rot = Rotation.from_euler('x', nail_angle, degrees=True).as_quat()  # [x, y, z, w]
            nail_quat = torch.tensor([nail_rot[3], nail_rot[0], nail_rot[1], nail_rot[2]], device=self.device, dtype=torch.float32).repeat(b, 1)
            nail_pose = Pose.create_from_pq(
                p=centers[:, 0, :], q=nail_quat
            )
            self.nails[0].set_pose(nail_pose)
            self.nails[0].set_linear_velocity(torch.zeros((b, 3), device=self.device))
            self.nails[0].set_angular_velocity(torch.zeros((b, 3), device=self.device))
            # 

            # Randomize hammer position
            # hammer_pos = self._hammer_rest_center.to(self.device).unsqueeze(0).repeat(b, 1)
            # # Add randomization: ±0.03m in X, ±0.03m in Y
            # hammer_pos[:, 0] += (torch.rand(b, device=self.device) - 0.5) * 0.06
            # hammer_pos[:, 1] += (torch.rand(b, device=self.device) - 0.5) * 0.06
            hammer_pos = torch.zeros((b, 3), device=self.device)
            hammer_pos[:, 0:2] = self._hammer_region.sample_xy(b, device=self.device)
            hammer_pos[:, 2] = self._hammer_rest_center[2]


            # Randomize hammer yaw (rotation around Z axis) between 0 and 90 degrees
            base_hammer_q = self._hammer_orientation.to(self.device).unsqueeze(0).repeat(b, 1)
            # Random angle between 0 and 90 degrees (0 to pi/2 radians)
            random_yaw_angle = torch.rand(b, device=self.device) * (torch.pi / 2)
            # Quaternion for Z rotation: [cos(θ/2), 0, 0, sin(θ/2)] in wxyz format
            random_yaw_q = torch.zeros((b, 4), device=self.device)
            random_yaw_q[:, 0] = torch.cos(random_yaw_angle / 2)  # w
            random_yaw_q[:, 3] = torch.sin(random_yaw_angle / 2)  # z
            hammer_q = quaternion_multiply(base_hammer_q, random_yaw_q)
            hammer_pose = Pose.create_from_pq(p=hammer_pos, q=hammer_q)
            self.hammer.set_pose(hammer_pose)
            self.hammer.set_linear_velocity(torch.zeros((b, 3), device=self.device))
            self.hammer.set_angular_velocity(torch.zeros((b, 3), device=self.device))

            self.initialize_episode_hook(env_idx, mo_pose=self.nail.pose, ro_pose=self.hammer.pose)

    def evaluate(self):
        # Check Y position instead of Z for horizontal nail
        nail_centers_y = torch.stack(
            [nail.pose.p[..., 1] for nail in self.nails], dim=1
        )
        # Success when nail has been driven far enough in +Y direction
        success = (
            self._nail_dir
            * (nail_centers_y - self._nail_success_threshold.to(self.device))
            >= 0
        ).all(dim=1)
        return {"success": success}
