from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


ASSET_DIR = Path(__file__).resolve().parents[3] / "assets" / "custom" / "hammer_scene"
HAMMER_ASSET_DIR = Path(__file__).resolve().parents[3] / "assets" / "custom" / "hammer"
HAMMER_MESH_PATH = Path("/home/ashvin/Downloads/obj_models/048_hammer/nontextured.stl")
BLOCK_MESH_PATH = ASSET_DIR / "block.glb"
NAIL_MESH_PATH = ASSET_DIR / "nail.glb"
HAMMER_SCALE = 0.001  # Scale for the STL model (YCB models are in mm)
NAIL_HEIGHT = 0.086


@dataclass
class NailSpec:
    name: str
    rest_center: np.ndarray
    success_center_x: float  # Changed from z to x for sideways hammering
    raise_range: float


@register_env("HammerNail-v1", max_episode_steps=200)
class HammerNailEnv(BaseEnv):
    """A tabletop hammering task with a single nail and a simple wooden block.

    **Task**: Pick up the hammer and drive the nail horizontally (sideways) into the
    vertical block until the nail center passes a target depth.

    Assets:
    - `hammer_scene/hammer.glb`: textured hammer mesh positioned flat beside the nail
    - `hammer_scene/nail.glb`: cylindrical nail with a rounded head (oriented horizontally)
    - `hammer_scene/block.glb`: wooden block positioned vertically to receive the nail from the side."""

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

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
        # Rotated 90° so the nail can be hammered horizontally into the left side
        self._block_half_size = torch.tensor([0.015, 0.03, 0.06], dtype=torch.float32)  # Thin in X, normal Y, tall in Z
        self._block_center = torch.tensor([0.15, -0.10, 0.06], dtype=torch.float32)  # Positioned to the right
        self._block_left_x = float(self._block_center[0] - self._block_half_size[0])  # Left face where nail enters

        # Nail starts sticking out horizontally (pointing in +X direction towards block)
        nail_initial_insert = 0.02  # Start with nail less inserted (more visible)
        nail_target_insert = 0.05   # Drive it 3cm deeper into block (in +X direction)
        nail_rest_center = np.array(
            [
                self._block_left_x - NAIL_HEIGHT / 2 + nail_initial_insert,  # Nail pointing towards block
                self._block_center[1].item(),
                self._block_center[2].item(),  # Center height of block
            ],
            dtype=np.float32,
        )
        self._nail_success_center_x = float(
            self._block_left_x - NAIL_HEIGHT / 2 + nail_target_insert  # Target X position when driven in
        )

        self._nail_specs = [
            NailSpec(
                name="nail",
                rest_center=nail_rest_center,
                success_center_x=self._nail_success_center_x,
                raise_range=0.005,
            )
        ]

        # Simple box hammer design: rotated 270° so handle at -X (left), head at +X (right)
        # Perfect for striking horizontally in +X direction
        handle_size = np.array([0.04, 0.15, 0.04])  # Handle: thicker for better grip (4cm x 15cm x 4cm)
        head_size = np.array([0.06, 0.09, 0.06])    # Head: larger for better collision (6cm x 9cm x 6cm)
        handle_density = 2000  # kg/m³
        head_density = 6173   # kg/m³ (head volume 0.000324 m³ * 6173 ≈ 2.0 kg)

        # Position hammer to the left of the nail, at same height, ready for sideways striking
        self._hammer_rest_center = torch.tensor(
            [-0.15, -0.10, 0.06], dtype=torch.float32  # Left of nail, same Y and Z as nail
        )
        # Rotate 270 degrees around Z axis (180 + 90)
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_euler('z', 270, degrees=True).as_quat()  # [x, y, z, w]
        self._hammer_orientation = torch.tensor(
            [rot[3], rot[0], rot[1], rot[2]], dtype=torch.float32  # [w, x, y, z]
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
                enable_ccd=True,                # Enable CCD globally
            )
        )

    @property
    def _default_sensor_configs(self):
        # View from front-right to see sideways hammering action
        pose = sapien_utils.look_at(eye=[0.3, 0.5, 0.3], target=[0.05, -0.10, 0.06])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 3,
                near=0.01,
                far=10,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        # Better view for sideways hammering - from front-right angle
        pose = sapien_utils.look_at([0.4, 0.6, 0.35], [0.05, -0.10, 0.06])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1.0, near=0.01, far=10
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.55, 0.0, 0.0]))

    def _hide_actor_from_cameras(self, actor):
        """Hide an actor from camera rendering (observations) while keeping physics.

        This is useful when you want an object to participate in physics simulation
        but not appear in visual observations (e.g., invisible collision guides,
        hidden goal objects, etc.).

        Example usage:
            self._hide_actor_from_cameras(self.nail_platform)
        """
        # Method 1: Hide all visual components
        # This removes the actor from all camera rendering
        actor.hide_visual()

        # Method 2: For partial hiding or more control, you can iterate components:
        # for comp in actor.components:
        #     if isinstance(comp, sapien.render.RenderBodyComponent):
        #         comp.set_visibility(0.0)  # Make invisible

        # Note: The actor will still have collision and participate in physics,
        # but won't appear in RGB, depth, or segmentation camera observations.

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self._load_block()
        self._load_nails()
        self._load_hammer()
        self._lock_table()

    def _get_obs_agent(self):
        obs = super()._get_obs_agent()
        for key in ("world__T__ee", "world__T__root"):
            value = obs.get(key, None)
            if isinstance(value, torch.Tensor) and value.ndim == 3:
                obs[key] = value.reshape(value.shape[0], -1)
        return obs

    def _load_block(self):
        """Create a vertical wooden block and a thin collision platform for the nail"""
        import sapien.physx as physx
        import sapien.render

        # Visual block (no collision) - now vertical
        builder = self.scene.create_actor_builder()
        block_half = self._block_half_size.cpu().numpy()
        builder.add_box_visual(
            pose=sapien.Pose(p=[0, 0, 0]),
            half_size=block_half,
            material=sapien.render.RenderMaterial(
                base_color=[0.6, 0.4, 0.2, 1.0],  # Wood color
                metallic=0.0,
                roughness=0.9
            )
        )
        builder.set_initial_pose(sapien.Pose(p=self._block_center.tolist()))
        self.block = builder.build_static(name="wood_block")

        # Thin collision platform for nail to stop against (now on the right side of block)
        # Position it beyond where the nail tip will be at target depth
        platform_builder = self.scene.create_actor_builder()
        platform_half_size = np.array([0.001, 0.02, 0.02])  # Thin wall on the right
        platform_material = physx.PhysxMaterial(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0
        )
        # Calculate where nail tip will be at target position (moving in +X direction)
        nail_target_tip_x = self._nail_success_center_x + NAIL_HEIGHT / 2
        platform_x = nail_target_tip_x + 0.005  # 5mm beyond target nail tip
        platform_builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, 0]),
            half_size=platform_half_size,
            material=platform_material
        )
        platform_builder.set_initial_pose(
            sapien.Pose(p=[
                platform_x,
                self._block_center[1].item(),
                self._block_center[2].item()
            ])
        )
        self.nail_platform = platform_builder.build_static(name="nail_platform")

        # Hide the collision platform from camera observations
        # (it's only needed for physics, not visual observation)
        self._hide_actor_from_cameras(self.nail_platform)

    def _load_nails(self):
        """Load nails that can only move in X axis (horizontally)"""
        self.nails = []
        rest_centers: List[torch.Tensor] = []
        success_thresholds: List[torch.Tensor] = []

        # Rotate nail 90° around Y axis to point in +X direction
        from scipy.spatial.transform import Rotation
        nail_rot = Rotation.from_euler('y', 90, degrees=True).as_quat()  # [x, y, z, w]
        nail_quat = [nail_rot[3], nail_rot[0], nail_rot[1], nail_rot[2]]  # [w, x, y, z]

        for spec in self._nail_specs:
            builder = self.scene.create_actor_builder()
            builder.add_visual_from_file(str(NAIL_MESH_PATH))
            builder.add_nonconvex_collision_from_file(str(NAIL_MESH_PATH), density=5000)  # Heavier nail
            # Nail now points in +X direction (horizontal)
            builder.set_initial_pose(sapien.Pose(p=[NAIL_HEIGHT / 2, 0, 0], q=nail_quat))
            nail = builder.build(name=spec.name)
            self.nails.append(nail)

            rest_center = torch.from_numpy(spec.rest_center).to(torch.float32)
            rest_centers.append(rest_center)
            success_thresholds.append(torch.tensor(spec.success_center_x, dtype=torch.float32))

        self._nail_rest_centers = torch.stack(rest_centers)
        self._nail_success_threshold = torch.stack(success_thresholds)
        self._nail_raise_ranges = torch.tensor(
            [spec.raise_range for spec in self._nail_specs], dtype=torch.float32
        )

    def _load_hammer(self):
        import sapien.physx as physx
        import sapien.render

        # Load hammer from STL file (YCB dataset model)
        builder = self.scene.create_actor_builder()

        # Try loading from STL file
        try:
            # High friction material
            hammer_material = physx.PhysxMaterial(
                static_friction=3.0,
                dynamic_friction=2.0,
                restitution=0.0
            )

            # Add visual and collision from STL file with scaling
            builder.add_visual_from_file(
                str(HAMMER_MESH_PATH),
                scale=[HAMMER_SCALE] * 3
            )
            builder.add_nonconvex_collision_from_file(
                str(HAMMER_MESH_PATH),
                scale=[HAMMER_SCALE] * 3,
                material=hammer_material,
                density=2000  # kg/m³ for reasonable mass
            )

            # Position hammer - rotated 270° around Z axis
            from scipy.spatial.transform import Rotation
            rot = Rotation.from_euler('z', 270, degrees=True).as_quat()  # [x, y, z, w]

            hammer_initial_pose = sapien.Pose(
                p=[-0.15, 0.0, 0.03],  # Much closer to robot, raised off table
                q=[rot[3], rot[0], rot[1], rot[2]]  # 270° rotation
            )
            builder.set_initial_pose(hammer_initial_pose)
            self.hammer = builder.build(name="hammer")

        except Exception as e:
            print(f"Failed to load hammer from {HAMMER_MESH_PATH}: {e}")
            print("Falling back to box primitive hammer")

            # Fallback to box primitives if STL loading fails
            handle_size = np.array([0.04, 0.15, 0.04])
            head_size = np.array([0.06, 0.09, 0.06])

            hammer_material = physx.PhysxMaterial(
                static_friction=3.0,
                dynamic_friction=2.0,
                restitution=0.0
            )

            builder_fallback = self.scene.create_actor_builder()

            handle_half = handle_size / 2
            builder_fallback.add_box_collision(
                pose=sapien.Pose(p=[0, 0, 0]),
                half_size=handle_half,
                material=hammer_material,
                density=2000
            )
            builder_fallback.add_box_visual(
                pose=sapien.Pose(p=[0, 0, 0]),
                half_size=handle_half,
                material=sapien.render.RenderMaterial(
                    base_color=[0.4, 0.25, 0.15, 1.0],
                    metallic=0.0,
                    roughness=0.8
                )
            )

            head_half = head_size / 2
            head_offset_y = handle_half[1] + head_half[1]
            head_pose = sapien.Pose(p=[0, head_offset_y, 0])
            builder_fallback.add_box_collision(
                pose=head_pose,
                half_size=head_half,
                material=hammer_material,
                density=6173
            )
            builder_fallback.add_box_visual(
                pose=head_pose,
                half_size=head_half,
                material=sapien.render.RenderMaterial(
                    base_color=[0.3, 0.3, 0.3, 1.0],
                    metallic=0.8,
                    roughness=0.3
                )
            )

            from scipy.spatial.transform import Rotation
            rot = Rotation.from_euler('z', 270, degrees=True).as_quat()

            hammer_initial_pose = sapien.Pose(
                p=[-0.15, 0.0, 0.03],
                q=[rot[3], rot[0], rot[1], rot[2]]
            )
            builder_fallback.set_initial_pose(hammer_initial_pose)
            self.hammer = builder_fallback.build(name="hammer")

    def _lock_table(self):
        components = getattr(self.table_scene.table, "components", None)
        if components is None:
            getter = getattr(self.table_scene.table, "get_components", None)
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
            # Use a collision-free start pose for the robot
            collision_free_qpos = np.array(
                [0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, np.pi / 4, 0.04, 0.04]
            )
            self.table_scene.initialize(env_idx, qpos_0=collision_free_qpos)

            base_centers = self._nail_rest_centers.to(self.device)
            raise_ranges = self._nail_raise_ranges.to(self.device)

            centers = base_centers.unsqueeze(0).repeat(b, 1, 1).clone()
            # Randomize slightly in X direction (pull nail back a bit from initial position)
            lift = torch.rand((b, len(self.nails)), device=self.device)
            centers[..., 0] = centers[..., 0] - lift * raise_ranges.unsqueeze(0)  # Subtract to pull back

            # Horizontal nail orientation (90° rotation around Y axis)
            from scipy.spatial.transform import Rotation
            nail_rot = Rotation.from_euler('y', 90, degrees=True).as_quat()  # [x, y, z, w]
            nail_quat = torch.tensor([nail_rot[3], nail_rot[0], nail_rot[1], nail_rot[2]],
                                    device=self.device, dtype=torch.float32).repeat(b, 1)

            for i, nail in enumerate(self.nails):
                pose = Pose.create_from_pq(
                    p=centers[:, i, :], q=nail_quat
                )
                nail.set_pose(pose)
                nail.set_linear_velocity(torch.zeros((b, 3), device=self.device))
                nail.set_angular_velocity(torch.zeros((b, 3), device=self.device))

                # Lock Y and Z linear movement, allow only X movement, lock all rotations
                # [lock_x, lock_y, lock_z, lock_rot_x, lock_rot_y, lock_rot_z]
                nail.set_locked_motion_axes(torch.tensor([[False, True, True, True, True, True]] * b, device=self.device))

            hammer_pos = self._hammer_rest_center.to(self.device).unsqueeze(0).repeat(b, 1)
            hammer_pose = Pose.create_from_pq(
                p=hammer_pos,
                q=self._hammer_orientation.to(self.device).unsqueeze(0).repeat(b, 1),
            )
            self.hammer.set_pose(hammer_pose)
            self.hammer.set_linear_velocity(torch.zeros((b, 3), device=self.device))
            self.hammer.set_angular_velocity(torch.zeros((b, 3), device=self.device))

    def evaluate(self):
        # Check X position instead of Z for horizontal nail
        nail_centers_x = torch.stack(
            [nail.pose.p[..., 0] for nail in self.nails], dim=1
        )
        # Success when nail has been driven far enough in +X direction
        success = (nail_centers_x >= self._nail_success_threshold.to(self.device)).all(
            dim=1
        )
        return {"success": success}

    def _get_obs_extra(self, info: Dict):
        nail_centers = torch.stack(
            [nail.pose.p for nail in self.nails], dim=1
        )  # (num_envs, num_nails, 3)
        flat_centers = nail_centers.reshape(nail_centers.shape[0], -1)
        target_depth = self._nail_success_threshold.to(self.device).unsqueeze(0).expand(
            nail_centers.shape[0], -1
        )
        hammer_pose = self.hammer.pose.raw_pose
        task_state = torch.hstack([flat_centers, target_depth, hammer_pose])
        return {"task_state": task_state}

    def compute_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ) -> torch.Tensor:
        # Current nail X positions (horizontal direction)
        nail_x = torch.stack(
            [nail.pose.p[..., 0] for nail in self.nails], dim=1
        )

        # Initial nail X positions
        nail_rest_x = self._nail_rest_centers[:, 0].to(self.device)

        # Reward based on how far the nail has been driven forward from its initial position
        # Higher current X - lower initial X = more reward (nail moves in +X direction)
        x_progress = nail_x - nail_rest_x.unsqueeze(0)  # Positive when nail moves forward in +X
        x_progress = torch.clamp(x_progress, min=0.0)  # Only reward forward movement

        # Normalize by total distance needed (from rest to success threshold)
        total_distance = self._nail_success_threshold.to(self.device) - nail_rest_x
        normalized_progress = (x_progress / total_distance.unsqueeze(0)).sum(dim=1)

        # Success bonus for fully driving the nail
        success_bonus = self.evaluate()["success"].to(self.device).float() * 2.0

        return normalized_progress + success_bonus

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ) -> torch.Tensor:
        return self.compute_dense_reward(obs, action, info) / 3.0
