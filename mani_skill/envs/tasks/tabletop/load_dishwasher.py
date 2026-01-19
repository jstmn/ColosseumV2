import logging
from typing import Any, Dict, Union

import numpy as np
import sapien
import sapien.render
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs import Articulation, Link
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from sapien.physx import PhysxMaterial

try:
    import coacd

    _HAS_COACD = True
except ImportError:
    _HAS_COACD = False

logger = logging.getLogger(__name__)

DISHWASHER_COLLISION_BIT = 30


@register_env("LoadDishwasher-v1", max_episode_steps=150)
class LoadDishwasherEnv(BaseEnv):
    """
    **Task Description:**
    Open the dishwasher drawer (pull out the sliding shelf), then place the plate onto the shelf.

    **Randomizations:**
    - The plate starts randomly on the table near the robot.
    - The dishwasher position is slightly randomized on the tabletop.

    **Success Conditions:**
    - The dishwasher drawer is open (shelf pulled out).
    - The plate is on the shelf inside the dishwasher.
    - The plate is released and static.
    """

    agent: Union[Panda, Fetch]

    # Dishwasher URDF path
    _dishwasher_urdf_path = PACKAGE_ASSET_DIR / "load_dishwasher/12085/mobility.urdf"

    # Plate mesh paths (reuse from PlaceDishInRack)
    _plate_visual_mesh_path = (
        PACKAGE_ASSET_DIR / "dish_into_rack/white_ceramic_serving_bowl.glb"
    )
    _plate_mesh_source_radius = 0.5
    _plate_mesh_source_height = 0.2494586706161499
    _plate_mesh_flat_quat = [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]

    # Plate geometry parameters
    _plate_outer_radius = 0.09
    _plate_density = 300.0
    _plate_total_height = (
        _plate_mesh_source_height * (_plate_outer_radius / _plate_mesh_source_radius)
    )
    _plate_spawn_buffer = 0.002

    # Dishwasher scale (the original model is large, scale it down)
    _dishwasher_scale = 0.35

    # Minimum open fraction for the drawer to be considered "open"
    min_open_frac = 0.7

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq=200,
            control_freq=20,
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**23, max_rigid_patch_count=2**17
            ),
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, -0.25, 0.35], target=[0.0, 0.0, 0.05])
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
        # Camera behind robot at angle looking towards dishwasher
        pose = sapien_utils.look_at([-1.0, -0.6, 0.7], [0.0, 0.2, 0.1])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1.0, near=0.01, far=100
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

        self.table_height_offset = 0.0

        # Load dishwasher
        self._load_dishwasher()

        # Load plate
        self.plate = self._build_plate()
        self.plate.set_linear_damping(5.0)
        self.plate.set_angular_damping(8.0)

    def _load_dishwasher(self):
        """Load the dishwasher as an articulated object from URDF."""
        loader = self.scene.create_urdf_loader()
        loader.scale = self._dishwasher_scale
        loader.fix_root_link = True
        loader.name = "dishwasher-0"

        self._dishwashers = []
        sapien.set_log_level("off")
        dishwasher = loader.load(str(self._dishwasher_urdf_path))
        sapien.set_log_level("warn")

        if dishwasher is None:
            raise ValueError(f"Failed to load dishwasher URDF from {self._dishwasher_urdf_path}")

        self.remove_from_state_dict_registry(dishwasher)

        # Set collision group
        for link in dishwasher.links:
            link.set_collision_group_bit(
                group=2, bit_idx=DISHWASHER_COLLISION_BIT, bit=1
            )
        self._dishwashers.append(dishwasher)

        # Merge into single articulation
        self.dishwasher = Articulation.merge(self._dishwashers, name="dishwasher")
        self.add_to_state_dict_registry(self.dishwasher)

        # Find the shelf link (link_1 with prismatic joint)
        shelf_links = []
        for link, joint in zip(dishwasher.links, dishwasher.joints):
            # joint.type can be a string or list depending on wrapper
            jtype = joint.type[0] if hasattr(joint.type, "__getitem__") and not isinstance(joint.type, str) else joint.type
            if jtype == "prismatic":
                shelf_links.append(link)

        if not shelf_links:
            raise ValueError(f"No prismatic joint found in dishwasher model. Available: {[(j.name, j.type) for j in dishwasher.joints]}")

        self.shelf_link = Link.merge(shelf_links, name="shelf_link")

        # Store the prismatic joint for later use
        for joint in dishwasher.joints:
            jtype = joint.type[0] if hasattr(joint.type, "__getitem__") and not isinstance(joint.type, str) else joint.type
            if jtype == "prismatic":
                self._drawer_joint = joint
                break

    def _build_plate(self):
        """Build the plate from the ceramic bowl mesh."""
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

        if _HAS_COACD:
            builder.add_multiple_convex_collisions_from_file(
                filename=str(self._plate_visual_mesh_path),
                scale=[collision_scale, collision_scale, collision_scale],
                pose=mesh_pose,
                material=physical_material,
                density=self._plate_density,
                decomposition="coacd",
            )
        else:
            logger.warning(
                "coacd not installed; falling back to nonconvex collision for plate."
            )
            builder.add_nonconvex_collision_from_file(
                filename=str(self._plate_visual_mesh_path),
                scale=[collision_scale, collision_scale, collision_scale],
                pose=mesh_pose,
                material=physical_material,
                density=self._plate_density,
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

    def _after_reconfigure(self, options):
        # Compute target qpos for drawer open
        target_qlimits = self.dishwasher.get_qlimits()
        qmin, qmax = target_qlimits[:, 0, 0], target_qlimits[:, 0, 1]
        self.target_qpos = qmin + (qmax - qmin) * self.min_open_frac

    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        device = self.device
        with torch.device(device):
            b = len(env_idx)

            # Initialize table scene
            self.table_scene.initialize(env_idx)

            # Get table top z
            table_p_arr = np.asarray(self.table_scene.table.pose.p).ravel()
            table_z = float(table_p_arr[-1])
            table_top_z = table_z + float(self.table_scene.table_height)

            # Position dishwasher on table (to the right side)
            dishwasher_pos = torch.zeros((b, 3), device=device)
            dishwasher_pos[:, 0] = 0.0  # In front of robot
            dishwasher_pos[:, 1] = 0.2  # To the right
            dishwasher_pos[:, 2] = table_top_z + 0.15  # On table surface

            # Rotate dishwasher to face the robot (front facing -Y direction)
            # The URDF has a rotation built in, we want the door to face the robot
            dishwasher_quat = torch.zeros((b, 4), device=device)
            dishwasher_quat[:, 0] = 1.0  # w=1, identity rotation

            dishwasher_pose = Pose.create_from_pq(p=dishwasher_pos, q=dishwasher_quat)
            self.dishwasher.set_pose(dishwasher_pose)

            # Start with drawer slightly open (15% open) so gripper can hook the lip
            qlimits = self.dishwasher.get_qlimits()
            qmin, qmax = qlimits[env_idx, :, 0], qlimits[env_idx, :, 1]
            initial_qpos = qmin + (qmax - qmin) * 0.15  # 15% open
            self.dishwasher.set_qpos(initial_qpos)
            self.dishwasher.set_qvel(self.dishwasher.qpos[env_idx] * 0)

            # Position plate on table (to the left side, near robot)
            plate_x = -0.35 + (torch.rand(b, device=device) - 0.5) * 0.1
            plate_y = -0.15 + (torch.rand(b, device=device) - 0.5) * 0.1

            xyz = torch.zeros((b, 3), device=device)
            xyz[:, 0] = plate_x
            xyz[:, 1] = plate_y
            plate_half_height = self._plate_total_height / 2.0
            xyz[:, 2] = table_top_z + plate_half_height + self._plate_spawn_buffer

            # Random plate yaw
            plate_yaw = torch.rand(b, device=device) * 2 * np.pi
            flat_quat = torch.zeros((b, 4), device=device)
            flat_quat[:, 0] = torch.cos(plate_yaw / 2)
            flat_quat[:, 3] = torch.sin(plate_yaw / 2)

            plate_pose = Pose.create_from_pq(p=xyz, q=flat_quat)
            self.plate.set_pose(plate_pose)

            # Let physics settle
            for _ in range(50):
                self.scene.step()

            # Reset plate to stable pose
            xyz[:, 2] = table_top_z + plate_half_height + self._plate_spawn_buffer
            plate_pose = Pose.create_from_pq(p=xyz, q=flat_quat)
            self.plate.set_pose(plate_pose)

            # Zero velocities
            zero_velocity = torch.zeros((b, 3), device=device)
            self.plate.set_linear_velocity(zero_velocity)
            self.plate.set_angular_velocity(zero_velocity)

    def evaluate(self):
        # Check if drawer is open enough
        qpos = self.dishwasher.qpos
        if qpos.ndim > 1:
            qpos = qpos.squeeze(-1)

        qlimits = self.dishwasher.get_qlimits()
        qmin, qmax = qlimits[:, 0, 0], qlimits[:, 0, 1]
        open_frac = (qpos - qmin) / (qmax - qmin + 1e-6)
        drawer_open = open_frac >= self.min_open_frac

        # Check if plate is on/near the dishwasher
        plate_pos = self.plate.pose.p
        dishwasher_pos = self.dishwasher.pose.p

        # Plate should be within the dishwasher bounds (roughly)
        plate_near_dishwasher_x = torch.abs(plate_pos[:, 0] - dishwasher_pos[:, 0]) < 0.2
        plate_near_dishwasher_y = torch.abs(plate_pos[:, 1] - dishwasher_pos[:, 1]) < 0.2
        plate_above_dishwasher = plate_pos[:, 2] > dishwasher_pos[:, 2] - 0.1
        plate_not_too_high = plate_pos[:, 2] < dishwasher_pos[:, 2] + 0.2
        plate_on_shelf = plate_near_dishwasher_x & plate_near_dishwasher_y & plate_above_dishwasher & plate_not_too_high

        # Plate should be flat (not vertical)
        rot_mats = quaternion_to_matrix(self.plate.pose.q)
        plate_norm = rot_mats[..., 2]
        plate_flat = torch.abs(plate_norm[..., 2]) >= 0.8  # Z component of normal should be high

        # Check if released and static
        is_grasped = self.agent.is_grasping(self.plate)
        is_static = self.plate.is_static(lin_thresh=0.02, ang_thresh=0.4)

        success = drawer_open & plate_on_shelf & plate_flat & is_static & ~is_grasped

        return {
            "success": success,
            "drawer_open": drawer_open,
            "plate_on_shelf": plate_on_shelf,
            "plate_flat": plate_flat,
            "is_static": is_static,
            "is_grasped": is_grasped,
            "open_frac": open_frac,
        }

    def _get_obs_extra(self, info: Dict):
        plate_pose = self.plate.pose
        dishwasher_pose = self.dishwasher.pose
        obs = {
            "plate_pos": plate_pose.p,
            "plate_quat": plate_pose.q,
            "dishwasher_pos": dishwasher_pose.p,
            "dishwasher_quat": dishwasher_pose.q,
            "dishwasher_qpos": self.dishwasher.qpos,
        }
        if "state" in self.obs_mode:
            obs.update(
                plate_to_dishwasher=plate_pose.p - dishwasher_pose.p,
                tcp_to_plate=plate_pose.p - self.agent.tcp_pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict) -> torch.Tensor:
        """Compute dense reward for the task."""
        # Phase 1: Open the drawer
        open_frac = info["open_frac"]
        drawer_reward = open_frac * 2.0

        # Phase 2: Place plate on dishwasher (only reward when drawer is open)
        plate_pos = self.plate.pose.p
        dishwasher_pos = self.dishwasher.pose.p
        dist_to_dishwasher = torch.linalg.norm(plate_pos - dishwasher_pos, axis=1)
        placement_reward = torch.where(
            info["drawer_open"],
            (1.0 - torch.tanh(2.0 * dist_to_dishwasher)) * 2.0,
            torch.zeros_like(dist_to_dishwasher),
        )

        # Release reward
        release_reward = torch.where(
            info["plate_on_shelf"] & ~info["is_grasped"],
            torch.ones_like(dist_to_dishwasher),
            torch.zeros_like(dist_to_dishwasher),
        )

        # Success bonus
        success_reward = info["success"].float() * 5.0

        reward = drawer_reward + placement_reward + release_reward + success_reward
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict) -> torch.Tensor:
        return self.compute_dense_reward(obs, action, info) / 10.0
