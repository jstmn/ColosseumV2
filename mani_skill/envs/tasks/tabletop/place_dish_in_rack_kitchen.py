from typing import Any, Dict, Union

import numpy as np
import sapien
import sapien.render
import torch
from sapien.physx import PhysxMaterial

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.kitchen import KitchenSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


@register_env("PlaceDishInRackKitchen-v1", max_episode_steps=100)
class PlaceDishInRackKitchenEnv(BaseEnv):
    """
    **Task Description:**
    Pick up the plate and place it vertically into the upright slots of the dish rack.
    This version has a realistic kitchen backdrop with counter, backsplash, and cabinets.

    **Randomizations:**
    - The plate starts randomly on the counter near the robot.
    - The dish rack pose is randomized slightly on the counter.

    **Success Conditions:**
    - The plate is upright and centered inside the dish rack while the robot releases it.
    """

    agent: Union[Panda, Fetch]

    _rack_mesh_path = PACKAGE_ASSET_DIR / "dish_into_rack/dish_rack_with_connectors.stl"
    _plate_visual_mesh_path = "/home/ashvin/Downloads/hollow-cylinder-with-floor-2025-11-05-03-03-07.stl"

    _rack_scale = 0.0015

    # Plate geometry parameters (meters)
    _plate_outer_radius = 0.04
    _plate_inner_radius = 0.02
    _plate_base_thickness = 0.005
    _plate_rim_height = 0.02
    _plate_density = 300.0
    _plate_total_height = _plate_base_thickness + _plate_rim_height
    _plate_extent = np.array(
        [_plate_outer_radius * 2, _plate_outer_radius * 2, _plate_total_height]
    )
    _plate_spawn_buffer = 0.002

    _rack_extent = np.array([0.12060600281, 0.16782440567, 0.085])
    # STL is now centered at origin, no offset needed

    _plate_goal_offset = np.array([0.0, 0.0, 0.2])
    # Position rack more forward on counter to avoid backsplash collision
    _rack_position = np.array([-0.2, 0.05, 0])  # Moved forward from -0.1 to 0.05

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**23, max_rigid_patch_count=2**17
            )
        )

    @property
    def _default_sensor_configs(self):
        # Camera positioned to view the kitchen counter
        pose = sapien_utils.look_at(eye=[0.5, -0.5, 1.3], target=[0.0, 0.0, 0.95])
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
        # Render camera with better view of kitchen scene
        pose = sapien_utils.look_at([0.8, -0.6, 1.2], [0.0, 0.0, 0.95])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_agent(self, options: Dict):
        # Position robot higher to match counter height (0.9m)
        # Robot base should be slightly below counter to allow arm reach
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0.7]))

    def _get_obs_agent(self):
        obs = super()._get_obs_agent()
        for key in ("world__T__ee", "world__T__root"):
            value = obs.get(key, None)
            if isinstance(value, torch.Tensor) and value.ndim == 3:
                obs[key] = value.reshape(value.shape[0], -1)
        return obs

    def _load_scene(self, options: Dict):
        # Use kitchen scene instead of table scene
        self.kitchen_scene = KitchenSceneBuilder(
            env=self,
            robot_init_qpos_noise=self.robot_init_qpos_noise,
            counter_length=1.2,
            counter_width=0.6,
            counter_height=0.9,
            add_backsplash=True,
            add_upper_cabinet=True,
        )
        self.kitchen_scene.build()

        self.plate = self._build_plate()
        self.dish_rack = self._build_rack()

    def _build_plate(self):
        """Build a plate with a raised rim using simple analytic geometry."""
        builder = self.scene.create_actor_builder()

        physical_material = PhysxMaterial(
            static_friction=20.0,
            dynamic_friction=20.0,
            restitution=0.0
        )

        density = self._plate_density

        # Use STL for collision
        collision_scale = 0.0025
        builder.add_nonconvex_collision_from_file(
            filename=str(self._plate_visual_mesh_path),
            scale=[collision_scale, collision_scale, collision_scale],
            pose=sapien.Pose(),
            material=physical_material,
            density=density,
        )

        # Visual mesh
        plate_visual_material = sapien.render.RenderMaterial(
            base_color=[0.98, 0.95, 0.90, 1.0],
            specular=0.6,
            roughness=0.25,
            metallic=0.0,
        )

        visual_scale = 0.0025
        builder.add_visual_from_file(
            filename=str(self._plate_visual_mesh_path),
            scale=[visual_scale, visual_scale, visual_scale],
            pose=sapien.Pose(p=[0, 0, 0.001]),
            material=plate_visual_material,
        )

        builder.initial_pose = sapien.Pose()
        return builder.build(name="plate")

    def _build_rack(self):
        builder = self.scene.create_actor_builder()

        rack_width = self._rack_extent[0]
        rack_depth = self._rack_extent[1]
        rack_height = self._rack_extent[2]

        # Base plate
        builder.add_box_collision(
            half_size=[rack_width / 2, rack_depth / 2, 0.005],
            pose=sapien.Pose(p=[0, 0, 0.005])
        )

        # Back wall
        builder.add_box_collision(
            half_size=[rack_width / 2, 0.005, rack_height / 2],
            pose=sapien.Pose(p=[0, -rack_depth / 2, rack_height / 2])
        )

        # Side walls
        builder.add_box_collision(
            half_size=[0.005, rack_depth / 2, rack_height / 2],
            pose=sapien.Pose(p=[-rack_width / 2, 0, rack_height / 2])
        )
        builder.add_box_collision(
            half_size=[0.005, rack_depth / 2, rack_height / 2],
            pose=sapien.Pose(p=[rack_width / 2, 0, rack_height / 2])
        )

        # Vertical dividers
        divider_thickness = 0.003
        num_dividers = 3
        for i in range(num_dividers):
            x_pos = -rack_width / 2 + (i + 1) * rack_width / (num_dividers + 1)
            builder.add_box_collision(
                half_size=[divider_thickness, rack_depth / 2, rack_height / 2],
                pose=sapien.Pose(p=[x_pos, 0, rack_height / 2])
            )

        # Visual mesh (now centered at origin)
        builder.add_visual_from_file(
            filename=str(self._rack_mesh_path),
            scale=[self._rack_scale] * 3,
        )
        builder.initial_pose = sapien.Pose()
        return builder.build_kinematic(name="dish_rack")

    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        device = self.device
        with torch.device(device):
            b = len(env_idx)
            self.kitchen_scene.initialize(env_idx)

            # Get counter top height
            counter_top_z = float(self.kitchen_scene.counter_top_height)

            # Place plate flat on counter - more forward to avoid backsplash
            xyz = torch.zeros((b, 3), device=device)
            xyz[:, 0] = -0.28
            xyz[:, 1] = -0.10  # More forward, was -0.26
            plate_half_height = self._plate_total_height / 2.0
            xyz[:, 2] = counter_top_z + plate_half_height + self._plate_spawn_buffer

            flat_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(b, 1)

            plate_pose = Pose.create_from_pq(p=xyz, q=flat_quat)
            self.plate.set_pose(plate_pose)

            # Place rack on counter
            rack_pos = torch.zeros((b, 3), device=device)
            rack_pos[:] = torch.tensor(self._rack_position, device=device)
            rack_pos[:, 2] = counter_top_z + float(self._rack_extent[2]) / 2.0
            rack_pose = Pose.create_from_pq(p=rack_pos)
            self.dish_rack.set_pose(rack_pose)

            # Let plate settle
            for _ in range(50):
                self.scene.step()

            zero_velocity = torch.zeros((b, 3), device=device)
            self.plate.set_linear_velocity(zero_velocity)
            self.plate.set_angular_velocity(zero_velocity)

    def evaluate(self):
        plate_pos = self.plate.pose.p
        rack_pos = self.dish_rack.pose.p
        target_offset = torch.tensor(
            self._plate_goal_offset, device=self.device, dtype=plate_pos.dtype
        )
        goal_pos = rack_pos + target_offset
        plate_to_goal = torch.linalg.norm(plate_pos - goal_pos, dim=1)

        rot_mats = quaternion_to_matrix(self.plate.pose.q)
        plate_norm = rot_mats[..., 2]
        plate_vertical = torch.abs(plate_norm[..., 2]) <= 0.35

        is_grasped = self.agent.is_grasping(self.plate)
        is_static = self.plate.is_static(lin_thresh=0.02, ang_thresh=0.4)

        counter_top_z = float(self.kitchen_scene.counter_top_height)
        above_counter = plate_pos[:, 2] > counter_top_z - 0.01

        close_to_rack = plate_to_goal <= 0.08
        success = close_to_rack & plate_vertical & (~is_grasped) & is_static & above_counter

        return {
            "success": success,
            "plate_close_to_goal": close_to_rack,
            "plate_vertical": plate_vertical,
            "is_static": is_static,
            "is_grasped": is_grasped,
            "above_counter": above_counter,
        }

    def _get_obs_extra(self, info: Dict):
        plate_pose = self.plate.pose
        rack_pose = self.dish_rack.pose
        obs = {
            "plate_pos": plate_pose.p,
            "plate_quat": plate_pose.q,
            "rack_pos": rack_pose.p,
            "rack_quat": rack_pose.q,
        }
        if "state" in self.obs_mode:
            obs.update(
                plate_to_goal=plate_pose.p - rack_pose.p,
                tcp_to_plate=plate_pose.p - self.agent.tcp_pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict) -> torch.Tensor:
        plate_pos = self.plate.pose.p
        rack_pos = self.dish_rack.pose.p
        target_offset = torch.tensor(
            self._plate_goal_offset, device=self.device, dtype=plate_pos.dtype
        )
        goal_pos = rack_pos + target_offset
        plate_to_goal_dist = torch.linalg.norm(plate_pos - goal_pos, dim=1)

        reaching_reward = 1.0 - torch.tanh(5.0 * plate_to_goal_dist)

        rot_mats = quaternion_to_matrix(self.plate.pose.q)
        plate_norm = rot_mats[..., 2]
        vertical_alignment = torch.abs(plate_norm[..., 2])
        orientation_reward = 1.0 - vertical_alignment

        is_grasped = self.agent.is_grasping(self.plate)
        close_to_goal = plate_to_goal_dist <= 0.08
        release_reward = torch.where(
            close_to_goal,
            torch.where(is_grasped, torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device)),
            torch.tensor(0.0, device=self.device)
        )

        success = info["success"].float()
        success_reward = success * 5.0

        reward = reaching_reward + orientation_reward + release_reward + success_reward
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict) -> torch.Tensor:
        return self.compute_dense_reward(obs, action, info) / 8.0
