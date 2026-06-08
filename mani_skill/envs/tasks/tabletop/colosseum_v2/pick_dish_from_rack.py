import logging
from typing import Dict, Union

import numpy as np
import sapien
import sapien.render
import torch
from sapien.physx import PhysxMaterial
import sapien

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import Fetch, Panda, PandaWristCam
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.envs.tasks.tabletop.colosseum_v2.colosseum_v2_core import ColosseumV2Env, DisabledPerturbationFactors, PlacementRegion


logger = logging.getLogger(__name__)


@register_env("PickDishFromRack-v1", max_episode_steps=100)
class PickDishFromRackEnv(ColosseumV2Env):
    """
    **Task Description:**
    Pick up the plate from the dish rack (where it starts vertically) and place it flat on the table.

    **Randomizations:**
    - The plate starts vertically in the dish rack.
    - The dish rack pose is randomized slightly on the tabletop.

    **Success Conditions:**
    - The plate is outside the rack's outer bounds, flat on the table, and released by the robot.
    """

    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    agent: Union[PandaWristCam, Fetch]

    _rack_mesh_path = PACKAGE_ASSET_DIR / "dish_into_rack/dish_rack_with_connectors.stl"
    _plate_visual_mesh_path = (
        PACKAGE_ASSET_DIR / "dish_into_rack/white_ceramic_serving_bowl.glb"
    )
    _plate_mesh_source_radius = 0.5  # Radius of the raw OBJ (measured once offline)
    _plate_mesh_source_height = 0.2494586706161499  # OBJ height once flattened
    # _plate_mesh_source_height = 0.1  # OBJ height once flattened
    _plate_mesh_flat_quat = [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]  # Rotate mesh so Z is the plate normal

    # Plate geometry parameters (meters) - SAME AS PLACE_DISH_IN_RACK
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

    _rack_extent = np.array([0.12060600281, 0.16782440567, 0.085])  # Normal rack size
    _plate_goal_offset = np.array([0.0, 0.0, 0.15])  # Above rack slots (same as place task)
    _rack_position = np.array([-0.1, 0.0, 0])  # Rack X position, Y will be randomized
    _plate_goal_position = np.array([-0.35, -0.15, 0])  # Reverse of place - where plate started in place task
    _plate_support_radius = 0.015
    _plate_support_height = 0.0  # No pedestal - plate flush with table

    # Randomization bounds (symmetric around robot Y=0)
    # Rack placed either right [-0.25, -0.15] or left [0.15, 0.25], avoiding center
    _rack_y_ranges = [(-0.25, -0.15), (0.15, 0.25)]

    DISABLED_PERTURBATION_FACTORS = DisabledPerturbationFactors(
        MO_size=True,
        RO_size=True,
        pose_randomization=True,
    )

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)


    @property
    def _default_sensor_configs(self):
        pose1 = sapien_utils.look_at(eye=[0.3, -0.25, 0.35], target=[0.0, 0.0, 0.05])
        pose2 = sapien_utils.look_at(eye=[-0.2, -0.15, 0.5], target=[0.0, 0.0, 0.05])
        return self.update_camera_configs([
            CameraConfig(
                "external1_camera",
                pose=pose1,
                width=224,
                height=224,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            ),
            CameraConfig(
                "external2_camera",
                pose=pose2,
                width=224,
                height=224,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ])

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.65, -0.35, 0.35], [0.05, 0.0, 0.1])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_agent(self, options: Dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))


    def _load_scene(self, options: Dict):
        self.plate = self._build_plate()
        self.dish_rack = self._build_rack()
        self._plate_gravity_enabled = False
        self.load_scene_hook(manipulation_objects=[self.plate], receiving_objects=[self.dish_rack])

    def _build_plate(self):
        """Build the plate directly from the high-fidelity ceramic bowl mesh."""

        def build_plate_fn(): 
            physical_material = PhysxMaterial(
                static_friction=20.0,
                dynamic_friction=20.0,
                restitution=0.0,
            )
            density = self._plate_density
            mesh_pose = sapien.Pose(q=self._plate_mesh_flat_quat)
            plate_visual_material = sapien.render.RenderMaterial(
                base_color=[1.0, 1.0, 1.0, 1.0],
                specular=0.4,
                roughness=0.2,
                metallic=0.0,
            )
            collision_scale = tuple[float, float, float]([self._plate_outer_radius / self._plate_mesh_source_radius] * 3)
            return self.get_glb_asset_builder(
                glb_filepath=str(self._plate_visual_mesh_path),
                object_type="MO",
                scale=collision_scale,
                physical_material=physical_material,
                density=density,
                visual_material=plate_visual_material,
                initial_pose=sapien.Pose(),
                mesh_pose=mesh_pose,
            )
        return self.add_asset_to_scene(build_plate_fn, name="plate", physics_type="dynamic", object_type="MO")

    def _build_rack(self):
        
        def rack_builder_fn():
            builder = self.scene.create_actor_builder()

            # Collision geometry matching the exact STL mesh: base + 4 vertical dividers
            # Extracted from dish_rack_with_connectors.stl after 0.0015 scaling
            # (Same as place_dish_in_rack.py for consistency)

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
            # divider_y_positions = [-0.105254, -0.046585, 0.015046, 0.074831]  # 4 dividers
            divider_y_positions = [-0.12, -0.04, 0.04, 0.12]  # 4 dividers
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
            rack_scale = 0.0015  # Rack to match
            builder.add_visual_from_file(
                filename=str(self._rack_mesh_path),
                scale=[rack_scale] * 3,
            )
            builder.initial_pose = sapien.Pose()
            return builder

        return self.add_asset_to_scene(rack_builder_fn, name="dish_rack", physics_type="kinematic", object_type="RO")

    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        device = self.device
        with torch.device(device):
            b = len(env_idx)

            # Get table top Z coordinate
            table_p_arr = np.asarray(self.table.pose.p.cpu()).ravel()

            table_z = float(table_p_arr[-1])
            table_top_z = table_z + float(self.table_scene_builders[0].table_height)

            # Position rack on table with Y randomization (symmetric around robot)
            rack_pos = torch.zeros((b, 3), device=device)
            rack_pos[:, 0] = self._rack_position[0]  # Fixed X
            # Randomly pick left or right range, then sample within that range
            for i in range(b):
                # Pick random range (0 = left/negative, 1 = right/positive)
                range_idx = torch.randint(0, 2, (1,)).item()
                y_min, y_max = self._rack_y_ranges[range_idx]
                rack_pos[i, 1] = torch.rand(1, device=device).item() * (y_max - y_min) + y_min
            rack_pos[:, 2] = table_top_z + float(self._rack_extent[2])


            rack_pose = Pose.create_from_pq(p=rack_pos)
            self.dish_rack.set_pose(rack_pose)

            # Place plate VERTICALLY between the dividers in the rack
            plate_pos = rack_pos.clone()
            # X,Y same as rack center

            # When plate is vertical (after 90° rotation around X), its radius extends in Z direction
            # Bottom of plate = center_z - plate_outer_radius
            # Position center so bottom is at table level (bottom won't clip through)
            drop_height = 0.05
            plate_pos[:, 2] = table_top_z + self._plate_outer_radius + drop_height  # Center at radius height

            # Plate vertical - normal pointing in +X direction (perpendicular to dividers)
            # Dividers run in X (left-right), plate face should be perpendicular to Y (front-back)
            # Quaternion for 90deg rotation around X axis to make plate stand vertical
            vertical_quat = torch.tensor(
                [[0.7071068, 0.7071068, 0, 0]],  # 90 deg around X axis
                device=device,
                dtype=torch.float32,
            ).repeat(b, 1)

            plate_pose = Pose.create_from_pq(p=plate_pos, q=vertical_quat)
            self.plate.set_pose(plate_pose)

            # Zero velocities so the plate remains stationary until grasped
            zero_velocity = torch.zeros((b, 3), device=device)
            self.plate.set_linear_velocity(zero_velocity)
            self.plate.set_angular_velocity(zero_velocity)

            # Save initial pose so we can hold the plate in place each step
            self._plate_initial_pose = plate_pose
            self._plate_gravity_enabled = False

            self.initialize_episode_hook(env_idx, mo_pose=plate_pose, ro_pose=rack_pose)

        self._initialize_agent()

    def _initialize_agent(self):
        qpos_0 = np.array([0.038014565, -0.05580395, 0.02353088, -2.1653874, 0.04206629, 2.127178, -0.7543587, 0.04, 0.04])
        self.agent.reset(qpos_0)


    def step(self, action):
        # Zero plate velocity each step to cancel gravity accumulation,
        # but do NOT reset pose — the plate must stay physically interactable
        # so the robot can grasp it.
        if not self._plate_gravity_enabled:
            zero_vel = torch.zeros((self.num_envs, 3), device=self.device)
            self.plate.set_linear_velocity(zero_vel)
            self.plate.set_angular_velocity(zero_vel)

        obs = super().step(action)

        if (
            not self._plate_gravity_enabled
            and bool(self.agent.is_grasping(self.plate).any())
        ):
            self._plate_gravity_enabled = True

        return obs

    def evaluate(self):
        plate_pos = self.plate.pose.p
        table_p_arr = np.asarray(self.table.pose.p.cpu()).ravel()

        table_z = float(table_p_arr[-1])
        table_top_z = table_z + float(self.table_scene_builders[0].table_height)

        # Check that plate is above table surface
        plate_above_table = plate_pos[:, 2] > table_top_z - 0.01

        rack_pos = self.dish_rack.pose.p
        rack_half = torch.tensor(
            self._rack_extent / 2.0, device=self.device, dtype=plate_pos.dtype
        )
        # within_x = torch.abs(plate_pos[:, 0] - rack_pos[:, 0]) <= rack_half[0]
        # within_y = torch.abs(plate_pos[:, 1] - rack_pos[:, 1]) <= rack_half[1]
        # within_z = torch.abs(plate_pos[:, 2] - rack_pos[:, 2]) <= rack_half[2]
        # plate_outside_rack = ~(within_x & within_y & within_z)
        within_z = torch.abs(plate_pos[:, 2] - rack_pos[:, 2]) <= rack_half[2]
        plate_outside_rack = ~(within_z)

        success = plate_above_table & plate_outside_rack

        return {
            "success": success,
            "plate_outside_rack": plate_outside_rack,
        }
