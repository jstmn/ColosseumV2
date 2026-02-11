from typing import Any, Dict, Union
import os

import numpy as np
import sapien
import sapien.render
import torch
import trimesh
from transforms3d.euler import euler2quat

from mani_skill import ASSET_DIR, PACKAGE_ASSET_DIR
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.envs.tasks.tabletop.colosseum_v2.colosseum_v2_core import ColosseumV2Env


@register_env("CookItemInPan-v1", max_episode_steps=150, asset_download_ids=["ycb"])
class CookItemInPanEnv(ColosseumV2Env):
    """
    **Task Description:**
    Pick a pan, place it on the stove, then place a food item inside the pan.

    **Randomizations:**
    - The pan spawns with small xy noise on the table.
    - The food item spawns away from the pan with small xy noise.

    **Success Conditions:**
    - The pan is on the stove and released.
    - The food item is inside the pan, static, and released.
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        pan_glb_path=None,  # Uses default from PACKAGE_ASSET_DIR
        pan_scale=0.0018,  # STL is in mm, 1.5x bigger
        pan_inner_radius=None,
        stove_glb_path=None,  # Uses default from PACKAGE_ASSET_DIR
        stove_scale=0.015,  # 1.5x bigger
        food_model_id="013_apple",
        **kwargs,
    ):
        # Use default asset paths if not provided
        if pan_glb_path is None:
            pan_glb_path = PACKAGE_ASSET_DIR / "cook_item_in_pan/panev2.stl"
        if stove_glb_path is None:
            stove_glb_path = PACKAGE_ASSET_DIR / "cook_item_in_pan/11633_Cooktop_v1_L3.obj"
        self.pan_glb_path = str(pan_glb_path)
        self.pan_scale = float(pan_scale)
        self.pan_inner_radius_override = pan_inner_radius
        self.food_model_id = food_model_id
        self.stove_glb_path = str(stove_glb_path)
        self.stove_scale = float(stove_scale)
        self.stove_mesh_pose = sapien.Pose(q=euler2quat(0, 0, np.pi / 2))  # 90 degree rotation
        self.stove_z_offset = 0.001  # small gap above table
        self.pan_handle_axis = np.array([-1.0, 0.0, 0.0], dtype=np.float32)  # Handle along -X in mesh
        self.pan_handle_offset_ratio = 0.65

        if not os.path.exists(self.pan_glb_path):
            raise FileNotFoundError(
                f"Pan mesh not found at {self.pan_glb_path}. Provide pan_glb_path."
            )
        if not os.path.exists(self.stove_glb_path):
            raise FileNotFoundError(
                f"Stove mesh not found at {self.stove_glb_path}. Provide stove_glb_path."
            )

        ycb_info = load_json(ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json")
        food_meta = ycb_info[self.food_model_id]

        self.food_scale = float(food_meta["scales"][0])
        self.food_bbox_min = np.array(food_meta["bbox"]["min"], dtype=np.float32)
        self.food_bbox_max = np.array(food_meta["bbox"]["max"], dtype=np.float32)
        food_extents = (self.food_bbox_max - self.food_bbox_min) * self.food_scale
        self.food_half_size = food_extents / 2
        self.food_radius = float(max(self.food_half_size[0], self.food_half_size[1]))
        self.food_bottom_offset = float(-self.food_bbox_min[2] * self.food_scale)

        self._load_pan_geometry()
        self._load_stove_geometry()

        self.stove_center_xy = np.array([0.15, 0.15], dtype=np.float32)  # back right of table
        self.table_surface_z = 0.0
        # Pan must be centered on stove
        self.stove_xy_radius = 0.08  # 8cm radius - pan must be on stove
        self.stove_z_tolerance = 0.05  # 5cm z tolerance

        # Pan flat on table, handle pointing to the side (+Y) for easy rim access
        self.pan_spawn_x_rot = 0.0
        self.pan_spawn_z_rot = np.pi / 2  # 90 degree rotation - handle points +Y
        self.pan_spawn_z_offset = 0.0
        # Pan spawns in front of robot for easy reach
        self.pan_spawn_center = np.array([0.0, 0.0], dtype=np.float32)
        self.pan_spawn_half_size = 0.0  # No randomization
        # Food on table with large randomization area (shifted right from robot's view)
        self.food_spawn_center = np.array([-0.12, -0.20], dtype=np.float32)
        self.food_spawn_half_size = 0.08  # Randomization within reachable area

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_pan_geometry(self):
        mesh = self._load_pan_mesh()
        bounds = mesh.bounds
        extents = (bounds[1] - bounds[0]) * self.pan_scale
        self.pan_half_size = extents / 2
        self.pan_radius = float(max(self.pan_half_size[0], self.pan_half_size[1]))
        self.pan_body_radius = float(min(self.pan_half_size[0], self.pan_half_size[1]))
        self.pan_bottom_offset = float(-bounds[0][2] * self.pan_scale)
        self.pan_top_offset = float(bounds[1][2] * self.pan_scale)
        self.pan_local_bounds_min = bounds[0] * self.pan_scale
        self.pan_local_bounds_max = bounds[1] * self.pan_scale
        center_xy = 0.5 * (self.pan_local_bounds_min[:2] + self.pan_local_bounds_max[:2])

        extent_x = self.pan_local_bounds_max[0] - self.pan_local_bounds_min[0]
        extent_y = self.pan_local_bounds_max[1] - self.pan_local_bounds_min[1]
        center_z = 0.5 * (self.pan_local_bounds_min[2] + self.pan_local_bounds_max[2])
        z_span = self.pan_local_bounds_max[2] - self.pan_local_bounds_min[2]
        handle_z = self.pan_local_bounds_max[2] - 0.2 * z_span
        axis = self.pan_handle_axis.copy()
        axis_norm = np.linalg.norm(axis[:2])
        if axis_norm < 1e-6:
            axis = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        else:
            axis[:2] /= axis_norm
        offset = self.pan_handle_offset_ratio * max(extent_x, extent_y)
        handle_point = np.array(
            [
                center_xy[0] + axis[0] * offset,
                center_xy[1] + axis[1] * offset,
                handle_z,
            ],
            dtype=np.float32,
        )

        self.pan_handle_local_point = handle_point
        self.pan_handle_local_offset = np.array(
            [
                handle_point[0] - center_xy[0],
                handle_point[1] - center_xy[1],
                handle_point[2] - center_z,
            ],
            dtype=np.float32,
        )
        self.pan_handle_local_dir = np.array([axis[0], axis[1], 0.0], dtype=np.float32)
        if self.pan_inner_radius_override is None:
            self.pan_inner_radius = self.pan_body_radius * 0.75
        else:
            self.pan_inner_radius = float(self.pan_inner_radius_override)

    def _load_stove_geometry(self):
        mesh = self._load_stove_mesh()
        if not np.allclose(self.stove_mesh_pose.q, np.array([1.0, 0.0, 0.0, 0.0])):
            rot = trimesh.transformations.quaternion_matrix(self.stove_mesh_pose.q)
            mesh = mesh.copy()
            mesh.apply_transform(rot)
        bounds = mesh.bounds
        extents = (bounds[1] - bounds[0]) * self.stove_scale
        self.stove_half_size = extents / 2
        self.stove_bottom_offset = float(-bounds[0][2] * self.stove_scale)
        self.stove_top_offset = float(bounds[1][2] * self.stove_scale)

    def _load_pan_mesh(self):
        loaded = trimesh.load(self.pan_glb_path, force="scene")
        if isinstance(loaded, trimesh.Scene):
            if not loaded.geometry:
                raise ValueError("Pan mesh has no geometry")
            mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
        else:
            mesh = loaded
        if mesh.is_empty:
            raise ValueError("Pan mesh is empty")
        return mesh

    def _load_stove_mesh(self):
        loaded = trimesh.load(self.stove_glb_path, force="scene")
        if isinstance(loaded, trimesh.Scene):
            if not loaded.geometry:
                raise ValueError("Stove mesh has no geometry")
            mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
        else:
            mesh = loaded
        if mesh.is_empty:
            raise ValueError("Stove mesh is empty")
        return mesh

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.35, -0.3, 0.4], target=[0.0, 0.0, 0.0])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, -0.35, 0.45], [0.0, 0.0, 0.05])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.45, 0, 0]))

    def _build_ycb_actor(self, model_id: str, name: str) -> Actor:
        actors_list = []
        for i in range(self.num_envs):
            builder.set_scene_idxs([i])
            actor = builder.build(name=f"{name}_{i}")
            self.remove_from_state_dict_registry(actor)
            actors_list.append(actor)
        merged = Actor.merge(actors_list, name=name)
        self.add_to_state_dict_registry(merged)
        return merged

    def _load_scene(self, options: dict):
        self._add_table_to_scene()
        raw_table = self._table_scene.table._objs[0]
        table_z = float(raw_table.pose.p[2])
        self.table_surface_z = table_z + float(self._table_scene.table_height)

        def _get_stove_builder():
            # builder = self.scene.create_actor_builder()
            scale = (self.stove_scale, self.stove_scale, self.stove_scale)
            # # Use convex collision for OBJ file
            # builder.add_convex_collision_from_file(
            #     filename=self.stove_glb_path,
            #     scale=scale,
            #     pose=self.stove_mesh_pose,
            #     density=300.0,
            # )
            # builder.add_visual_from_file(
            #     filename=self.stove_glb_path,
            #     scale=scale,
            #     pose=self.stove_mesh_pose,
            # )
            initial_pose = sapien.Pose(
                p=[
                    float(self.stove_center_xy[0]),
                    float(self.stove_center_xy[1]),
                    self.table_surface_z + self.stove_bottom_offset + self.stove_z_offset,
                ]
            )
            return self.get_glb_asset_builder(
                glb_filepath=self.stove_glb_path,
                object_type="RO",
                density=300.0,
                scale=scale,
                pose=initial_pose,
            )

        def _get_pan_builder():
            scale = (self.pan_scale, self.pan_scale, self.pan_scale)
            # builder = self.scene.create_actor_builder()
            # builder.add_multiple_convex_collisions_from_file(
            #     filename=self.pan_glb_path,
            #     scale=scale,
            #     decomposition="coacd",
            #     density=300.0,
            # )
            # builder.add_visual_from_file(filename=self.pan_glb_path, scale=scale)
            # builder.initial_pose = sapien.Pose()
            # return builder
            return self.get_glb_asset_builder(
                glb_filepath=self.pan_glb_path,
                object_type="MO",
                density=300.0,
                scale=scale,
            )

        def _get_food_builder():
            return self.get_ycb_asset_builder(
                ycb_id=self.food_model_id,
                object_type="MO",
            )

        self.stove = self.add_asset_to_scene(_get_stove_builder, name="stove", type_="kinematic",  object_type="RO")
        self.pan = self.add_asset_to_scene(_get_pan_builder, name="pan", type_="dynamic",  object_type="MO")
        self.food = self.add_asset_to_scene(_get_food_builder, name="food", type_="dynamic",  object_type="MO")
        # self.load_scene_hook(manipulation_objects=[self.pan, self.food], receiving_objects=[self.stove], add_table_to_scene=False)
        self.load_scene_hook(manipulation_objects=[self.pan, self.food], receiving_objects=[], add_table_to_scene=False)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):

        with torch.device(self.device):
            b = len(env_idx)
            # self.table_scene.initialize(env_idx)

            # Initialize robot at pregrasp pose (above pan rim, ready to grasp)
            pregrasp_qpos = np.array([
                -0.321, 0.314, 0.635, -2.240, -0.300, 2.473, -1.820,
                0.04, 0.04  # gripper open
            ])
            qpos = torch.tensor(pregrasp_qpos, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(b, 1)
            self.agent.reset(qpos)

            pan_xy = (
                torch.rand((b, 2), device=self.device) * 2 - 1
            ) * self.pan_spawn_half_size + torch.tensor(
                self.pan_spawn_center, device=self.device
            )
            pan_pos = torch.zeros((b, 3), device=self.device)
            pan_pos[:, :2] = pan_xy
            pan_pos[:, 2] = self.table_surface_z + self.pan_bottom_offset + self.pan_spawn_z_offset
            pan_q = torch.tensor(
                euler2quat(self.pan_spawn_x_rot, 0, self.pan_spawn_z_rot),
                device=self.device,
            ).repeat(b, 1)
            self.pan.set_pose(Pose.create_from_pq(p=pan_pos, q=pan_q))

            food_xy = (
                torch.rand((b, 2), device=self.device) * 2 - 1
            ) * self.food_spawn_half_size + torch.tensor(
                self.food_spawn_center, device=self.device
            )
            food_pos = torch.zeros((b, 3), device=self.device)
            food_pos[:, :2] = food_xy
            food_pos[:, 2] = self.table_surface_z + self.food_bottom_offset
            self.food.set_pose(Pose.create_from_pq(p=food_pos))
            
            self.initialize_episode_hook(env_idx, mo_pose=self.pan.pose)

    def evaluate(self):
        pan_pose = self.pan.pose
        food_pose = self.food.pose

        pan_pos = pan_pose.p
        food_pos = food_pose.p
        stove_pos = self.stove.pose.p

        # Pan mesh is offset from its pose origin, so when pan is visually on stove,
        # pan_pos reports ~[0.22, -0.26]. Use a bounding box that matches this.
        # Stove visual bounds roughly X:[0, 0.3], Y:[0, 0.3]
        # Pan pose when on stove: X ~0.22, Y ~-0.26
        is_pan_on_stove = (
            (pan_pos[:, 0] >= 0.0) & (pan_pos[:, 0] <= 0.35) &
            (pan_pos[:, 1] >= -0.35) & (pan_pos[:, 1] <= 0.0)
        )

        # Pan mesh is offset from pan_pos. When pan_pos=[0.22,-0.26], body is at [0.15,0.15]
        # offset = pan_pos - body → [0.07, -0.41]
        # body = pan_pos - offset
        pan_body_offset = torch.tensor([0.07, -0.41], device=self.device)
        pan_body_center = pan_pos[:, :2] - pan_body_offset
        food_to_pan_body_dist = torch.linalg.norm(food_pos[:, :2] - pan_body_center, dim=1)
        is_food_in_pan = food_to_pan_body_dist <= 0.15  # generous threshold

        is_pan_static = self.pan.is_static(lin_thresh=0.1, ang_thresh=1.0)
        is_food_static = self.food.is_static(lin_thresh=0.1, ang_thresh=1.0)
        is_pan_grasped = self.agent.is_grasping(self.pan)
        is_food_grasped = self.agent.is_grasping(self.food)

        success = (
            is_pan_on_stove
            & is_food_in_pan
            & is_pan_static
            & is_food_static
            & ~is_pan_grasped
            & ~is_food_grasped
        )

        return {
            "success": success,
            "is_pan_on_stove": is_pan_on_stove,
            "is_food_in_pan": is_food_in_pan,
            "is_pan_static": is_pan_static,
            "is_food_static": is_food_static,
            "is_pan_grasped": is_pan_grasped,
            "is_food_grasped": is_food_grasped,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                stove_pose=self.stove.pose.raw_pose,
                pan_pose=self.pan.pose.raw_pose,
                food_pose=self.food.pose.raw_pose,
                pan_to_stove_pos=self.stove.pose.p - self.pan.pose.p,
                food_to_pan_pos=self.pan.pose.p - self.food.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        pan_pos = self.pan.pose.p
        food_pos = self.food.pose.p
        stove_pos = self.stove.pose.p

        pan_to_stove_dist = torch.linalg.norm(pan_pos[:, :2] - stove_pos[:, :2], dim=1)
        food_to_pan_dist = torch.linalg.norm(food_pos[:, :2] - pan_pos[:, :2], dim=1)

        pan_reach = 1 - torch.tanh(5 * pan_to_stove_dist)
        food_reach = 1 - torch.tanh(5 * food_to_pan_dist)
        stove_bonus = info["is_pan_on_stove"].float()
        pan_bonus = info["is_food_in_pan"].float()

        reward = 0.5 * pan_reach + 0.5 * food_reach + stove_bonus + pan_bonus
        reward[info["success"]] = 3.0
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 3.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
