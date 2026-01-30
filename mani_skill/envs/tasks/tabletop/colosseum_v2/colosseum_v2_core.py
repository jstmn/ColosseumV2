from typing import Optional, Callable
import os

import numpy as np
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
import numpy as np
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
import os
from mani_skill.envs.tasks.tabletop.colosseum_v2.distraction_set import DistractionSet
from mani_skill.utils.building.actor_builder import ActorBuilder


import numpy as np
import torch
import sapien
import numpy as np
from sapien.render import RenderBodyComponent
from transforms3d.euler import euler2quat

from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.actor import Actor


FLOOR_HEIGHT = -0.920


# def get_camera_configs(xy_offset: float, z_offset: float, target: tuple[float, float, float], camera_width: int = DEFAULT_CAMERA_WIDTH, camera_height: int = DEFAULT_CAMERA_HEIGHT):
#     pose_center = sapien_utils.look_at(eye=[xy_offset, 0,  z_offset], target=target)
#     pose_left = sapien_utils.look_at(eye=[0.0, -xy_offset, z_offset], target=target)
#     pose_right = sapien_utils.look_at(eye=[0.0, xy_offset, z_offset], target=target)
#     return [
#         CameraConfig(
#             uid="camera_center",
#             pose=pose_center,
#             width=camera_width,
#             height=camera_height,
#             fov=REALSENSE_DEPTH_FOV_VERTICAL_RAD,
#             near=0.01,
#             far=100,
#             shader_pack=SHADER,
#         ),
#         CameraConfig(
#             uid="camera_left",
#             pose=pose_left,
#             width=camera_width,
#             height=camera_height,
#             fov=REALSENSE_DEPTH_FOV_VERTICAL_RAD,
#             near=0.01,
#             far=100,
#             shader_pack=SHADER,
#         ),
#         CameraConfig(
#             uid="camera_right",
#             pose=pose_right,
#             width=camera_width,
#             height=camera_height,
#             fov=REALSENSE_DEPTH_FOV_VERTICAL_RAD,
#             near=0.01,
#             far=100,
#             shader_pack=SHADER,
#         )]

def _set_color_or_texture(actor: Actor, color_cfg: dict | None, texture_cfg: dict | None, set_color: bool, set_texture: bool, use_single_texture_or_texture: bool = False):

    if not set_color and not set_texture:
        return

    # If using a single texture / color for the entire actor, decide the color/texture here.
    if use_single_texture_or_texture:
        if set_texture:
            assert texture_cfg is not None, "texture_cfg is not set"
            texture = _get_random_texture(texture_cfg["textures_directory"])
        if set_color:
            assert color_cfg is not None, "color_cfg is not set"
            color = color_cfg["color_range"].sample_rgba()
        if set_color and set_texture:
            if np.random.random() < 0.5:
                use_color = True
            else:
                use_color = False

    # The following code is borrowed from here: https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/domain_randomization.html
    for obj in actor._objs:
        # modify the i-th object which is in parallel environment i
        render_body_component: RenderBodyComponent = obj.find_component_by_type(RenderBodyComponent)
        for render_shape in render_body_component.render_shapes:
            for part in render_shape.parts:
                # part.material: sapien.core.pysapien.RenderMaterial
                if not use_single_texture_or_texture:
                    if set_texture:
                        assert texture_cfg is not None, "texture_cfg is not set"
                        texture = _get_random_texture(texture_cfg["textures_directory"])
                    if set_color:
                        assert color_cfg is not None, "color_cfg is not set"
                        color = color_cfg["color_range"].sample_rgba()

                if set_color and not set_texture:
                    part.material.set_base_color(color)
                elif not set_color and set_texture:
                    part.material.set_base_color_texture(texture)
                elif set_color and set_texture:
                    if (use_single_texture_or_texture and use_color) or (not use_single_texture_or_texture and (np.random.random() < 0.5)):
                        part.material.set_base_color(color)
                    else:
                        part.material.set_base_color_texture(texture)




def _get_random_texture(texture_dir: str):
    texture_files = [f for f in os.listdir(texture_dir) if f.endswith('.png')]
    texture_file = np.random.choice(texture_files)
    return sapien.render.RenderTexture2D(filename=os.path.join(texture_dir, texture_file))



class ColosseumV2Env(BaseEnv):

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs
    ):
        distraction_set: DistractionSet | dict | None = kwargs.pop("distraction_set", None)
        if distraction_set is None:
            self._ds = DistractionSet()
        elif isinstance(distraction_set, dict):
            self._ds = DistractionSet(**distraction_set)
        elif isinstance(distraction_set, DistractionSet):
            self._ds = distraction_set
        else:
            raise ValueError(f"Invalid distraction set type: {type(distraction_set)}")

        # 
        self._human_render_shader = kwargs.pop("human_render_shader", "default")

        # We will use self._table_scenes if the table color or texture is enabled, otherwise the single table_scene
        self._table_scenes: list[TableSceneBuilder] = []
        self._table_scene: TableSceneBuilder | None = None
        self._table_actors: Actor | None = None

        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)


    def _get_human_render_camera_config(self, eye: tuple[float, float, float], target: tuple[float, float, float]):
        """ Configures the human render camera. Shader options:
            - minimal: The fastest shader with minimal GPU memory usage. Note that the background will always be black (normally it is the color of the ambient light)
            - default: A balance between speed and texture availability
            - rt: A shader optimized for photo-realistic rendering via ray-tracing
            - rt-med: Same as rt but runs faster with slightly lower quality
            - rt-fast: Same as rt-med but runs faster with slightly lower quality
            -> https://maniskill.readthedocs.io/en/latest/user_guide/concepts/sensors.html#shaders-and-textures
        """
        pose = sapien_utils.look_at(eye=eye, target=target)
        return CameraConfig("render_camera", pose=pose, width=500, height=500, fov=np.pi / 3, near=0.01, far=100, shader_pack=self._human_render_shader)



    def _load_lighting(self, options: dict):
        if self._ds.light_color_enabled():
            for sub_scene in self.scene.sub_scenes:
                sub_scene.ambient_light = self._ds.light_color_cfg["color_range"].sample_rgba(include_alpha=False)
                theta_rand = np.random.uniform(0, 2*np.pi)
                sub_scene.add_directional_light(
                    direction=[-np.cos(theta_rand), -np.sin(theta_rand), -1],
                    position=[np.cos(theta_rand), np.sin(theta_rand), 1],
                    color=self._ds.light_color_cfg["color_range"].sample_rgba(include_alpha=False),
                    shadow=True,
                    shadow_scale=5,
                    shadow_map_size=4096,
                )
                # maniskill only allows for shadow from one directional light at a time for some reason
        else:
            super()._load_lighting(options)

    def update_camera_configs(self, cfgs: list[CameraConfig]) -> list[CameraConfig]:
        if not self._ds.camera_pose_enabled():
            return cfgs

        rpy_range = self._ds.camera_pose_cfg["rpy_range"]
        xyz_range = self._ds.camera_pose_cfg["xyz_range"]

        for cfg in cfgs:
            rpy = np.random.uniform(*rpy_range)
            xyz = np.random.uniform(*xyz_range)
            delta_quat = euler2quat(rpy[0], rpy[1], rpy[2]).astype(np.float32)
            delta_pose = sapien.Pose(p=xyz, q=delta_quat)
            cfg.pose *= delta_pose

        return cfgs

    def _load_from_builder(self, get_builder_fn: Callable[[], ActorBuilder], name: str, type_: str) -> Actor:
        """ It's expected that the builder has had the following set:
            - add_box_collision
            - add_box_visual
            - initial_pose


        For example:
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=[0.05] * 3)
            builder.add_box_visual(
                half_size=[0.05] * 3,
                material=sapien.render.RenderMaterial(
                    base_color=np.array([12, 42, 160, 255]) / 255,
                ),
            )
            builder.initial_pose = sapien.Pose(p=[0, 0, 0.05])

            self._load_from_builder(builder, name="cube", type_="dynamic")
        """

        actors = []
        for i in range(self.num_envs):
            name_i = f"{name}_env:{i}"
            builder: ActorBuilder = get_builder_fn()
            builder.set_scene_idxs([i])
            if type_ == "dynamic":
                actor = builder.build_dynamic(name=name_i)
            elif type_ == "static":
                actor = builder.build_static(name=name_i)
            elif type_ == "kinematic":
                actor = builder.build_kinematic(name=name_i)
            else:
                raise ValueError(f"Invalid type: {type_}")
            self.remove_from_state_dict_registry(actor)
            actors.append(actor)
        actor = Actor.merge(actors, name=name)
        self.add_to_state_dict_registry(actor)
        return actor



    def load_glb_as_actor(self, glb_file_path: str, pose: sapien.Pose, name: str, type_: str, object_type: str, color: list | None = None):
        """Load GLB file as a static actor in the scene"""
        assert object_type in ["MO", "RO"]

        if self._ds.MO_size_enabled() and object_type == "MO":
            scale_range = self._ds.MO_size_cfg["scale_range"]
            scale = (np.random.uniform(*scale_range), np.random.uniform(*scale_range), np.random.uniform(*scale_range))

        elif self._ds.RO_size_enabled() and object_type == "RO":
            scale_range = self._ds.RO_size_cfg["scale_range"]
            scale = (np.random.uniform(*scale_range), np.random.uniform(*scale_range), np.random.uniform(*scale_range))
        else:
            scale = (1, 1, 1)

        actors = []

        for i in range(self.num_envs):
            name_i = f"{name}_env:{i}"

            builder = self.scene.create_actor_builder()
            if color is not None:
                custom_material = sapien.render.RenderMaterial()
                custom_material.base_color = color  # Green [R, G, B, A]
                custom_material.roughness = 0.8
                custom_material.metallic = 0.0
                builder.add_visual_from_file(filename=glb_file_path, scale=scale, material=custom_material)
            else:
                builder.add_visual_from_file(filename=glb_file_path, scale=scale)

            builder.add_multiple_convex_collisions_from_file(glb_file_path, decomposition="coacd", scale=scale)
            builder.set_initial_pose(pose)
            builder.set_scene_idxs([i])
            if type_ == "dynamic":
                actor = builder.build_dynamic(name_i)
            elif type_ == "static":
                actor = builder.build_static(name_i)
            elif type_ == "kinematic":
                actor = builder.build_kinematic(name_i)
            else:
                raise ValueError(f"Invalid type: {type_}")
            self.remove_from_state_dict_registry(actor)

            if object_type == "MO" and self._ds.MO_mass_enabled():
                mass_scale = np.random.uniform(*self._ds.MO_mass_cfg["mass_scale_range"])
                new_mass = (actor.get_mass() * mass_scale).item()
                actor.set_mass(new_mass)

            actors.append(actor)

        actor_merged = Actor.merge(actors, name=name)
        self.add_to_state_dict_registry(actor_merged)
        return actor_merged


    def load_scene_hook(self, manipulation_object: Optional[Actor], receiving_object: Actor | None = None):
        """
        This function is called when the scene is loaded.
        Args:
            scene (ManiSkillScene): The scene to modify.
            manipulation_object (Optional[Actor]): The manipulation object to modify. Note that this is a wrapper around
                                                    a sapien.Entity.
        """

        # New distractor spheres
        # TODO: Add YCB objects
        if self._ds.distractor_object_enabled():
            n_spheres = self._ds.distractor_object_cfg["n_spheres"]
            radius_range = self._ds.distractor_object_cfg["radius_range"]
            color_range = self._ds.distractor_object_cfg["color_range"]
            self._ds._internal["distractor_object_cfg"]["sphere_actors"] = []

            for i in range(n_spheres):
                def get_sphere_builder():
                    builder: ActorBuilder = self.scene.create_actor_builder()
                    builder.add_sphere_collision(
                        radius=np.random.uniform(*radius_range),
                    )
                    builder.add_sphere_visual(
                        radius=np.random.uniform(*radius_range),
                        material=sapien.render.RenderMaterial(
                        base_color=color_range.sample_rgba(),
                        ),
                    )
                    builder.set_initial_pose(sapien.Pose())
                    return builder

                self._ds._internal["distractor_object_cfg"]["sphere_actors"].append(self._load_from_builder(get_sphere_builder, name=f"distractor_sphere_{i}", type_="dynamic"))


        # Create the table and optionally set its color and/or texture
        if self._ds.table_color_enabled() or self._ds.table_texture_enabled():
            # Note: you can't add a texture to the table if you've set its color already.
            add_visual_from_file = not self._ds.table_color_enabled()
            for i in range(self.num_envs):
                table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
                table_scene.build(remove_table_from_state_dict_registry=True, scene_idx=i, name_suffix=f"table_env:{i}", add_visual_from_file=add_visual_from_file)
                self._table_scenes.append(table_scene)

            table_actors = Actor.merge([ts.table for ts in self._table_scenes], name="table_scene")
            self.add_to_state_dict_registry(table_actors)
            _set_color_or_texture(table_actors, self._ds.table_color_cfg, self._ds.table_texture_cfg, self._ds.table_color_enabled(), self._ds.table_texture_enabled())
        else:
            self._table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
            self._table_scene.build()


        # Manipulation object
        if manipulation_object is not None:
            _set_color_or_texture(manipulation_object, self._ds.MO_color_cfg, self._ds.MO_texture_cfg, self._ds.MO_color_enabled(), self._ds.MO_texture_enabled())

        # Receiving object
        if receiving_object is not None:
            _set_color_or_texture(receiving_object, self._ds.RO_color_cfg, self._ds.RO_texture_cfg, self._ds.RO_color_enabled(), self._ds.RO_texture_enabled())

        if self._ds.background_texture_enabled() or self._ds.background_color_enabled():

            # Build a single static "compound" wall actor (with 4 box segments) so we only
            # have one wall actor to manage/texture/etc. The actor itself is placed at the
            # origin; each segment uses a local pose.
            builder_wall = self.scene.create_actor_builder()
            dist_from_world = 1.5
            height = 2
            width = 0.1
            length = dist_from_world*2
            z = FLOOR_HEIGHT + (height / 2)
            material = sapien.render.RenderMaterial()
            material.roughness = 0.5
            material.metallic = 0.5 # < these don't seem to do anything

            # Left
            builder_wall.add_box_collision(half_size=(length / 2, width / 2, height / 2), pose=sapien.Pose(p=[0.0, dist_from_world, z]))
            builder_wall.add_box_visual(half_size=(length / 2, width / 2, height / 2), pose=sapien.Pose(p=[0.0, dist_from_world, z]))
            # Right
            builder_wall.add_box_collision(half_size=(length / 2, width / 2, height / 2), pose=sapien.Pose(p=[0.0, -dist_from_world, z]))
            builder_wall.add_box_visual(half_size=(length / 2, width / 2, height / 2), pose=sapien.Pose(p=[0.0, -dist_from_world, z]))
            # Back
            builder_wall.add_box_collision(half_size=(width / 2, length / 2, height / 2), pose=sapien.Pose(p=[-dist_from_world, 0.0, z]))
            builder_wall.add_box_visual(half_size=(width / 2, length / 2, height / 2), pose=sapien.Pose(p=[-dist_from_world, 0.0, z]))
            # Front
            builder_wall.add_box_collision(half_size=(width / 2, length / 2, height / 2), pose=sapien.Pose(p=[dist_from_world, 0.0, z]))
            builder_wall.add_box_visual(half_size=(width / 2, length / 2, height / 2), pose=sapien.Pose(p=[dist_from_world, 0.0, z]))

            # Add
            builder_wall.set_initial_pose(sapien.Pose(p=[0.0, 0.0, 0.0]))
            wall = builder_wall.build_static(name="walls")
            _set_color_or_texture(
                actor=wall,
                color_cfg=self._ds.background_color_cfg,
                texture_cfg=self._ds.background_texture_cfg,
                set_color=self._ds.background_color_enabled(),
                set_texture=self._ds.background_texture_enabled(),
                use_single_texture_or_texture=True,
            )

    def initialize_episode_hook(self, env_idx: torch.Tensor, mo_pose: torch.Tensor | None = None, ro_pose: torch.Tensor | None = None):
        if mo_pose is not None:
            assert mo_pose.shape[0] == self.num_envs
            assert mo_pose.shape[1] >= 2, f"mo_pose must have at least 2 dimensions, got {mo_pose.shape[1]}"
        if ro_pose is not None:
            assert ro_pose.shape[0] == self.num_envs
            assert ro_pose.shape[1] >= 2, f"ro_pose must have at least 2 dimensions, got {ro_pose.shape[1]}"
        
        if self._ds.table_color_enabled() or self._ds.table_texture_enabled():
            for ts in self._table_scenes:
                ts.initialize(env_idx)
        else:
            assert self._table_scene is not None, "Table has not been built yet"
            self._table_scene.initialize(env_idx)

        # TODO: Make sure that the sampled poses are beyond some epsilon of RO/ro objects
        if self._ds.distractor_object_enabled():

            radius_range = self._ds.distractor_object_cfg["radius_range"]
            x_lims = self._ds.distractor_object_cfg["x_lims"]
            y_lims = self._ds.distractor_object_cfg["y_lims"]
            x_range = x_lims[1] - x_lims[0]
            y_range = y_lims[1] - y_lims[0]


            for i in range(self._ds.distractor_object_cfg["n_spheres"]):
                # What happens if you set the poses such that the spheres collide with one another?
                # for i, sphere in enumerate(self._ds._internal["distractor_object_cfg"]["sphere_actors"]):
                xyz = torch.rand((self.num_envs, 3), dtype=torch.float32)
                xyz[:, 0] = x_range * xyz[:, 0] + x_lims[0]
                xyz[:, 1] = y_range * xyz[:, 1] + y_lims[0]
                xyz[:, 2] = radius_range[1] + 0.01 # get the maximum radius
                self._ds._internal["distractor_object_cfg"]["sphere_actors"][i].set_pose(Pose.create_from_pq(p=xyz))