from typing import Callable
import os

from termcolor import cprint
from sapien.physx import PhysxMaterial
import numpy as np
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
import numpy as np
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import sapien_utils
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.envs.tasks.tabletop.colosseum_v2.distraction_set import DistractionSet
from mani_skill.utils.building.actor_builder import ActorBuilder
from mani_skill.utils.scene_builder.robocasa.fixtures.cabinet import OpenCabinet
from mani_skill.utils.building.articulation_builder import ArticulationBuilder
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.io_utils import load_json
from sapien.render import RenderBodyComponent
from transforms3d.euler import euler2quat
from mani_skill.utils.structs.actor import Actor
from mani_skill import ASSET_DIR

FLOOR_HEIGHT = -0.920

def _set_color_or_texture(actor: Actor, color_cfg: dict | None, texture_cfg: dict | None, set_color: bool, set_texture: bool, use_single_texture_or_texture: bool = False):

    assert actor is not None, "actor must be provided, got None"

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

    objs = []
    if isinstance(actor, OpenCabinet):
        for shelf in actor.shelves:
            objs.append(shelf._objs)
    elif isinstance(actor, Articulation):
        for obj in actor._objs:
            for link in obj.links:
                objs.append(link.entity)
        cprint(f"WARNING: Articulation {actor.name} is not supported for color/texture randomization, skipping", "yellow")
        return
    else:
        objs = actor._objs

    # The following code is borrowed from here: https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/domain_randomization.html
    for obj in objs:
        # modify the i-th object which is in parallel environment i
        render_body_component: RenderBodyComponent = obj.find_component_by_type(RenderBodyComponent)
        assert isinstance(render_body_component, RenderBodyComponent), f"render_body_component must be a RenderBodyComponent, got {type(render_body_component)}"
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
        self.ignored_variation_factors = kwargs.pop("ignored_variation_factors", [])
        self._ds.disable_variation_factors(self.ignored_variation_factors)

        # 
        self._human_render_shader = kwargs.pop("human_render_shader", "default")

        # We will use self._table_scenes if the table color or texture is enabled, otherwise the single table_scene
        # self._table_scenes: list[TableSceneBuilder] = []
        self._table_scene_builders: list[TableSceneBuilder] = []
        self._table: Actor | None = None

        self.robot_init_qpos_noise = robot_init_qpos_noise
        self._load_scene_hool_called = False
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

    @property
    def table(self) -> Actor:
        assert self._table is not None, "table is None"
        return self._table

    @property
    def table_scene_builders(self) -> list[TableSceneBuilder]:
        assert self._table_scene_builders is not None, "table_scene_builders is None"
        return self._table_scene_builders

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

    def _add_articulation_to_scene(self, get_builder_fn: Callable[[], ArticulationBuilder], name: str, physics_type: str, object_type: str) -> Articulation:
        assert physics_type == "articulation", "physics_type must be articulation"
        
        articulations = []
        for i in range(self.num_envs):
            name_i = f"{name}_env:{i}"
            builder: ArticulationBuilder = get_builder_fn()
            builder.set_scene_idxs([i])
            articulation = builder.build(name=name_i)

            if object_type == "MO" and self._ds.MO_mass_enabled():
                mass_scale = np.random.uniform(*self._ds.MO_mass_cfg["mass_scale_range"])
                for link in articulation.links:
                    new_mass = (link.get_mass() * mass_scale).item()
                    link.set_mass(new_mass)

            self.remove_from_state_dict_registry(articulation)
            articulations.append(articulation)

        actor = Articulation.merge(articulations, name=name, merge_links=True)

        self.add_to_state_dict_registry(actor)
        assert isinstance(actor, Actor | Articulation), f"actor must be an actor or articulation, is {type(actor)}"
        return actor


    def add_asset_to_scene(self, get_builder_fn: Callable[[], ActorBuilder | ArticulationBuilder], name: str, physics_type: str, object_type: str) -> Actor | Articulation:
        """
        def builder_fn():
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=[0.05] * 3)
            builder.add_box_visual(
                half_size=[0.05] * 3,
                material=sapien.render.RenderMaterial(
                    base_color=np.array([12, 42, 160, 255]) / 255,
                ),
            )
            builder.initial_pose = sapien.Pose(p=[0, 0, 0.05])
            return builder

        self.add_asset_to_scene(get_builder_fn, name="cube", physics_type="dynamic")
        """
        assert object_type in ["MO", "RO", "DISTRACTOR"]
        if physics_type == "articulation":
            return self._add_articulation_to_scene(get_builder_fn, name, physics_type, object_type)

        actors = []
        for i in range(self.num_envs):
            name_i = f"{name}_env:{i}"
            builder: ActorBuilder | ArticulationBuilder = get_builder_fn()
            builder.set_scene_idxs([i])
            assert isinstance(builder, ActorBuilder), "builder must be an actor builder"
            if physics_type == "dynamic":
                actor = builder.build_dynamic(name=name_i)
            elif physics_type == "static":
                actor = builder.build_static(name=name_i)
            elif physics_type == "kinematic":
                actor = builder.build_kinematic(name=name_i)
            else:
                raise ValueError(f"Invalid type: {physics_type}")

            if object_type == "MO" and self._ds.MO_mass_enabled():
                mass_scale = np.random.uniform(*self._ds.MO_mass_cfg["mass_scale_range"])
                assert isinstance(actor, Actor), "actor must be an actor"
                new_mass = (actor.get_mass() * mass_scale).item()
                actor.set_mass(new_mass)

            self.remove_from_state_dict_registry(actor)
            actors.append(actor)

        actor = Actor.merge(actors, name=name)
        self.add_to_state_dict_registry(actor)
        return actor


    def get_box_asset_builder(
        self,
        half_size: tuple[float, float, float],
        color: list,
        object_type: str,
    ):
        if self._ds.MO_size_enabled() and object_type == "MO":
            scale_multiplier = self._ds.MO_size_cfg["scale_range"]
            scale = np.random.uniform(*scale_multiplier)
            half_size = (scale * half_size[0], scale * half_size[1], scale * half_size[2])

        elif self._ds.RO_size_enabled() and object_type == "RO":
            scale_multiplier = self._ds.RO_size_cfg["scale_range"]
            scale = np.random.uniform(*scale_multiplier)
            half_size = (scale * half_size[0], scale * half_size[1], scale * half_size[2])

        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=half_size)
        builder.add_box_visual(
            half_size=half_size,
            material=sapien.render.RenderMaterial(
                base_color=color,
            ),
        )
        # cube_builder.initial_pose = sapien.Pose(p=[0, 0, self.cube_half_size])
        return builder


    # Borrowed from mani_skill/utils/building/actors/ycb.py:get_ycb_builder(...)
    def get_ycb_asset_builder(
        self,
        ycb_id: str,
        object_type: str,
    ):
        builder = self.scene.create_actor_builder()

        model_db = load_json(ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json")
        metadata = model_db[ycb_id]
        density = metadata.get("density", 1000)
        model_scales = metadata.get("scales", [1.0])
        model_scale = model_scales[0]
        scale = (model_scale, model_scale, model_scale)

        # Optionally increase / decrease the scale
        if self._ds.MO_size_enabled() and object_type == "MO":
            scale_multiplier = np.random.uniform(*self._ds.MO_size_cfg["scale_range"])
            scale = (scale_multiplier * model_scale, scale_multiplier * model_scale, scale_multiplier * model_scale)

        elif self._ds.RO_size_enabled() and object_type == "RO":
            scale_multiplier = np.random.uniform(*self._ds.RO_size_cfg["scale_range"])
            scale = (scale_multiplier * model_scale, scale_multiplier * model_scale, scale_multiplier * model_scale)

        physical_material = None
        model_dir = ASSET_DIR / "assets/mani_skill2_ycb/models" / ycb_id
        collision_file = str(model_dir / "collision.ply")
        builder.add_multiple_convex_collisions_from_file(
            filename=collision_file,
            scale=scale,
            material=physical_material,
            density=density,
        )
        visual_file = str(model_dir / "textured.obj")
        builder.add_visual_from_file(filename=visual_file, scale=scale)
        return builder

    def get_glb_asset_builder(
        self, 
        glb_filepath: str, 
        object_type: str, 
        color: list | None = None, 
        scale: tuple[float, float, float] | None = None, 
        density: float | None = None,
        physical_material: PhysxMaterial | None = None,
        visual_material: sapien.render.RenderMaterial | None = None,
        initial_pose: sapien.Pose = sapien.Pose(),
        mesh_pose: sapien.Pose = sapien.Pose(),
    ):
        """Load GLB file as a static actor in the scene"""
        assert object_type in ["MO", "RO"]

        if scale is None:
            scale = (1.0, 1.0, 1.0)
        else:
            scale = scale

        if self._ds.MO_size_enabled() and object_type == "MO":
            scale_multiplier = self._ds.MO_size_cfg["scale_range"]
            scale = (
                np.random.uniform(*scale_multiplier) * scale[0],
                np.random.uniform(*scale_multiplier) * scale[1],
                np.random.uniform(*scale_multiplier) * scale[2],
            )

        elif self._ds.RO_size_enabled() and object_type == "RO":
            scale_multiplier = self._ds.RO_size_cfg["scale_range"]
            scale = (
                np.random.uniform(*scale_multiplier) * scale[0],
                np.random.uniform(*scale_multiplier) * scale[1],
                np.random.uniform(*scale_multiplier) * scale[2],
            )
        builder = self.scene.create_actor_builder()
        builder.set_initial_pose(initial_pose)

        # Visual
        if color is not None:
            assert visual_material is None, "color and visual_material cannot be set at the same time"
            custom_material = sapien.render.RenderMaterial()
            custom_material.base_color = color  # Green [R, G, B, A]
            custom_material.roughness = 0.8
            custom_material.metallic = 0.0
            builder.add_visual_from_file(filename=glb_filepath, scale=scale, material=custom_material)
        elif visual_material is not None:
            assert color is None, "color and visual_material cannot be set at the same time"
            builder.add_visual_from_file(filename=glb_filepath, scale=scale, material=visual_material, pose=mesh_pose)
        else:
            builder.add_visual_from_file(filename=glb_filepath, scale=scale, pose=mesh_pose)

        # Collision
        if (density is None) and (physical_material is None):
            builder.add_multiple_convex_collisions_from_file(glb_filepath, decomposition="coacd", scale=scale, pose=mesh_pose)
        elif (density is None) and (physical_material is not None):
            builder.add_multiple_convex_collisions_from_file(glb_filepath, decomposition="coacd", scale=scale, material=physical_material, pose=mesh_pose)
        elif (density is not None) and (physical_material is None):
            builder.add_multiple_convex_collisions_from_file(glb_filepath, decomposition="coacd", scale=scale, density=density, pose=mesh_pose)
        elif (density is not None) and (physical_material is not None):
            builder.add_multiple_convex_collisions_from_file(glb_filepath, decomposition="coacd", scale=scale, material=physical_material, density=density, pose=mesh_pose)
        else:
            raise ValueError(f"Unhandled combination of density and physical_material: {density} and {physical_material}")

        return builder

    def _add_table_to_scene(self):
        # Create the table and optionally set its color and/or texture
        if self._ds.table_color_enabled() or self._ds.table_texture_enabled():
            # Note: you can't add a texture to the table if you've set its color already.
            add_visual_from_file = not self._ds.table_color_enabled()
            for i in range(self.num_envs):
                table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
                table_scene.build(remove_table_from_state_dict_registry=True, scene_idx=i, name_suffix=f"table_env:{i}", add_visual_from_file=add_visual_from_file)
                self._table_scene_builders.append(table_scene)

            self._table = Actor.merge([ts.table for ts in self._table_scene_builders], name="table_scene")
            self.add_to_state_dict_registry(self._table)
            _set_color_or_texture(self._table, self._ds.table_color_cfg, self._ds.table_texture_cfg, self._ds.table_color_enabled(), self._ds.table_texture_enabled())
        else:
            self._table_scene_builders = [TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)]
            self._table_scene_builders[0].build()
            self._table = self._table_scene_builders[0].table


    def load_scene_hook(self, manipulation_objects: list[Actor], receiving_objects: list[Actor] | None = None, add_table_to_scene: bool = True):
        """
        This function is called when the scene is loaded.
        Args:
            manipulation_objects (list[Actor]): The manipulation objects to modify.
            receiving_objects (list[Actor]): The receiving objects to modify.
        """
        self._load_scene_hool_called = True

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

                self._ds._internal["distractor_object_cfg"]["sphere_actors"].append(
                    self.add_asset_to_scene(get_sphere_builder, name=f"distractor_sphere_{i}", physics_type="dynamic", object_type="DISTRACTOR")
                )

        if add_table_to_scene:
            self._add_table_to_scene()

        # Manipulation object
        for mo in manipulation_objects:
            assert mo is not None, "mo must be provided, got None"
            _set_color_or_texture(mo, self._ds.MO_color_cfg, self._ds.MO_texture_cfg, self._ds.MO_color_enabled(), self._ds.MO_texture_enabled())

        # Receiving object
        if receiving_objects is not None:
            for ro in receiving_objects:
                assert ro is not None, "ro must be provided, got None"
                _set_color_or_texture(ro, self._ds.RO_color_cfg, self._ds.RO_texture_cfg, self._ds.RO_color_enabled(), self._ds.RO_texture_enabled())

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


    def initialize_episode_hook(self, env_idx: torch.Tensor, mo_pose: torch.Tensor | None = None, ro_pose: torch.Tensor | None = None, qpos_0: np.ndarray | None = None, initialize_table_scene: bool = True):
        
        assert self._load_scene_hool_called, "load_scene_hook must be called before initialize_episode_hook"

        if mo_pose is not None:
            assert mo_pose.shape[0] == self.num_envs
            assert mo_pose.shape[1] >= 2, f"mo_pose must have at least 2 dimensions, got {mo_pose.shape[1]}"
        if ro_pose is not None:
            assert ro_pose.shape[0] == self.num_envs
            assert ro_pose.shape[1] >= 2, f"ro_pose must have at least 2 dimensions, got {ro_pose.shape[1]}"


        if initialize_table_scene:
            for ts in self._table_scene_builders:
                ts.initialize(env_idx, qpos_0=qpos_0)

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