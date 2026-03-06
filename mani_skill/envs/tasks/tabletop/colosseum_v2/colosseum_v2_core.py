from typing import Callable
import os
import random
from dataclasses import dataclass

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
from mani_skill.utils.geometry.rotation_conversions import euler_angles_to_matrix, matrix_to_quaternion
from mani_skill.utils.structs.actor import Actor
from mani_skill import ASSET_DIR
from yaml import load
from yaml.loader import SafeLoader

YCB_DISTRACTOR_OBJECTS = (
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "011_banana",
    "012_strawberry",
    "013_apple",
    "014_lemon",
    "015_peach",
    "016_pear",
    "017_orange",
    "018_plum",
    "019_pitcher_base",
    "021_bleach_cleanser",
    "022_windex_bottle",
    "024_bowl",
    "025_mug",
    "026_sponge",
    "028_skillet_lid",
    "029_plate",
    "030_fork",
    "031_spoon",
    "032_knife",
    "033_spatula",
    "035_power_drill",
    "036_wood_block",
    "037_scissors",
    "038_padlock",
    "040_large_marker",
    "042_adjustable_wrench",
    "043_phillips_screwdriver",
    "044_flat_screwdriver",
    "048_hammer",
    "050_medium_clamp",
    "051_large_clamp",
    "052_extra_large_clamp",
    "059_chain",
    "061_foam_brick",
    "062_dice",
    "065-a_cups",
    "065-b_cups",
    "065-c_cups",
    "065-d_cups",
    "065-e_cups",
    "065-f_cups",
    "065-g_cups",
    "065-h_cups",
    "065-i_cups",
    "065-j_cups",
    "070-a_colored_wood_blocks",
    "070-b_colored_wood_blocks",
    "072-a_toy_airplane",
    "072-b_toy_airplane",
    "072-c_toy_airplane",
    "072-d_toy_airplane",
    "072-e_toy_airplane",
    "073-a_lego_duplo",
    "073-b_lego_duplo",
    "073-c_lego_duplo",
    "073-d_lego_duplo",
    "077_rubiks_cube",
)


class VariationFactorDisabledError(Exception):
    """
    Raised when a variation factor is disabled but is enabled in the distraction set.
    """
    pass


@dataclass
class DisabledVariationFactors:
    """
    Stores which variation factors are disabled.
    """
    MO_color: bool = False
    MO_texture: bool = False
    MO_size: bool = False
    MO_mass: bool = False
    RO_color: bool = False
    RO_texture: bool = False
    RO_size: bool = False
    table_color: bool = False
    light_color: bool = False
    table_texture: bool = False
    distractor_object: bool = False
    background_texture: bool = False
    background_color: bool = False
    camera_pose: bool = False
    pose_randomization: bool = False

    def to_list(self):
        return [k for k, v in self.__dict__.items() if v]


@dataclass
class PlacementRegion:
    """
    Controls where objects are spawned in the scene.
    """
    x_lims: tuple[float, float] | np.ndarray
    y_lims: tuple[float, float] | np.ndarray

    def __post_init__(self):
        assert isinstance(self.x_lims, (tuple, np.ndarray)) and isinstance(self.y_lims, (tuple, np.ndarray)), "x_lims and y_lims must be a tuple or numpy array"
        if isinstance(self.x_lims, tuple):
            assert len(self.x_lims) == 2, "x_lims must be a tuple of length 2"
            assert self.x_lims[0] <= self.x_lims[1], "x_lims must be in increasing order"
        if isinstance(self.y_lims, tuple):
            assert len(self.y_lims) == 2, "y_lims must be a tuple of length 2"
            assert self.y_lims[0] <= self.y_lims[1], "y_lims must be in increasing order"
        if isinstance(self.x_lims, np.ndarray):
            assert self.x_lims.shape == (2,), "x_lims must be a numpy array of shape (2,)"
            assert self.x_lims[0] <= self.x_lims[1], "x_lims must be in increasing order"
        if isinstance(self.y_lims, np.ndarray):
            assert self.y_lims.shape == (2,), "y_lims must be a numpy array of shape (2,)"
            assert self.y_lims[0] <= self.y_lims[1], "y_lims must be in increasing order"

    @staticmethod
    def from_center_and_width(center: tuple[float, float] | np.ndarray, width: tuple[float, float] | np.ndarray) -> "PlacementRegion":
        assert isinstance(center, (tuple, np.ndarray)) and isinstance(width, (tuple, np.ndarray)), "center and width must be a tuple or numpy array"
        if isinstance(center, tuple):
            assert len(center) == 2, "center must be a tuple of length 2"
            assert len(width) == 2, "width must be a tuple of length 2"
        if isinstance(center, np.ndarray):
            assert center.shape == (2,), "center must be a numpy array of shape (2,)"
        if isinstance(width, np.ndarray):
            assert width.shape == (2,), "width must be a numpy array of shape (2,)"
        assert width[0] >= 0 and width[1] >= 0, "width must be non-negative"
        if width[0] < 1e-6:
            width = (1e-6, width[1])
        if width[1] < 1e-6:
            width = (width[0], 1e-6)
        return PlacementRegion(
            x_lims=(center[0] - width[0] / 2, center[0] + width[0] / 2),
            y_lims=(center[1] - width[1] / 2, center[1] + width[1] / 2),
        )

    @property
    def width_x(self) -> float:
        return self.x_lims[1] - self.x_lims[0]

    @property
    def width_y(self) -> float:
        return self.y_lims[1] - self.y_lims[0]

    def to_bounds(self) -> tuple[list[float], list[float]]:
        """ Follows the convention of UniformPlacementSampler: ((low1, low2, ...), (high1, high2, ...))
        """
        return ([self.x_lims[0], self.y_lims[0]], [self.x_lims[1], self.y_lims[1]])

    def sample_xy(self, b: int, device: torch.device) -> torch.Tensor:
        rand = (2*torch.rand((b, 2), device=device)) - 1
        rand[:, 0] *= self.width_x / 2
        rand[:, 1] *= self.width_y / 2
        center = torch.tensor(
            [(self.x_lims[0] + self.x_lims[1]) * 0.5, (self.y_lims[0] + self.y_lims[1]) * 0.5],
            device=device,
        )
        return rand + center

def _warn_or_raise_if_shared_materials(objs: list, actor_name: str, *, set_color: bool, set_texture: bool):
    """
    Heuristic guard for a common domain-randomization footgun:
    if parallel env instances share the same RenderMaterial, then per-env randomization
    becomes "last write wins" and all envs may look identical.

    This check errors out immediately when shared materials are detected.
    """
    if not set_color and not set_texture:
        return

    # Some actors (e.g. cabinets/shelves) store per-env objects as nested lists.
    # Flatten defensively so this check never crashes.
    flat_objs = []
    for o in objs:
        if isinstance(o, (list, tuple)):
            flat_objs.extend(o)
        else:
            flat_objs.append(o)

    # NOTE: `part.material` returns a fresh Python wrapper frequently, so `id(part.material)`
    # is NOT stable and can produce false positives. SAPIEN's RenderMaterial implements `==`
    # to compare underlying materials, so we use that to group materials robustly.
    material_groups: list[tuple[sapien.render.RenderMaterial, set[int]]] = []
    for obj_i, obj in enumerate(flat_objs):
        render_body_component = obj.find_component_by_type(RenderBodyComponent)
        if not isinstance(render_body_component, RenderBodyComponent):
            continue
        for render_shape in render_body_component.render_shapes:
            for part in render_shape.parts:
                m = part.material
                found = False
                for rep, obj_is in material_groups:
                    if m == rep:
                        obj_is.add(obj_i)
                        found = True
                        break
                if not found:
                    material_groups.append((m, {obj_i}))

    shared = [(repr(rep), sorted(list(obj_is))) for rep, obj_is in material_groups if len(obj_is) > 1]
    if not shared:
        return

    msg = (
        f"[ColosseumV2][MaterialUniqueness] Actor '{actor_name}' appears to share one or more "
        f"RenderMaterial instances across parallel env objects. This can make per-env color/texture "
        f"randomization collapse to identical visuals (last assignment wins). "
        f"Example shared materials/obj indices: {shared[:3]}{' ...' if len(shared) > 3 else ''}. "
        f"Fix by creating a fresh RenderMaterial per env instance (e.g., inside builder_fn) or cloning "
        f"materials per render part before mutation. "
    )
    raise RuntimeError(msg)


def _set_color_or_texture_objs(objs: list, actor_name: str, color_cfg: dict | None, texture_cfg: dict | None, set_color: bool, set_texture: bool, use_single_texture_or_texture: bool = False):
    if not set_color and not set_texture:
        return

    # Some call sites provide nested lists (e.g., cabinets/shelves). Flatten defensively.
    flat_objs = []
    for o in objs:
        if isinstance(o, (list, tuple)):
            flat_objs.extend(o)
        else:
            flat_objs.append(o)

    _warn_or_raise_if_shared_materials(flat_objs, actor_name, set_color=set_color, set_texture=set_texture)

    # The following code is borrowed from here: https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/domain_randomization.html
    for obj in flat_objs:

        # modify the i-th object which is in parallel environment i
        if use_single_texture_or_texture:
            # Pick one color/texture per parallel env instance (obj), and apply it to all
            # render parts of that env instance.
            if set_texture:
                assert texture_cfg is not None, "texture_cfg is not set"
                texture = _get_random_texture(texture_cfg["textures_directory"])
            if set_color:
                assert color_cfg is not None, "color_cfg is not set"
                color = color_cfg["color_range"].sample_rgba()
            if set_color and set_texture:
                use_color = np.random.random() < 0.5

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
                    # If a GLB came with a base-color texture, setting only base_color often
                    # won't visually change anything (the texture dominates). Clear textures
                    # so "color randomization" actually produces a solid color.
                    part.material.set_base_color_texture(None)
                    part.material.set_diffuse_texture(None)
                    part.material.set_base_color(color)
                elif not set_color and set_texture:
                    # Avoid accidental tinting from a previously set base_color.
                    part.material.set_base_color([1, 1, 1, 1])
                    part.material.set_base_color_texture(texture)
                elif set_color and set_texture:
                    if (use_single_texture_or_texture and use_color) or (not use_single_texture_or_texture and (np.random.random() < 0.5)):
                        part.material.set_base_color_texture(None)
                        part.material.set_diffuse_texture(None)
                        part.material.set_base_color(color)
                    else:
                        part.material.set_base_color([1, 1, 1, 1])
                        part.material.set_base_color_texture(texture)


def _set_color_or_texture(actor: Actor, color_cfg: dict | None, texture_cfg: dict | None, set_color: bool, set_texture: bool, use_single_texture_or_texture: bool = False):

    assert actor is not None, "actor must be provided, got None"

    if not set_color and not set_texture:
        return

    objs = []
    if isinstance(actor, OpenCabinet):
        for shelf in actor.shelves:
            objs.append(shelf._objs)
    elif isinstance(actor, Articulation):
        for obj in actor._objs:
            for link in obj.links:
                objs.append(link.entity)
        raise ValueError(f"Articulation {actor.name} is not supported for color/texture randomization")
    else:
        objs = actor._objs

    _set_color_or_texture_objs(
        objs=objs,
        actor_name=getattr(actor, "name", str(actor)),
        color_cfg=color_cfg,
        texture_cfg=texture_cfg,
        set_color=set_color,
        set_texture=set_texture,
        use_single_texture_or_texture=use_single_texture_or_texture,
    )




def _get_random_texture(texture_dir: str):
    texture_files = [f for f in os.listdir(texture_dir) if f.endswith('.png')]
    texture_file = np.random.choice(texture_files)
    return sapien.render.RenderTexture2D(filename=os.path.join(texture_dir, texture_file))



class ColosseumV2Env(BaseEnv):

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs
    ):
        env_id: str | None = kwargs.pop("_env_id", None)
        assert env_id is not None, "env_id must be provided"
        self._env_id = env_id

        max_n_distractor_objects = kwargs.pop("max_n_distractor_objects", 1000)

        # 
        distraction_set: DistractionSet | dict | None = kwargs.pop("distraction_set", None)
        if distraction_set is None:
            self._ds = DistractionSet()
        elif isinstance(distraction_set, dict):
            self._ds = DistractionSet(**distraction_set)
        elif isinstance(distraction_set, DistractionSet):
            self._ds = distraction_set
        else:
            raise ValueError(f"Invalid distraction set type: {type(distraction_set)}")
        if max_n_distractor_objects is not None and self._ds.distractor_object_enabled():
            self._ds.distractor_object_cfg["n_distractors"] = min(max_n_distractor_objects, self._ds.distractor_object_cfg["n_distractors"])

        # Verify that the variation factors are consistent
        if hasattr(self, "DISABLED_VARIATION_FACTORS"):
            dvf = self.DISABLED_VARIATION_FACTORS
            assert isinstance(dvf, DisabledVariationFactors)
            all_enabled = self._ds.all_are_enabled()
            for disabled in dvf.to_list():
                var_is_enabled = self._ds.variation_is_enabled(disabled)
                if all_enabled:
                    msg = (
                        f"Warning: variation '{disabled.upper()}' is disabled by the env, but enabled in the "
                        "distraction_set. However, 'all' variation factors are enabled, so this variation is disabled."
                    )
                    cprint(msg, "yellow")
                    self._ds.disable_variation_factors([disabled])
                else:
                    if var_is_enabled:
                        raise VariationFactorDisabledError(f"Variation {disabled} is enabled in distraction_set but is disabled by env")

        # 
        self._human_render_shader = kwargs.pop("human_render_shader", "default")

        # We will use self._table_scenes if the table color or texture is enabled, otherwise the single table_scene
        # self._table_scenes: list[TableSceneBuilder] = []
        self._table_scene_builders: list[TableSceneBuilder] = []
        self._table: Actor | None = None

        self._language_randomizations: dict[str, list[str]] = {}
        if self._ds.language_enabled():
            self._language_randomizations = load(open(self._ds.language_cfg["randomization_file"]), Loader=SafeLoader)

        self._robot_uids = robot_uids
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
            # self.scene.set_ambient_light([0.3, 0.3, 0.3])
            # self.scene.add_directional_light(
            #     [1, 1, -1], [1, 1, 1], shadow=shadow, shadow_scale=5, shadow_map_size=2048
            # )
            # self.scene.add_directional_light([0, 0, -1], [1, 1, 1])
            # ^ maniskill default
            for sub_scene in self.scene.sub_scenes:
                ambient_light_scales = [
                    np.random.uniform(*self._ds.light_color_cfg["ambient_light_scale_range"]),
                    np.random.uniform(*self._ds.light_color_cfg["ambient_light_scale_range"]),
                    np.random.uniform(*self._ds.light_color_cfg["ambient_light_scale_range"]),
                ]
                sub_scene.ambient_light = [
                    scale * 0.3 for scale in ambient_light_scales
                ]
                # ^ 0.3 is the default ambient light color

                theta_rand = np.random.uniform(0, 2*np.pi)
                color = self._ds.light_color_cfg["color_range"].sample_rgba(include_alpha=False)
                sub_scene.add_directional_light(
                    direction=[-np.cos(theta_rand), -np.sin(theta_rand), -1],
                    position=[np.cos(theta_rand), np.sin(theta_rand), 1],
                    color=color,
                    shadow=True,
                    shadow_map_size=1024
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
            rpy_low = torch.tensor(rpy_range[0], device=self.device, dtype=torch.float32)
            rpy_high = torch.tensor(rpy_range[1], device=self.device, dtype=torch.float32)
            rpy = torch.rand((self.num_envs, 3), device=self.device) * (rpy_high - rpy_low) + rpy_low
            delta_quat = matrix_to_quaternion(euler_angles_to_matrix(rpy, convention="XYZ"))
            xyz_low = torch.tensor(xyz_range[0], device=self.device, dtype=torch.float32)
            xyz_high = torch.tensor(xyz_range[1], device=self.device, dtype=torch.float32)
            delta_xyz = torch.rand((self.num_envs, 3), device=self.device) * (xyz_high - xyz_low) + xyz_low
            cfg.pose = cfg.pose * Pose.create_from_pq(p=delta_xyz, q=delta_quat)

        return cfgs

    def _add_articulation_to_scene(self, get_builder_fn: Callable[[], ArticulationBuilder], name: str, physics_type: str, object_type: str) -> Articulation:
        assert physics_type == "articulation", "physics_type must be articulation"

        articulations = []
        for i in range(self.num_envs):
            name_i = f"{name}_env:{i}"
            builder: ArticulationBuilder = get_builder_fn()
            builder.set_scene_idxs([i])
            articulation = builder.build(name=name_i)

            if object_type.upper() == "MO" and self._ds.MO_mass_enabled():
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
        assert object_type in ["MO", "RO", "DISTRACTOR", "BACKGROUND"]
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
        initial_pose: sapien.Pose = sapien.Pose(),
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
        builder.set_initial_pose(initial_pose)
        return builder


    # Borrowed from mani_skill/utils/building/actors/ycb.py:get_ycb_builder(...)
    def get_ycb_asset_builder(
        self,
        ycb_id: str,
        object_type: str,
        physical_material: PhysxMaterial | None = None,
        density: float | None = None,
        initial_pose: sapien.Pose | None = None,
    ):
        assert "ycb:" not in ycb_id, "ycb_id shouldn't contain 'ycb:'. Remove that substring if it does"
        builder = self.scene.create_actor_builder()

        try:
            model_db = load_json(ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json")
        except FileNotFoundError:
            cprint(f"YCB model dataset isn't downloaded. Run: 'python mani_skill/utils/download_asset.py ycb'", "red")
            exit(1)


        metadata = model_db[ycb_id]
        if density is None:
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
        if initial_pose is not None:
            builder.set_initial_pose(initial_pose)
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
        assert object_type in ["MO", "RO"], f"object_type must be MO or RO, got {object_type}"

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
        collision_kwargs = {"decomposition": "coacd", "scale": scale, "pose": mesh_pose}
        if physical_material is not None:
            collision_kwargs["material"] = physical_material
        if density is not None:
            collision_kwargs["density"] = density
        builder.add_multiple_convex_collisions_from_file(glb_filepath, **collision_kwargs)

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


    def get_MO_scale(self) -> tuple[float, float, float]:
        if self._ds.MO_size_enabled():
            scale_multiplier = self._ds.MO_size_cfg["scale_range"]
            scale = np.random.uniform(*scale_multiplier)
            return (scale, scale, scale)
        return (1.0, 1.0, 1.0)

    def load_scene_hook(self, manipulation_objects: list[Actor] = [], receiving_objects: list[Actor] = [], add_table_to_scene: bool = True):
        """
        This function is called when the scene is loaded.
        Args:
            manipulation_objects (list[Actor]): The manipulation objects to modify.
            receiving_objects (list[Actor]): The receiving objects to modify.
        """
        # First verify that the variation factors match
        mo_variation_enabled = self._ds.MO_color_enabled() or self._ds.MO_texture_enabled() or self._ds.MO_size_enabled()
        ro_variation_enabled = self._ds.RO_color_enabled() or self._ds.RO_texture_enabled() or self._ds.RO_size_enabled()
        if mo_variation_enabled and len(manipulation_objects) == 0:
            raise VariationFactorDisabledError("MO variation factors are enabled, but no manipulation_objects is provided.")
        if ro_variation_enabled and len(receiving_objects) == 0:
            raise VariationFactorDisabledError("RO variation factors are enabled, but no receiving_objects is provided.")

        # 
        self._load_scene_hool_called = True

        # New distractor objs
        # TODO: Add YCB objects
        if self._ds.distractor_object_enabled():
            n_distractors = self._ds.distractor_object_cfg["n_distractors"]
            self._ds._internal["distractor_object_cfg"]["actors"] = []
            for i in range(n_distractors):
                def get_ycb_builder():
                    return self.get_ycb_asset_builder(
                        ycb_id=YCB_DISTRACTOR_OBJECTS[random.randint(0, len(YCB_DISTRACTOR_OBJECTS) - 1)],
                        object_type="DISTRACTOR",
                        initial_pose=sapien.Pose(),
                    )

                self._ds._internal["distractor_object_cfg"]["actors"].append(
                    self.add_asset_to_scene(get_ycb_builder, name=f"distractor_obj_{i}", physics_type="dynamic", object_type="DISTRACTOR")
                )


        if add_table_to_scene:
            self._add_table_to_scene()

        # DualPanda includes a `table` link inside its URDF (`dual_panda_table.urdf`).
        # If users enable table color/texture randomization, apply it to that link's visuals as well.
        if self._robot_uids == "dual_panda" and (self._ds.table_color_enabled() or self._ds.table_texture_enabled()):
            table_link = self.agent.robot.links_map.get("table", None)
            # PhysX link components have an `.entity` that holds render components.
            table_entities = [o.entity for o in table_link._objs]
            _set_color_or_texture_objs(
                objs=table_entities,
                actor_name="dual_panda:table_link",
                color_cfg=self._ds.table_color_cfg,
                texture_cfg=self._ds.table_texture_cfg,
                set_color=self._ds.table_color_enabled(),
                set_texture=self._ds.table_texture_enabled(),
                use_single_texture_or_texture=True,
            )

        # Manipulation object
        for mo in manipulation_objects:
            assert mo is not None, "mo must be provided, got None"
            _set_color_or_texture(mo, self._ds.MO_color_cfg, self._ds.MO_texture_cfg, self._ds.MO_color_enabled(), self._ds.MO_texture_enabled())

        # Receiving object
        for ro in receiving_objects:
            assert ro is not None, "ro must be provided, got None"
            _set_color_or_texture(ro, self._ds.RO_color_cfg, self._ds.RO_texture_cfg, self._ds.RO_color_enabled(), self._ds.RO_texture_enabled())

        if self._ds.background_texture_enabled() or self._ds.background_color_enabled():

            def get_builder_fn():
                # Build a single static "compound" wall actor (with 4 box segments) so we only
                # have one wall actor to manage/texture/etc. The actor itself is placed at the
                # origin; each segment uses a local pose.
                builder_wall = self.scene.create_actor_builder()
                dist_from_world = 1.5
                height = 2
                width = 0.1
                length = dist_from_world*2
                FLOOR_HEIGHT = -0.920
                z = FLOOR_HEIGHT + (height / 2)

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
                return builder_wall
            
            self._background_wall = self.add_asset_to_scene(get_builder_fn, name="background_wall", physics_type="static", object_type="BACKGROUND")
            assert isinstance(self._background_wall, Actor), "self._background_wall must be an Actor"
            _set_color_or_texture(
                actor=self._background_wall,
                color_cfg=self._ds.background_color_cfg,
                texture_cfg=self._ds.background_texture_cfg,
                set_color=self._ds.background_color_enabled(), 
                set_texture=self._ds.background_texture_enabled(),
                use_single_texture_or_texture=True,
            )


    def initialize_episode_hook(
        self, 
        env_idx: torch.Tensor, 
        mo_pose: torch.Tensor | sapien.Pose | Pose | None = None, 
        ro_pose: torch.Tensor | None = None, 
        qpos_0: np.ndarray | None = None, 
        initialize_table_scene: bool = True, 
        table_z_rotation_angle: float | None = None,
        distractor_object_bounds: PlacementRegion | None = None,
        distractor_object_height: float = 0.25,
        max_n_distractor_objects: int = 2,
    ):

        assert self._load_scene_hool_called, "load_scene_hook must be called before initialize_episode_hook"

        if mo_pose is not None:
            assert mo_pose.shape[0] == self.num_envs
            assert mo_pose.shape[1] >= 2, f"mo_pose must have at least 2 dimensions, got {mo_pose.shape[1]}"
        if ro_pose is not None:
            assert ro_pose.shape[0] == self.num_envs
            assert ro_pose.shape[1] >= 2, f"ro_pose must have at least 2 dimensions, got {ro_pose.shape[1]}"


        if initialize_table_scene:
            init_kwargs = {}
            if table_z_rotation_angle is not None:
                init_kwargs["table_z_rotation_angle"] = table_z_rotation_angle
            if qpos_0 is not None:
                init_kwargs["qpos_0"] = qpos_0
            for ts in self._table_scene_builders:
                ts.initialize(env_idx, **init_kwargs)

        # TODO: Make sure that the sampled poses are beyond some epsilon of RO/ro objects
        if self._ds.distractor_object_enabled():

            for i in range(min(max_n_distractor_objects, self._ds.distractor_object_cfg["n_distractors"])):
                # What happens if you set the poses such that the objs collide with one another?
                # for i, obj in enumerate(self._ds._internal["distractor_object_cfg"]["obj_actors"]):
                if distractor_object_bounds is None:
                    region = PlacementRegion(
                        x_lims=tuple(self._ds.distractor_object_cfg["x_lims"]),
                        y_lims=tuple(self._ds.distractor_object_cfg["y_lims"]),
                    )
                else:
                    region = distractor_object_bounds
                xyz = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
                xyz[:, :2] = region.sample_xy(self.num_envs, device=self.device)
                xyz[:, 2] = distractor_object_height # 
                if mo_pose is not None:
                    if isinstance(mo_pose, torch.Tensor):
                        xyz[:, 2] += mo_pose[:, 2] # add the height of the MO
                    elif isinstance(mo_pose, (sapien.Pose, Pose)):
                        xyz[:, 2] += mo_pose.p[:, 2] # add the height of the MO
                    else:
                        raise ValueError(f"mo_pose must be a torch.Tensor or sapien.Pose, got {type(mo_pose)}")
                self._ds._internal["distractor_object_cfg"]["actors"][i].set_pose(Pose.create_from_pq(p=xyz))

    def _get_obs_extra(self, info: dict):
        if self._robot_uids == "dual_panda":
            return dict(
                left_arm_tcp=self.agent.tcp_1_pose.raw_pose,
                right_arm_tcp=self.agent.tcp_2_pose.raw_pose,
            )
        return dict(tcp_pose=self.agent.tcp_pose.raw_pose)

    def update_language_instructions(self, language_instructions: list[str] | None) -> list[str] | None:
        if language_instructions is None:
            return None
        if self._ds.language_enabled():
            return [
                random.choice(self._language_randomizations[self._env_id]) for _ in range(len(language_instructions))
            ]
        return language_instructions

    def update_placement_region(self, region: PlacementRegion):
        """
        Updates the default placement regions if 
        Args:
            placement_region (PlacementRegion): The placement region to update.
        """
        if not self._ds.pose_randomization_enabled():
            return region

        region_cp = PlacementRegion(x_lims=region.x_lims, y_lims=region.y_lims)
        x_region_multiplier = self._ds.pose_randomization_cfg["x_region_multiplier"]
        y_region_multiplier = self._ds.pose_randomization_cfg["y_region_multiplier"]
        center_x = (region.x_lims[0] + region.x_lims[1]) / 2
        center_y = (region.y_lims[0] + region.y_lims[1]) / 2
        width_x = (region.x_lims[1] - region.x_lims[0])
        width_y = (region.y_lims[1] - region.y_lims[0])

        # Set a minumum width
        if width_x < 1e-6:
            width_x = self._ds.pose_randomization_cfg["min_width_x"]
        if width_y < 1e-6:
            width_y = self._ds.pose_randomization_cfg["min_width_y"]

        new_width_x = width_x * x_region_multiplier
        new_width_y = width_y * y_region_multiplier
        region_cp.x_lims = (center_x - (new_width_x/2), center_x + (new_width_x/2))
        region_cp.y_lims = (center_y - (new_width_y/2), center_y + (new_width_y/2))
        return region_cp