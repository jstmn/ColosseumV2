from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional
import os
import random

from matplotlib.pylab import geometric
import numpy as np
import torch
import sapien
import numpy as np
from sapien.render import RenderBodyComponent
from transforms3d.euler import euler2quat

from mani_skill.sensors.camera import CameraConfig
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.building import actors
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.actor import Actor
from mani_skill.envs.utils.randomization import enhanced_distractors
from mani_skill.envs.utils.randomization.enhanced_distractors import EnhancedDistractorManager, TextureManager

@dataclass
class DistractionSet:
    """
    Factor of Variation | Description
    ---------------------------------
    MO color            | Modifies the color of the MO
    RO color            | Modifies the color of the RO (if applicable)
    MO texture          | Modifies the texture applied to the MO
    RO texture          | Modifies the texture applied to the RO (if applicable)
    MO size             | Scales the MO by a given factor
    RO size             | Scales the RO (if applicable) by a given factor
    Table color         | Modifies the color of the tabletop of the robot setup
    Light color         | Modifies the color of the lights setup in the scene.
    Table texture       | Modifies the texture applied to the tabletop of the robot setup.
    Background texture  | Modifies the textures applied to the walls of the scene.
    Camera pose         | Randomly perturbs the pose of a camera.
    Enhanced distractors| Spawns customizable enhanced distractor objects in the workspace.

    from https://robot-colosseum.readthedocs.io/en/latest/overview.html
    """
    MO_color_cfg: dict = field(default_factory=dict)
    RO_color_cfg: dict = field(default_factory=dict)
    MO_texture_cfg: dict = field(default_factory=dict)
    RO_texture_cfg: dict = field(default_factory=dict)
    MO_size_cfg: dict = field(default_factory=dict)
    RO_size_cfg: dict = field(default_factory=dict)
    table_color_cfg: dict = field(default_factory=dict)
    light_color_cfg: dict = field(default_factory=dict)
    table_texture_cfg: dict = field(default_factory=dict)
    background_texture_cfg: dict = field(default_factory=dict)
    camera_pose_cfg: dict = field(default_factory=dict)
    enhanced_distractor_cfg: dict = field(default_factory=dict)  # Enhanced distractor configuration

    unimplemented = {
        "RO_color",
        "RO_texture",
        "MO_size",
        "RO_size",
        "light_color",
        "background_texture"
    }

    def get_partial_copy(self, keys: list[str]) -> "DistractionSet":
        return DistractionSet(**{k: v for k, v in self.__dict__.items() if k in keys})

    def MO_color_enabled(self) -> bool:
        return len(self.MO_color_cfg) > 0

    def RO_color_enabled(self) -> bool:
        return len(self.RO_color_cfg) > 0

    def MO_texture_enabled(self) -> bool:
        return len(self.MO_texture_cfg) > 0

    def RO_texture_enabled(self) -> bool:
        return len(self.RO_texture_cfg) > 0

    def MO_size_enabled(self) -> bool:
        return len(self.MO_size_cfg) > 0

    def RO_size_enabled(self) -> bool:
        return len(self.RO_size_cfg) > 0

    def table_color_enabled(self) -> bool:
        return len(self.table_color_cfg) > 0

    def light_color_enabled(self) -> bool:
        return len(self.light_color_cfg) > 0

    def table_texture_enabled(self) -> bool:
        return len(self.table_texture_cfg) > 0

    def background_texture_enabled(self) -> bool:
        return len(self.background_texture_cfg) > 0

    def camera_pose_enabled(self) -> bool:
        return len(self.camera_pose_cfg) > 0

    def enhanced_distractor_enabled(self) -> bool:
        return len(self.enhanced_distractor_cfg) > 0

    def which_enabled_str(self) -> list[str]:
        enabled_strs = []
        disabled_strs = []
        for k in [attr for attr in dir(self) if not attr.startswith('_')]:
            if k.endswith('_enabled') and hasattr(self, k):
                enabled_fn = getattr(self, k)
                if enabled_fn():
                    enabled_strs.append(k[:-8]) # Remove '_enabled' suffix and append
                else:
                    disabled_strs.append(k[:-8])
        return enabled_strs, disabled_strs

    def __post_init__(self):
        self._internal = {}
        for key in [
            "MO_color_cfg",
            "RO_color_cfg",
            "MO_texture_cfg",
            "RO_texture_cfg",
            "MO_size_cfg",
            "RO_size_cfg",
            "table_color_cfg",
            "light_color_cfg",
            "table_texture_cfg",
            "background_texture_cfg",
            "camera_pose_cfg",
            "enhanced_distractor_cfg",
        ]:
            self._internal[key] = {}

        for k in self.which_enabled_str()[0]:
            assert k not in self.unimplemented, f"Distractor {k} is enabled but in the unimplemented set"

        def assert_range_correct(range: tuple):
            assert len(range) == 2, "range must be a tuple of two values"
            assert len(range[0]) == 3, "range[0] must be a tuple of three values"
            assert len(range[1]) == 3, "range[1] must be a tuple of three values"
            assert range[0][0] <= range[1][0], "range[0][0] must be less than range[1][0]"
            assert range[0][1] <= range[1][1], "range[0][1] must be less than range[1][1]"
            assert range[0][2] <= range[1][2], "range[0][2] must be less than range[1][2]"

        if self.camera_pose_enabled():
            assert_range_correct(self.camera_pose_cfg["rpy_range"])
            assert_range_correct(self.camera_pose_cfg["xyz_range"])
        if self.table_color_enabled():
            assert_range_correct(self.table_color_cfg["color_range"])

    def to_dict(self):
        return dict(
            MO_color_cfg=self.MO_color_cfg,
            RO_color_cfg=self.RO_color_cfg,
            MO_texture_cfg=self.MO_texture_cfg,
            RO_texture_cfg=self.RO_texture_cfg,
            MO_size_cfg=self.MO_size_cfg,
            RO_size_cfg=self.RO_size_cfg,
            table_color_cfg=self.table_color_cfg,
            light_color_cfg=self.light_color_cfg,
            table_texture_cfg=self.table_texture_cfg,
            background_texture_cfg=self.background_texture_cfg,
            camera_pose_cfg=self.camera_pose_cfg,
            enhanced_distractor_cfg=self.enhanced_distractor_cfg,
        )

    def update_camera_configs(self, cfgs: list[CameraConfig]) -> list[CameraConfig]:
        if not self.camera_pose_enabled():
            return cfgs

        rpy_range = self.camera_pose_cfg["rpy_range"]
        xyz_range = self.camera_pose_cfg["xyz_range"]

        for cfg in cfgs:
            rpy = np.random.uniform(*rpy_range)
            xyz = np.random.uniform(*xyz_range)
            delta_pose = sapien.Pose(p=xyz, q=euler2quat(rpy[0], rpy[1], rpy[2]))
            cfg.pose *= delta_pose

        return cfgs

    def load_scene_hook(self, scene: ManiSkillScene, manipulation_object: Optional[Actor], table: Optional[Actor], manipulation_object_size):
        """
        This function is called when the scene is loaded.
        Args:
            scene (ManiSkillScene): The scene to modify.
            manipulation_object (Optional[Actor]): The manipulation object to modify. Note that this is a wrapper around
            table (Optional[Actor]): The table object in the scene.
            manipulation_object_size: The size (half-size) of the manipulation object
        """

        # Add enhanced distractors if enabled
        if self.enhanced_distractor_enabled() and table is not None and manipulation_object is not None:

            cube_pose = manipulation_object.pose
            cube_pos = cube_pose.p.cpu().numpy()[0]  # Get first env's position
            
            # Create enhanced distractors using the EnhancedDistractorManager class
            internal_objects = EnhancedDistractorManager.create_enhanced_distractors(
                scene=scene,
                manipulation_obj_pos=cube_pos,
                cfg=self.enhanced_distractor_cfg,
                manipulation_object_size=manipulation_object_size
            )
            
            # Store the objects for the initialize_episode_hook
            self._internal["enhanced_distractor_cfg"]["internal__objects"] = internal_objects

        def get_random_color(color_range: tuple):
            assert (len(color_range) == 2) and (len(color_range[0]) == 3) and (len(color_range[1]) == 3), "color_range must be a tuple of two tuples of three floats"
            return np.random.uniform(*color_range).tolist() + [1]
            
        def get_random_texture(texture_dir: str):
            # Use TextureManager for consistent texture handling
            return TextureManager.get_random_texture(texture_dir)

        # Set table color and texture
        if (table is not None) and (self.table_color_enabled() or self.table_texture_enabled()):
            assert isinstance(table, Actor), "table must be a ManiSkill Actor, is {}".format(type(table))
            # The following code is borrowed from here: https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/domain_randomization.html
            for obj in table._objs:
                # modify the i-th object which is in parallel environment i
                render_body_component: RenderBodyComponent = obj.find_component_by_type(RenderBodyComponent)
                for render_shape in render_body_component.render_shapes:
                    for part in render_shape.parts:
                        # part.material: sapien.core.pysapien.RenderMaterial
                        if self.table_texture_enabled():
                            texture = get_random_texture(self.table_texture_cfg["textures_directory"])
                        if self.table_color_enabled():
                            color = get_random_color(self.table_color_cfg["color_range"])

                        if self.table_color_enabled() and not self.table_texture_enabled():
                            part.material.set_base_color(color)
                        elif self.table_texture_enabled() and not self.table_color_enabled():
                            part.material.set_base_color_texture(texture)
                        else:
                            use_color = np.random.random() < 0.5
                            if use_color:
                                part.material.set_base_color(color)
                            else:
                                part.material.set_base_color_texture(texture)

        # Set manipulation object color and texture
        if (manipulation_object is not None) and (self.MO_color_enabled() or self.MO_texture_enabled()):
            assert isinstance(manipulation_object, Actor), "manipulation_object must be a ManiSkill Actor, is {}".format(type(manipulation_object))

            # The following code is borrowed from here: https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/domain_randomization.html
            for obj in manipulation_object._objs:
                # modify the i-th object which is in parallel environment i
                render_body_component: RenderBodyComponent = obj.find_component_by_type(RenderBodyComponent)
                for render_shape in render_body_component.render_shapes:
                    for part in render_shape.parts:
                        # part.material: sapien.core.pysapien.RenderMaterial
                        if self.MO_color_enabled():
                            color = get_random_color(self.MO_color_cfg["color_range"])
                        if self.MO_texture_enabled():
                            texture = get_random_texture(self.MO_texture_cfg["textures_directory"])

                        if self.MO_color_enabled() and not self.MO_texture_enabled():
                            part.material.set_base_color(color)
                        elif self.MO_texture_enabled() and not self.MO_color_enabled():
                            part.material.set_base_color_texture(texture)
                        else:
                            use_color = np.random.random() < 0.5
                            if use_color:
                                part.material.set_base_color(color)
                            else:
                                part.material.set_base_color_texture(texture)

    def initialize_episode_hook(self, n_envs: int, mo_pose: torch.Tensor):
        """Set positions of all objects at episode start."""
        assert mo_pose.shape == (n_envs, 3), f"mo_pose must be of shape (n_envs, 3), got {mo_pose.shape}"
        
        # Position enhanced distractors if enabled
        if self.enhanced_distractor_enabled():
            if "enhanced_distractor_cfg" in self._internal:
                if "internal__objects" in self._internal["enhanced_distractor_cfg"]:
                    internal_objects = self._internal["enhanced_distractor_cfg"]["internal__objects"]
                    # Use EnhancedDistractorManager to position enhanced distractors
                    EnhancedDistractorManager.position_enhanced_distractors(internal_objects, n_envs)


_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "../assets")
all_distractor_set = DistractionSet(
    MO_color_cfg ={
        "color_range": ((0, 0, 0), (1, 1, 1)),
    },
    MO_texture_cfg = {
        "textures_directory": os.path.join(_ASSETS_DIR, "textures"),
    },
    table_color_cfg = {
        "color_range": ((0, 0, 0), (1, 1, 1)),
    },
    table_texture_cfg = {
        "textures_directory": os.path.join(_ASSETS_DIR, "textures"),
    },
    camera_pose_cfg = {
        "rpy_range": ((-0.035, -0.035, -0.035), (0.035, 0.035, 0.035)), # aproximately 2 degrees
        "xyz_range": ((-0.025, -0.025, 0.025), (0.025, 0.025, 0.025)),        # 2.5 cm
    },
    enhanced_distractor_cfg={
        # Sometimes less than 4 objects may be spawned due to overlaps with other distractors objects
        "max_objects": 4,
        "max_attempts": 100,
        "textures_directory": os.path.join(_ASSETS_DIR, "textures"),
        "table_bounds": {
            "x_min": -0.20,
            "x_max": 0.20,
            "y_min": -0.20,
            "y_max": 0.20
        },
        "cylinder": {
            "count": 2,
            "radius_range": (0.025, 0.035),  # Random radius between 2.5-3.5cm
            "height_range": (0.05, 0.07),    # Random height between 5-7cm
            "color_range": ((0.7, 0, 0), (1, 1, 1)),               
            "rotation_range": (0, np.pi/2),  # Random rotation 0-90 degrees around y-axis
        },
        "sphere": {
            "count": 2,
            "radius_range": (0.025, 0.035),  # Random radius between 2.5-3.5cm
                "color_range": ((0, 0, 0), (1, 1, 1)),
            },
    },
)

DISTRACTION_SETS = {
    "none".upper(): DistractionSet(),
    "dev".upper(): DistractionSet(
        MO_color_cfg ={
            "color_range": ((0, 0, 0), (1, 1, 1)),
        },
        MO_texture_cfg = {
            "textures_directory": os.path.join(_ASSETS_DIR, "textures"),
        },
        table_color_cfg = {
            "color_range": ((0, 0, 0), (1, 1, 1)),
        },
        table_texture_cfg = {
            "textures_directory": os.path.join(_ASSETS_DIR, "textures"),
        },
        camera_pose_cfg = {
            "rpy_range": ((-0.035, -0.035, -0.035), (0.035, 0.035, 0.035)),
            "xyz_range": ((-0.025, -0.025, 0.025), (0.025, 0.025, 0.025)),
        },
        enhanced_distractor_cfg={
            # Sometimes less than 4 objects may be spawned due to overlaps with other distractors objects
            "max_objects": 4,
            "max_attempts": 100,
            "textures_directory": os.path.join(_ASSETS_DIR, "textures"),
            "table_bounds": {
                "x_min": -0.20,
                "x_max": 0.20,
                "y_min": -0.20,
                "y_max": 0.20
            },
            "cylinder": {
                "count": 2,
                "radius_range": (0.025, 0.035),  # Random radius between 2.5-3.5cm
                "height_range": (0.05, 0.07),    # Random height between 5-7cm
                "color_range": ((0.7, 0, 0), (1, 1, 1)),
                "rotation_range": (0, np.pi/2),  # Random rotation 0-90 degrees around y-axis
            },
            "sphere": {
                "count": 2,
                "radius_range": (0.025, 0.035),  # Random radius between 2.5-3.5cm
                "color_range": ((0, 0, 0), (1, 1, 1)),
            },
        },
    ),
    "all".upper(): deepcopy(all_distractor_set),
    "MO_color_cfg".upper(): all_distractor_set.get_partial_copy(["MO_color_cfg"]),
    "MO_texture_cfg".upper(): all_distractor_set.get_partial_copy(["MO_texture_cfg"]),
    "table_color_cfg".upper(): all_distractor_set.get_partial_copy(["table_color_cfg"]),
    "table_texture_cfg".upper(): all_distractor_set.get_partial_copy(["table_texture_cfg"]),
    "camera_pose_cfg".upper(): all_distractor_set.get_partial_copy(["camera_pose_cfg"]),
    "enhanced_distractor_cfg".upper(): all_distractor_set.get_partial_copy(["enhanced_distractor_cfg"]),
}
