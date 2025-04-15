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
        Updated scene hook that works with multiple environments and handles enhanced distractors safely.
        Args:
            scene (ManiSkillScene): The scene to modify.
            manipulation_object (Optional[Actor]): The manipulation object to modify.  Note that this is a wrapper around a sapien.Entity.
            table (Optional[Actor]): The table object in the scene.
            manipulation_object_size: The size (half-size) of the manipulation object.
        """
        # Create enhanced distractors for each environment independently.
        if self.enhanced_distractor_enabled() and table is not None and manipulation_object is not None:
            distractors_all_envs = []
            # Loop over each env-specific manipulation object instance.
            for env_idx, obj in enumerate(manipulation_object._objs):
                pose = obj.get_pose()  # Get the individual pose for this env.
                try:
                    # Create distractors for this environment with the environment index.
                    distractors = EnhancedDistractorManager.create_enhanced_distractors(
                        scene=scene,
                        manipulation_obj_pos=pose.p,
                        cfg=self.enhanced_distractor_cfg,
                        manipulation_object_size=manipulation_object_size,
                        env_index=env_idx
                    )
                except Exception as e:
                    distractors = None
                distractors_all_envs.append(distractors)
            # Store all distractor objects per environment.
            self._internal["enhanced_distractor_cfg"]["internal__objects"] = distractors_all_envs

        def get_random_color(color_range: tuple):
            return np.random.uniform(*color_range).tolist() + [1]

        def get_random_texture(texture_dir: str):
            return TextureManager.get_random_texture(texture_dir)

        if (table is not None) and (self.table_color_enabled() or self.table_texture_enabled()):
            assert isinstance(table, Actor), f"table must be a ManiSkill Actor, is {type(table)}"
            for obj in table._objs:
                render_body_component: RenderBodyComponent = obj.find_component_by_type(RenderBodyComponent)
                for render_shape in render_body_component.render_shapes:
                    for part in render_shape.parts:
                        if self.table_texture_enabled():
                            texture = get_random_texture(self.table_texture_cfg["textures_directory"])
                        if self.table_color_enabled():
                            color = get_random_color(self.table_color_cfg["color_range"])
                        if self.table_color_enabled() and not self.table_texture_enabled():
                            part.material.set_base_color(color)
                        elif self.table_texture_enabled() and not self.table_color_enabled():
                            part.material.set_base_color_texture(texture)
                        else:
                            if np.random.random() < 0.5:
                                part.material.set_base_color(color)
                            else:
                                part.material.set_base_color_texture(texture)

        if (manipulation_object is not None) and (self.MO_color_enabled() or self.MO_texture_enabled()):
            assert isinstance(manipulation_object, Actor), f"manipulation_object must be a ManiSkill Actor, is {type(manipulation_object)}"
            for obj in manipulation_object._objs:
                render_body_component: RenderBodyComponent = obj.find_component_by_type(RenderBodyComponent)
                for render_shape in render_body_component.render_shapes:
                    for part in render_shape.parts:
                        if self.MO_color_enabled():
                            color = get_random_color(self.MO_color_cfg["color_range"])
                        if self.MO_texture_enabled():
                            texture = get_random_texture(self.MO_texture_cfg["textures_directory"])
                        if self.MO_color_enabled() and not self.MO_texture_enabled():
                            part.material.set_base_color(color)
                        elif self.MO_texture_enabled() and not self.MO_color_enabled():
                            part.material.set_base_color_texture(texture)
                        else:
                            if np.random.random() < 0.5:
                                part.material.set_base_color(color)
                            else:
                                part.material.set_base_color_texture(texture)
                                

    def initialize_episode_hook(self, n_envs: int, mo_pose: torch.Tensor):
        """Set positions of all objects at episode start."""
        assert mo_pose.shape == (n_envs, 3), f"mo_pose must be of shape (n_envs, 3), got {mo_pose.shape}"

        if self.enhanced_distractor_enabled():
            if "enhanced_distractor_cfg" in self._internal and "internal__objects" in self._internal["enhanced_distractor_cfg"]:
                distractors_all_envs = self._internal["enhanced_distractor_cfg"]["internal__objects"]
                
                # For each environment, position its distractor objects
                for env_idx, internal_objects in enumerate(distractors_all_envs):
                    if internal_objects:
                        # Position each distractor for this environment
                        for obj_idx, obj_data in enumerate(internal_objects):
                            # Get the position for this specific environment
                            pos = torch.tensor(obj_data["position"], dtype=torch.float32).unsqueeze(0)

                            obj = obj_data["object"]

                            obj_type = "cylinder" if "y_rotation" in obj_data else "sphere"

                            if obj_type == "cylinder":
                                upright_q = obj_data["upright_rotation"]
                                y_rotation_q = obj_data["y_rotation"]
                                    
                                upright_qt = torch.tensor(upright_q, dtype=torch.float32).unsqueeze(0)
                                y_rotation_qt = torch.tensor(y_rotation_q, dtype=torch.float32).unsqueeze(0)
                                    
                                upright_pose = Pose.create_from_pq(p=pos, q=upright_qt)

                                y_rotation_pose = Pose.create_from_pq(
                                    p=torch.zeros((1, 3), dtype=torch.float32), 
                                    q=y_rotation_qt
                                )

                                final_pose = upright_pose * y_rotation_pose
                            else:
                                # For spheres, just set position
                                final_pose = Pose.create_from_pq(p=pos)
          
                            obj.set_pose(final_pose)


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
            "radius_range": (0.025, 0.035),
            "height_range": (0.05, 0.07),
            "color_range": ((0.7, 0, 0), (1, 1, 1)),               
            "rotation_range": (0, np.pi/2),
        },
        "sphere": {
            "count": 2,
            "radius_range": (0.025, 0.035),
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
                "radius_range": (0.025, 0.035),
                "height_range": (0.05, 0.07),
                "color_range": ((0.7, 0, 0), (1, 1, 1)),
                "rotation_range": (0, np.pi/2),
            },
            "sphere": {
                "count": 2,
                "radius_range": (0.025, 0.035),
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
