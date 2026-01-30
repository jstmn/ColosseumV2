from dataclasses import dataclass, field
import os

from mani_skill import PACKAGE_ASSET_DIR

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
    Distractor object   | Spawns a random object in the workspace of the robot.
    Background texture  | Modifies the textures applied to the walls of the scene.
    Camera pose         | Randomly perturbs the pose of a camera.

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
    distractor_object_cfg: dict = field(default_factory=dict)
    background_texture_cfg: dict = field(default_factory=dict)
    camera_pose_cfg: dict = field(default_factory=dict)

    unimplemented = {
        "MO_size",
        "RO_size",
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

    def table_texture_enabled(self) -> bool:
        return len(self.table_texture_cfg) > 0

    def light_color_enabled(self) -> bool:
        return len(self.light_color_cfg) > 0

    def distractor_object_enabled(self) -> bool:
        return len(self.distractor_object_cfg) > 0

    def background_texture_enabled(self) -> bool:
        return len(self.background_texture_cfg) > 0

    def camera_pose_enabled(self) -> bool:
        return len(self.camera_pose_cfg) > 0

    def which_enabled_str(self) -> tuple[list[str], list[str]]:
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
            "distractor_object_cfg",
            "background_texture_cfg",
            "camera_pose_cfg",
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
        if self.distractor_object_enabled():
            assert_range_correct(self.distractor_object_cfg["color_range"])
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
            distractor_object_cfg=self.distractor_object_cfg,
            background_texture_cfg=self.background_texture_cfg,
            camera_pose_cfg=self.camera_pose_cfg,
        )


all_distractor_set = DistractionSet(
    distractor_object_cfg={
        "n_spheres": 1,
        "radius_range": (0.01, 0.02),"color_range": ((0, 0, 0), (1, 1, 1)),
        "x_lims": (-0.1, 0.1),
        "y_lims": (-0.1, 0.1),
    },
    MO_color_cfg ={
        "color_range": ((0, 0, 0), (1, 1, 1)),
    },
    MO_texture_cfg = {
        "textures_directory": os.path.join(PACKAGE_ASSET_DIR, "textures"),
    },
    RO_color_cfg ={
        "color_range": ((0, 0, 0), (1, 1, 1)),
    },
    RO_texture_cfg = {
        "textures_directory": os.path.join(PACKAGE_ASSET_DIR, "textures"),
    },
    table_color_cfg = {
        "color_range": ((0, 0, 0), (1, 1, 1)),
    },
    table_texture_cfg = {
        "textures_directory": os.path.join(PACKAGE_ASSET_DIR, "textures"),
    },
    camera_pose_cfg = {
        "rpy_range": ((-0.035, -0.035, -0.035), (0.035, 0.035, 0.035)), # aproximately 2 degrees
        "xyz_range": ((-0.025, -0.025, 0.025), (0.025, 0.025, 0.025)),        # 2.5 cm
    },
    light_color_cfg = {
        "ambient_light_range": (0.25, 0.75),
    },
)

DISTRACTION_SETS = {
    "none".upper(): DistractionSet(),
    "all".upper(): all_distractor_set,
    "distractor_object".upper(): all_distractor_set.get_partial_copy(["distractor_object_cfg"]),
    "MO_color".upper(): all_distractor_set.get_partial_copy(["MO_color_cfg"]),
    "MO_texture".upper(): all_distractor_set.get_partial_copy(["MO_texture_cfg"]),
    "RO_color".upper(): all_distractor_set.get_partial_copy(["RO_color_cfg"]),
    "RO_texture".upper(): all_distractor_set.get_partial_copy(["RO_texture_cfg"]),
    "table_color".upper(): all_distractor_set.get_partial_copy(["table_color_cfg"]),
    "table_texture".upper(): all_distractor_set.get_partial_copy(["table_texture_cfg"]),
    "camera_pose".upper(): all_distractor_set.get_partial_copy(["camera_pose_cfg"]),
}