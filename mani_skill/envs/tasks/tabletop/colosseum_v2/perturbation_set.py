from dataclasses import dataclass, field
from pathlib import Path
from termcolor import cprint
import os
import numpy as np

from mani_skill import PACKAGE_ASSET_DIR


@dataclass
class ColorRange:
    low: tuple[float, float, float , float] | tuple[float, float, float]
    high: tuple[float, float, float, float]| tuple[float, float, float]

    def __post_init__(self):
        assert isinstance(self.low, tuple) and isinstance(self.high, tuple), "low and high must be tuples"
        assert len(self.low) == len(self.high), "low and high must have the same length"
        assert len(self.low) == 3 or len(self.low) == 4, "low and high must have 3 or 4 values"
        assert all(self.low[i] <= self.high[i] for i in range(len(self.low))), "low must be less than high for all values"

    def sample_rgba(self, include_alpha: bool = True):
        rgb = np.random.uniform(self.low, self.high).tolist()
        if include_alpha:
            return rgb
        else:
            return rgb[:3]

    def to_dict(self):
        return dict(
            low=self.low,
            high=self.high,
        )

@dataclass
class PerturbationSet:
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
    Language paraphrase | Replaces the instruction with a paraphrase of the same task.
    Language other task | Replaces the instruction with a paraphrase from a different task.
    Language random     | Replaces the instruction with N random English words.
    Language none       | Replaces the instruction with an empty string.

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
    background_color_cfg: dict = field(default_factory=dict)
    camera_pose_cfg: dict = field(default_factory=dict)
    # MO_mass_cfg: dict = field(default_factory=dict)
    language_paraphrase_cfg: dict = field(default_factory=dict)
    language_other_task_cfg: dict = field(default_factory=dict)
    language_random_cfg: dict = field(default_factory=dict)
    language_none_cfg: dict = field(default_factory=dict)
    pose_randomization_cfg: dict = field(default_factory=dict)

    unimplemented = {}

    def get_partial_copy(self, keys: list[str]) -> "PerturbationSet":
        return PerturbationSet(**{k: v for k, v in self.__dict__.items() if k in keys})

    def perturbation_is_enabled(self, perturbation: str) -> bool:
        if perturbation.endswith("_cfg"):
            perturbation = perturbation[:-4]
        
        enabled, disabled = self.which_enabled_str()
        if perturbation in enabled:
            return True
        elif perturbation in disabled:
            return False
        else:
            raise ValueError(f"Variation {perturbation} is not found in the perturbation set. Enabled: {enabled}, Disabled: {disabled}")

    def all_are_enabled(self) -> bool:
        _, disabled = self.which_enabled_str()
        return len(disabled) == 0

    @staticmethod
    def merge(perturbation_sets: list["PerturbationSet"]) -> "PerturbationSet":

        if len(perturbation_sets) == 0:
            return PerturbationSet()

        elif len(perturbation_sets) == 1:
            return perturbation_sets[0]
        
        elif len(perturbation_sets) == 2:
            ds_1, ds_2 = perturbation_sets
            ds_1_enabled, _ = ds_1.which_enabled_str()
            ds_2_enabled, _ = ds_2.which_enabled_str()
            for k in ds_1_enabled:
                assert k not in ds_2_enabled, f"Variation {k} is enabled in both ds_1 and ds_2"
            ds_merged = PerturbationSet()
            for k in ds_1_enabled:
                setattr(ds_merged, f"{k}_cfg", getattr(ds_1, f"{k}_cfg"))
            for k in ds_2_enabled:
                setattr(ds_merged, f"{k}_cfg", getattr(ds_2, f"{k}_cfg"))
            return ds_merged

        elif len(perturbation_sets) > 2:
            merged_12 = PerturbationSet.merge([perturbation_sets[0], perturbation_sets[1]])
            return PerturbationSet.merge([merged_12] + perturbation_sets[2:])

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

    def background_color_enabled(self) -> bool:
        return len(self.background_color_cfg) > 0

    def camera_pose_enabled(self) -> bool:
        return len(self.camera_pose_cfg) > 0

    # def MO_mass_enabled(self) -> bool:
    #     enabled = len(self.MO_mass_cfg) > 0
    #     assert not enabled, "MO mass is disabled"
    #     return False

    def language_paraphrase_enabled(self) -> bool:
        return len(self.language_paraphrase_cfg) > 0

    def language_other_task_enabled(self) -> bool:
        return len(self.language_other_task_cfg) > 0

    def language_random_enabled(self) -> bool:
        return len(self.language_random_cfg) > 0

    def language_none_enabled(self) -> bool:
        return len(self.language_none_cfg) > 0

    def enabled_language_modes(self) -> list[str]:
        return [
            name
            for name, enabled in [
                ("language_paraphrase", self.language_paraphrase_enabled()),
                ("language_other_task", self.language_other_task_enabled()),
                ("language_random", self.language_random_enabled()),
                ("language_none", self.language_none_enabled()),
            ]
            if enabled
        ]

    def has_language_perturbation(self) -> bool:
        return len(self.enabled_language_modes()) > 0

    def pose_randomization_enabled(self) -> bool:
        return len(self.pose_randomization_cfg) > 0

    def which_enabled_str(self) -> tuple[list[str], list[str]]:
        enabled_strs = []
        disabled_strs = []
        ignored_fns = ["perturbation_is_enabled", "all_are_enabled"]
        for k in [attr for attr in dir(self) if not attr.startswith('_')]:
            if k.endswith('_enabled') and hasattr(self, k) and (not k in ignored_fns):
                enabled_fn = getattr(self, k)
                if enabled_fn():
                    enabled_strs.append(k[:-8]) # Remove '_enabled' suffix and append
                else:
                    disabled_strs.append(k[:-8])
        return enabled_strs, disabled_strs

    def disable_perturbation_factors(self, perturbation_factors: list[str]):
        for k in perturbation_factors:
            if k.endswith("_cfg"):
                k = k[:-4]
            current_cfg = getattr(self, f"{k}_cfg")
            if len(current_cfg) == 0:
                continue
            setattr(self, f"{k}_cfg", {})
            cprint(f"WARNING: Disabling perturbation factor: '{k.upper()}'.", "yellow")

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
            "background_color_cfg",
            "camera_pose_cfg",
            # "MO_mass_cfg",
            "language_paraphrase_cfg",
            "language_other_task_cfg",
            "language_random_cfg",
            "language_none_cfg",
            "pose_randomization_cfg",
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

    def to_dict(self):
        d = {}
        for k in self.__dict__.keys():
            if k.endswith("_cfg"):
                d[k] = getattr(self, k)
        return d


# mani_skill/agents/base_agent.py
# ^ can set the scale of the robot here
_LANGUAGE_RANDOMIZATION_FILE = Path(__file__).parent / "language_randomizations.yaml"
_SYSTEM_WORDS_FILE = Path(__file__).parent / "random_words"
assert _LANGUAGE_RANDOMIZATION_FILE.exists(), f"Language randomization file does not exist: {_LANGUAGE_RANDOMIZATION_FILE}"
assert _SYSTEM_WORDS_FILE.exists(), f"System words file does not exist: {_SYSTEM_WORDS_FILE}"

all_distractor_set = PerturbationSet(
    distractor_object_cfg={
        "n_distractors": 2,
        "x_lims": (-0.4, 0.4),
        "y_lims": (-0.4, 0.4),
    },
    language_paraphrase_cfg={
        "randomization_file": _LANGUAGE_RANDOMIZATION_FILE,
    },
    language_other_task_cfg={
        "randomization_file": _LANGUAGE_RANDOMIZATION_FILE,
    },
    language_random_cfg={
        "num_words": 5,
        "words_file": _SYSTEM_WORDS_FILE,
    },
    language_none_cfg={
        "enabled": True,
    },
    MO_color_cfg ={"color_range": ColorRange(low=(0, 0, 0, 1), high=(1, 1, 1, 1)),},
    MO_texture_cfg = {"textures_directory": os.path.join(PACKAGE_ASSET_DIR, "textures"),},
    # MO_mass_cfg = {"mass_scale_range": (3, 8),},
    # MO_mass_cfg = {}, # ignored
    MO_size_cfg = {"scale_range": (0.8, 1.2)},
    # 
    RO_color_cfg ={"color_range": ColorRange(low=(0, 0, 0, 1), high=(1, 1, 1, 1)),},
    RO_texture_cfg = {"textures_directory": os.path.join(PACKAGE_ASSET_DIR, "textures"),},
    RO_size_cfg = {"scale_range": (0.8, 1.2)},
    # 
    table_color_cfg = {"color_range": ColorRange(low=(0, 0, 0, 1), high=(1, 1, 1, 1)),},
    table_texture_cfg = {"textures_directory": os.path.join(PACKAGE_ASSET_DIR, "textures"),},
    camera_pose_cfg = {
        "rpy_range": ((-0.035, -0.035, -0.035), (0.035, 0.035, 0.035)), # aproximately 2 degrees
        "xyz_range": ((-0.03, -0.03, 0.03), (0.03, 0.03, 0.03)),  # 3 cm
    },
    light_color_cfg = {
        "color_range": ColorRange(low=(0.5, 0.5, 0.5), high=(1, 1, 1)),
        "ambient_light_scale_range": (0.5, 1.5),
    },
    # ^ this works but makes it hard to see the color of the objects
    background_texture_cfg = {"textures_directory": os.path.join(PACKAGE_ASSET_DIR, "textures"),},
    background_color_cfg = {"color_range": ColorRange(low=(0, 0, 0, 1.0), high=(1, 1, 1, 1.0)),},
    pose_randomization_cfg = {"x_region_multiplier": 1.5, "y_region_multiplier": 1.5, "min_width_x": 0.1, "min_width_y": 0.1}
)

pose_randomization_2x_set = DistractionSet(
    pose_randomization_cfg={
        "x_region_multiplier": 2,
        "y_region_multiplier": 2,
        "min_width_x": 0.1,
        "min_width_y": 0.1,
    }
)

PERTURBATION_SETS = {
    "none".upper(): PerturbationSet(),
    "all".upper(): all_distractor_set,
    ### Distractor object
    "distractor_object".upper(): all_distractor_set.get_partial_copy(["distractor_object_cfg"]),
    ### MO
    "MO_color".upper(): all_distractor_set.get_partial_copy(["MO_color_cfg"]),
    "MO_texture".upper(): all_distractor_set.get_partial_copy(["MO_texture_cfg"]),
    "MO_size".upper(): all_distractor_set.get_partial_copy(["MO_size_cfg"]),
    # "MO_mass".upper(): all_distractor_set.get_partial_copy(["MO_mass_cfg"]),
    # ^ mass is ignored
    ### RO
    "RO_color".upper(): all_distractor_set.get_partial_copy(["RO_color_cfg"]),
    "RO_texture".upper(): all_distractor_set.get_partial_copy(["RO_texture_cfg"]),
    "RO_size".upper(): all_distractor_set.get_partial_copy(["RO_size_cfg"]),
    # ### Table
    "table_color".upper(): all_distractor_set.get_partial_copy(["table_color_cfg"]),
    "table_texture".upper(): all_distractor_set.get_partial_copy(["table_texture_cfg"]),
    ### Camera
    "camera_pose".upper(): all_distractor_set.get_partial_copy(["camera_pose_cfg"]),
    ### Light
    "light_color".upper(): all_distractor_set.get_partial_copy(["light_color_cfg"]),
    ### Background
    "background_texture".upper(): all_distractor_set.get_partial_copy(["background_texture_cfg"]),
    "background_color".upper(): all_distractor_set.get_partial_copy(["background_color_cfg"]),
    ### Language (ALL enables all four; update_language_instructions samples among enabled modes)
    "language_paraphrase".upper(): all_distractor_set.get_partial_copy(["language_paraphrase_cfg"]),
    "language_other_task".upper(): all_distractor_set.get_partial_copy(["language_other_task_cfg"]),
    "language_random".upper(): all_distractor_set.get_partial_copy(["language_random_cfg"]),
    "language_none".upper(): all_distractor_set.get_partial_copy(["language_none_cfg"]),
    ### Pose randomization
    "pose_randomization".upper(): all_distractor_set.get_partial_copy(["pose_randomization_cfg"]),
    "pose_randomization_2x".upper(): pose_randomization_2x_set,
}