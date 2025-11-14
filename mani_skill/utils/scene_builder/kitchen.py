"""Kitchen scene builder with counter and backdrop inspired by AI2THOR kitchens."""
import numpy as np
import sapien
import sapien.render
import torch

from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.scene_builder import SceneBuilder
from mani_skill.utils.structs.pose import Pose


class KitchenSceneBuilder(SceneBuilder):
    """Simple kitchen scene with counter, backsplash, and cabinet."""

    def __init__(
        self,
        env,
        robot_init_qpos_noise=0.02,
        counter_length=1.2,
        counter_width=0.6,
        counter_height=0.9,
        add_backsplash=True,
        add_upper_cabinet=True,
    ):
        super().__init__(env, robot_init_qpos_noise)
        self.counter_length = counter_length
        self.counter_width = counter_width
        self.counter_height = counter_height
        self.add_backsplash = add_backsplash
        self.add_upper_cabinet = add_upper_cabinet

    def build(self):
        # Build ground plane
        build_ground(self.scene)

        # Build kitchen counter (like a table but taller and with sides)
        self._build_counter()

        # Build backsplash wall
        if self.add_backsplash:
            self._build_backsplash()

        # Build upper cabinet
        if self.add_upper_cabinet:
            self._build_upper_cabinet()

    def _build_counter(self):
        """Build kitchen counter with realistic materials."""
        builder = self.scene.create_actor_builder()

        # Counter top (work surface)
        counter_thickness = 0.04
        builder.add_box_collision(
            half_size=[self.counter_length / 2, self.counter_width / 2, counter_thickness / 2],
            pose=sapien.Pose(p=[0, 0, self.counter_height - counter_thickness / 2]),
        )
        builder.add_box_visual(
            half_size=[self.counter_length / 2, self.counter_width / 2, counter_thickness / 2],
            pose=sapien.Pose(p=[0, 0, self.counter_height - counter_thickness / 2]),
            material=sapien.render.RenderMaterial(
                base_color=[0.9, 0.9, 0.85, 1.0],  # Light granite/marble color
                metallic=0.1,
                roughness=0.3,
                specular=0.5,
            ),
        )

        # Counter base (cabinet body) - darker wood
        base_height = self.counter_height - counter_thickness
        builder.add_box_collision(
            half_size=[self.counter_length / 2 - 0.01, self.counter_width / 2 - 0.01, base_height / 2],
            pose=sapien.Pose(p=[0, 0, base_height / 2]),
        )
        builder.add_box_visual(
            half_size=[self.counter_length / 2 - 0.01, self.counter_width / 2 - 0.01, base_height / 2],
            pose=sapien.Pose(p=[0, 0, base_height / 2]),
            material=sapien.render.RenderMaterial(
                base_color=[0.4, 0.3, 0.2, 1.0],  # Dark wood
                metallic=0.0,
                roughness=0.6,
            ),
        )

        builder.initial_pose = sapien.Pose()
        self.counter = builder.build_static(name="kitchen_counter")

    def _build_backsplash(self):
        """Build wall behind counter (backsplash)."""
        builder = self.scene.create_actor_builder()

        # Backsplash wall
        wall_height = 0.6
        wall_thickness = 0.02
        builder.add_box_collision(
            half_size=[self.counter_length / 2, wall_thickness / 2, wall_height / 2],
            pose=sapien.Pose(
                p=[0, -self.counter_width / 2 - wall_thickness / 2, self.counter_height + wall_height / 2]
            ),
        )
        builder.add_box_visual(
            half_size=[self.counter_length / 2, wall_thickness / 2, wall_height / 2],
            pose=sapien.Pose(
                p=[0, -self.counter_width / 2 - wall_thickness / 2, self.counter_height + wall_height / 2]
            ),
            material=sapien.render.RenderMaterial(
                base_color=[0.95, 0.95, 0.92, 1.0],  # Off-white tile
                metallic=0.0,
                roughness=0.4,
                specular=0.3,
            ),
        )

        builder.initial_pose = sapien.Pose()
        self.backsplash = builder.build_static(name="backsplash")

    def _build_upper_cabinet(self):
        """Build upper cabinet above counter."""
        builder = self.scene.create_actor_builder()

        # Upper cabinet
        cabinet_height = 0.4
        cabinet_depth = 0.3
        cabinet_y_offset = -self.counter_width / 2 + cabinet_depth / 2
        cabinet_z = self.counter_height + 0.6 + cabinet_height / 2  # 60cm above counter

        builder.add_box_collision(
            half_size=[self.counter_length / 2 - 0.1, cabinet_depth / 2, cabinet_height / 2],
            pose=sapien.Pose(p=[0, cabinet_y_offset, cabinet_z]),
        )
        builder.add_box_visual(
            half_size=[self.counter_length / 2 - 0.1, cabinet_depth / 2, cabinet_height / 2],
            pose=sapien.Pose(p=[0, cabinet_y_offset, cabinet_z]),
            material=sapien.render.RenderMaterial(
                base_color=[0.35, 0.25, 0.18, 1.0],  # Dark wood cabinet
                metallic=0.0,
                roughness=0.5,
            ),
        )

        builder.initial_pose = sapien.Pose()
        self.upper_cabinet = builder.build_static(name="upper_cabinet")

    def initialize(self, env_idx: torch.Tensor):
        """Initialize episode - nothing to randomize in static kitchen."""
        pass

    @property
    def counter_top_height(self):
        """Height of the counter work surface."""
        return self.counter_height
