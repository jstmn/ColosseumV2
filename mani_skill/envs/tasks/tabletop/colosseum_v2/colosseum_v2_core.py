from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional
import os

import numpy as np
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from typing import Any, Dict, Union
import numpy as np
import sapien
import torch
import trimesh
from mani_skill import PACKAGE_ASSET_DIR, PACKAGE_DIR
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from math import fabs
from mani_skill.utils.geometry import rotation_conversions
import os
import gymnasium as gym
from mani_skill.envs.tasks.tabletop.colosseum_v2.distraction_set import DistractionSet


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


REALSENSE_DEPTH_FOV_VERTICAL_RAD = 58.0 * np.pi / 180
REALSENSE_DEPTH_FOV_HORIZONTAL_RAD = 87.0 * np.pi / 180

DEFAULT_CAMERA_WIDTH = 256
DEFAULT_CAMERA_HEIGHT = 256

SHADER = "default"

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

def get_human_render_camera_config(eye: tuple[float, float, float], target: tuple[float, float, float], shader: str | None = None):
    """ Configures the human render camera. Shader options:
        - minimal: The fastest shader with minimal GPU memory usage. Note that the background will always be black (normally it is the color of the ambient light)
        - default: A balance between speed and texture availability
        - rt: A shader optimized for photo-realistic rendering via ray-tracing
        - rt-med: Same as rt but runs faster with slightly lower quality
        - rt-fast: Same as rt-med but runs faster with slightly lower quality
        -> https://maniskill.readthedocs.io/en/latest/user_guide/concepts/sensors.html#shaders-and-textures
    """
    SHADER = "default" if shader is None else shader
    pose = sapien_utils.look_at(eye=eye, target=target)
    return CameraConfig("render_camera", pose=pose, width=1264, height=1264, fov=np.pi / 3, near=0.01, far=100, shader_pack=SHADER)



def _get_random_color(color_range: tuple):
    assert (len(color_range) == 2) and (len(color_range[0]) == 3) and (len(color_range[1]) == 3), "color_range must be a tuple of two tuples of three floats"
    return np.random.uniform(*color_range).tolist() + [1]

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

        # We will use self._table_scenes if the table color or texture is enabled, otherwise the single table_scene
        self._table_scenes: list[TableSceneBuilder] = []
        self._table_scene: TableSceneBuilder | None = None
        self._table_actors: Actor | None = None

        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)


    def _load_lighting(self, options: dict):
        if self._ds.light_color_enabled():
            light_low, light_high = self._ds.light_color_cfg["ambient_light_range"]
            for sub_scene in self.scene.sub_scenes:
                sub_scene.ambient_light = [np.random.uniform(light_low, light_high), np.random.uniform(light_low, light_high), np.random.uniform(light_low, light_high)]
                theta_rand = np.random.uniform(0, 2*np.pi)
                sub_scene.add_directional_light(
                    direction=[-np.cos(theta_rand), -np.sin(theta_rand), -1],
                    position=[np.cos(theta_rand), np.sin(theta_rand), 1],
                    color=[1, 1, 1],
                    shadow=True,
                    shadow_scale=5,
                    shadow_map_size=4096,
                )
                # maniskill only allows for shadow from one directional light at a time for some reason

    def update_camera_configs(self, cfgs: list[CameraConfig]) -> list[CameraConfig]:
        if not self._ds.camera_pose_enabled():
            return cfgs

        rpy_range = self._ds.camera_pose_cfg["rpy_range"]
        xyz_range = self._ds.camera_pose_cfg["xyz_range"]

        for cfg in cfgs:
            rpy = np.random.uniform(*rpy_range)
            xyz = np.random.uniform(*xyz_range)
            delta_pose = sapien.Pose(p=xyz, q=euler2quat(rpy[0], rpy[1], rpy[2]))
            cfg.pose *= delta_pose

        return cfgs

    def set_color_or_texture(self, actor: Actor, color_cfg: dict, texture_cfg: dict, set_color: bool, set_texture: bool):

        if not set_color and not set_texture:
            return

        # The following code is borrowed from here: https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/domain_randomization.html
        for obj in actor._objs:
            # modify the i-th object which is in parallel environment i
            render_body_component: RenderBodyComponent = obj.find_component_by_type(RenderBodyComponent)
            for render_shape in render_body_component.render_shapes:
                for part in render_shape.parts:
                    # part.material: sapien.core.pysapien.RenderMaterial
                    if set_texture:
                        texture = _get_random_texture(texture_cfg["textures_directory"])
                    if set_color:
                        color = _get_random_color(color_cfg["color_range"])

                    if set_color and not set_texture:
                        part.material.set_base_color(color)
                    elif not set_color and set_texture:
                        part.material.set_base_color_texture(texture)
                    elif set_color and set_texture:
                        if np.random.random() < 0.5:
                            part.material.set_base_color(color)
                        else:
                            part.material.set_base_color_texture(texture)


    def load_scene_hook(self, manipulation_object: Optional[Actor], receiving_object: Actor | None = None):
        """
        This function is called when the scene is loaded.
        Args:
            scene (ManiSkillScene): The scene to modify.
            manipulation_object (Optional[Actor]): The manipulation object to modify. Note that this is a wrapper around
                                                    a sapien.Entity.
        """

        # New distractor spheres
        if self._ds.distractor_object_enabled():
            n_spheres = self._ds.distractor_object_cfg["n_spheres"]
            radius_range = self._ds.distractor_object_cfg["radius_range"]
            color_range = self._ds.distractor_object_cfg["color_range"]
            radii = np.random.uniform(*radius_range, size=n_spheres)

            self._ds._internal["distractor_object_cfg"]["internal__radii"] = radii
            self._ds._internal["distractor_object_cfg"]["internal__spheres"] = [
                actors.build_sphere(
                    self.scene,
                    initial_pose=sapien.Pose(),
                    name=f"distractor_sphere_{i}",
                    radius=radii[i],
                    color=np.random.uniform(*color_range).tolist() + [1.0], # alpha=1.0
                )
                for i in range(n_spheres)
            ]

        # Create the table and optionally set its color and/or texture
        if self._ds.table_color_enabled() or self._ds.table_texture_enabled():
            # Note: you can't add a texture to the table if you've set its color already.
            add_visual_from_file = not self._ds.table_color_enabled()
            for i in range(self.num_envs):
                table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
                table_scene.build(remove_table_from_state_dict_registry=True, scene_idx=i, name_suffix=f"env-{i}", add_visual_from_file=add_visual_from_file)
                self._table_scenes.append(table_scene)

            table_actors = Actor.merge([ts.table for ts in self._table_scenes], name="table_scene")
            self.add_to_state_dict_registry(table_actors)
            self.set_color_or_texture(table_actors, self._ds.table_color_cfg, self._ds.table_texture_cfg, self._ds.table_color_enabled(), self._ds.table_texture_enabled())
        else:
            self._table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
            self._table_scene.build()


        # Manipulation object
        if manipulation_object is not None:
            self.set_color_or_texture(manipulation_object, self._ds.MO_color_cfg, self._ds.MO_texture_cfg, self._ds.MO_color_enabled(), self._ds.MO_texture_enabled())

        # Receiving object
        if receiving_object is not None:
            self.set_color_or_texture(receiving_object, self._ds.RO_color_cfg, self._ds.RO_texture_cfg, self._ds.RO_color_enabled(), self._ds.RO_texture_enabled())


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

            x_lims = self._ds.distractor_object_cfg["x_lims"]
            y_lims = self._ds.distractor_object_cfg["y_lims"]
            radii = self._ds._internal["distractor_object_cfg"]["internal__radii"]
            x_range = x_lims[1] - x_lims[0]
            y_range = y_lims[1] - y_lims[0]

            # What happens if you set the poses such that the spheres collide with one another?
            for i, sphere in enumerate(self._ds._internal["distractor_object_cfg"]["internal__spheres"]):
                xyz = torch.rand((self.num_envs, 3), dtype=torch.float32)
                xyz[:, 0] = x_range * xyz[:, 0] + x_lims[0]
                xyz[:, 1] = y_range * xyz[:, 1] + y_lims[0]
                xyz[:, 2] = radii[i]
                sphere.set_pose(Pose.create_from_pq(p=xyz))
