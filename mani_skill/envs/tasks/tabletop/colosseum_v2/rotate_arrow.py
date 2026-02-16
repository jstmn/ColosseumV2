from typing import Any, Dict

import numpy as np
import sapien
import torch
import torch.random
from mani_skill.utils.geometry.rotation_conversions import euler_angles_to_quaternion, quaternions_to_euler_angles
import os
from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import PandaStick
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.envs.tasks.tabletop.colosseum_v2.distraction_set import DistractionSet

@register_env("RotateArrow-v1", max_episode_steps=50)
class RotateArrowEnv(BaseEnv):
    """
    **Task Description:**
    The goal is to pick up a book and place it inside a shelf with other books already in it.

    **Randomizations:**
    - books on the table have their z-axis rotation randomized.
    - books have their xy positions on top of the table scene randomized. The positions are sampled such that the books do not collide with each other.

    **Success Conditions:**
    - the book is inside the shelf. (to within half of the book size)
    - the book is static
    - the book is not being grasped by the robot (robot must let go of the cube)

    """
    # 3D T center of mass spawnbox dimensions
    arrow_spawnbox_xlength = 0.2
    arrow_spawnbox_ylength = 0.2

    # translation of the spawnbox from goal tee as upper left of spawnbox
    arrow_spawnbox_xoffset = -0.4
    arrow_spawnbox_yoffset = -0.3
    #  end randomizations - rotation around z is simply uniform

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/StackCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    agent: PandaStick

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs
    ):
        distraction_set: DistractionSet | dict | None = kwargs.pop("distraction_set", None)
        self._distraction_set: DistractionSet | None = DistractionSet(**distraction_set) if isinstance(distraction_set, dict) else distraction_set
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        # sim_backend="physx_cuda:0", render_backend="sapien_cuda:0"
        if self.scene is not None:
            print(f"Is GPU simulation enabled for this scene? {self.scene.gpu_sim_enabled}")


    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.4], target=[-0.1, -0.1, 0])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([-0.6, -0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0])) # Loads the panda arm

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        # All values obtained carefully from blender
        self.arrow = self.add_glb_asset_to_scene(self.scene, 
                                            os.path.join(PACKAGE_ASSET_DIR,"push_arrow/arrow.glb"), 
                                            sapien.Pose(p=[0.293, -0.1, 0], q=[-0.5, -0.5, 0.5, 0.5]), 
                                            name="arrow",
                                            type="dynamic")


    @staticmethod
    def add_glb_asset_to_scene(scene, glb_filepath, pose, name, type="static"):
        """Load GLB file as a static actor in the scene"""
        builder = scene.create_actor_builder()
        # custom_material = sapien.render.RenderMaterial(base_color=[1, 0, 0, 1])
        custom_material = sapien.render.RenderMaterial()
        custom_material.base_color_texture = sapien.render.RenderTexture2D(filename = os.path.join(PACKAGE_ASSET_DIR, "textures/ceramic.png"))
            #     builder.add_visual_from_file(glb_filepath)
        builder.add_visual_from_file(glb_filepath, material=custom_material)
        builder.add_multiple_convex_collisions_from_file(glb_filepath, decomposition="coacd")
        
        builder.set_initial_pose(pose)
        if type=="dynamic":
            actor = builder.build_dynamic(name)
        else:
            actor = builder.build_static(name)
        return actor

    def quat_to_z_euler(self, quats):
        # sxyz convention, we want the z-axis rotation
        eulers = quaternions_to_euler_angles(quats, "XYZ")
        z_euler = eulers[..., 2]
        return z_euler

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):

        qpos0 = np.array(
            [0.0, 0, -0.04, -2.21, 0.0, 2.28, 0.66, 0.04, 0.04]
        )
        # qpos0 = np.zeros_like(qpos0)

        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx, qpos_0=qpos0)
            # setting the goal tee position, which is fixed, offset from center, and slightly rotated
            target_region_xyz = torch.zeros((b, 3))

#             # randomization code that randomizes the x, y position of the arrow we
#             # goal tee is alredy at y = -0.1 relative to robot, so we allow the tee to be only -0.2 y relative to robot arm
            target_region_xyz[..., 0] += (
                torch.rand(b) * (self.arrow_spawnbox_xlength) + self.arrow_spawnbox_xoffset
            )
            target_region_xyz[..., 1] += (
                torch.rand(b) * (self.arrow_spawnbox_ylength) + self.arrow_spawnbox_yoffset
            )

            target_region_xyz[..., 2] = (
                0.01982/2 + 2*1e-3
            )  # this is the half thickness of the tee plus a little

#             # rotation for pose is just random rotation around z axis
#             # z axis rotation euler to quaternion = [cos(theta/2),0,0,sin(theta/2)]

            q_euler_angle = torch.rand(b, device=self.device) * (torch.pi/6) - torch.pi/12
            self.init_angle = q_euler_angle
            euler_angles = torch.zeros((b, 3), device=self.device)
            # The original code used euler2quat(z, y, x, 'sxyz') from transforms3d.
            # So x_angle=q_euler_angle, y_angle=0, z_angle=np.pi/2
            euler_angles[:, 0] = q_euler_angle
            euler_angles[:, 2] = np.pi / 2
            q = euler_angles_to_quaternion(euler_angles, "ZYX")
            obj_pose = Pose.create_from_pq(p=target_region_xyz, q=q)
            self.arrow.set_pose(obj_pose)


    def evaluate(self):
        arrow_z_eulers = self.quat_to_z_euler(self.arrow.pose.q)
        rot_rew = (arrow_z_eulers - self.init_angle + torch.pi).cos()
        success = (rot_rew >= 0.9)
        return {"success": success}

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self.obs_mode_struct.use_state:
            # state based gets info on goal position and t full pose - necessary to learn task
            obs.update(
                # goal_pos=self.goal_arrow.pose.p,
                obj_pose=self.arrow.pose.raw_pose,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        arrow_z_eulers = self.quat_to_z_euler(self.arrow.pose.q)
        rot_rew = (arrow_z_eulers - self.init_angle+torch.pi).cos()
        reward = (rot_rew + 1) / 2
        reward[info["success"]] = 3
        return reward


    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 1
