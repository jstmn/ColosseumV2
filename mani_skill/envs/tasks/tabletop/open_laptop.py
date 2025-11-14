from typing import Dict

import numpy as np
import sapien
import sapien.physx as physx
import sapien.render
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.geometry.rotation_conversions import (
    quaternion_apply,
    quaternion_to_matrix,
)
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


@register_env("OpenLaptop-v1", max_episode_steps=120)
class OpenLaptopEnv(BaseEnv):
    """
    **Task Description:**
    Use the Panda arm to lift the laptop lid past the target angle.

    **Randomizations:**
    - Laptop base pose on the table (planar position and yaw).
    - Initial lid angle has a small random slack so the lid is not perfectly closed.

    **Success Conditions:**
    - The lid angle is opened beyond the success threshold.
    - The lid joint velocity is near zero (lid is stationary).
    """

    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    _base_size = np.array([0.30, 0.20, 0.02])  # Realistic laptop base: 30cm × 20cm × 2cm
    _lid_size = np.array([0.30, 0.20, 0.008])  # Realistic lid (screen): 30cm × 20cm × 0.8cm
    _lip_size = np.array([0.015, 0.20, 0.008])  # Realistic lip: 1.5cm deep, 20cm wide, 0.8cm tall
    _hinge_offset = 0.010
    _lid_clearance = 0.002

    _max_open_angle = np.deg2rad(130.0)
    _target_open_angle = np.deg2rad(50.0)  # More achievable target
    _success_open_angle = np.deg2rad(50.0)  # Success at 50 degrees open
    _initial_open_slack = np.deg2rad(40.0)  # Start at -40 degrees (working configuration)

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        # Initialize lip grasp point BEFORE calling super().__init__() to avoid AttributeError
        # when _get_obs_extra() is called during reset
        # Grasp point: center of the thin lip edge
        self._handle_local = torch.tensor(
            [
                -self._base_size[0] / 2 - self._lip_size[0] / 2,  # Center of lip (extends 1.5cm beyond base)
                0.0,
                0.0,  # Center height of thin lip
            ],
            dtype=torch.float32,
        )
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**23,
                max_rigid_patch_count=2**17,
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=[0.35, -0.25, 0.35],
            target=[0.18, 0.0, 0.1],
        )
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=[0.65, -0.35, 0.45],
            target=[0.15, 0.0, 0.15],
        )
        return CameraConfig(
            "render_camera",
            pose=pose,
            width=512,
            height=512,
            fov=1.0,
            near=0.01,
            far=100,
        )

    def _load_agent(self, options: Dict):
        # Position robot closer for realistic laptop dimensions (thin lip needs close access)
        super()._load_agent(options, sapien.Pose(p=[-0.55, 0, 0]))

    def _load_scene(self, options: Dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.laptop = self._build_laptop()
        self.lid_link = self.laptop.links_map["lid"]
        self.hinge_joint = self.laptop.active_joints_map["laptop_hinge"]

    def _build_laptop(self):
        builder = self.scene.create_articulation_builder()
        builder.disable_self_collisions = True

        base_half = self._base_size / 2
        lid_half = self._lid_size / 2
        hinge_x = -self._base_size[0] / 2 + self._hinge_offset
        base_z = base_half[2] + self._lid_clearance
        lid_z = -lid_half[2] - self._lid_clearance
        hinge_quat = euler2quat(0.0, 0.0, np.pi / 2)
        base_material = physx.PhysxMaterial(static_friction=1.2, dynamic_friction=0.9, restitution=0.0)
        lid_material = physx.PhysxMaterial(static_friction=0.8, dynamic_friction=0.6, restitution=0.0)

        base_builder = builder.create_link_builder()
        base_builder.set_name("base")
        base_builder.add_box_collision(
            pose=sapien.Pose([0, 0, 0]),
            half_size=base_half,
            material=base_material,
            density=800,
        )
        base_builder.add_box_visual(
            pose=sapien.Pose([0, 0, 0]),
            half_size=base_half,
            material=sapien.render.RenderMaterial(
                base_color=[0.15, 0.15, 0.15, 1.0],
                metallic=0.2,
                roughness=0.7,
            ),
        )

        lid_builder = builder.create_link_builder(base_builder)
        lid_builder.set_name("lid")
        lid_builder.add_box_collision(
            pose=sapien.Pose([0, 0, 0]),
            half_size=lid_half,
            material=lid_material,
            density=60,
        )
        lid_builder.add_box_visual(
            pose=sapien.Pose([0, 0, 0]),
            half_size=lid_half,
            material=sapien.render.RenderMaterial(
                base_color=[0.18, 0.18, 0.18, 1.0],
                metallic=0.1,
                roughness=0.6,
            ),
        )

        # Add graspable lip extending from front of lid
        lip_half = self._lip_size / 2
        lip_pose = sapien.Pose(
            [
                -self._base_size[0] / 2 - lip_half[0],  # Extends beyond base
                0.0,
                0.0,  # At lid level
            ]
        )
        lid_builder.add_box_collision(
            pose=lip_pose,
            half_size=lip_half,
            material=lid_material,
            density=40,
        )
        lid_builder.add_box_visual(
            pose=lip_pose,
            half_size=lip_half,
            material=sapien.render.RenderMaterial(
                base_color=[0.18, 0.18, 0.18, 1.0],  # Same color as lid
                metallic=0.1,
                roughness=0.6,
            ),
        )

        lid_builder.set_joint_name("laptop_hinge")
        lid_builder.set_joint_properties(
            type="revolute",
            limits=[[-self._max_open_angle, 0.1]],
            pose_in_parent=sapien.Pose([hinge_x, 0.0, base_z], hinge_quat),
            pose_in_child=sapien.Pose([hinge_x, 0.0, lid_z], hinge_quat),
            friction=0.002,
            damping=0.15,
        )

        laptop = builder.build(name="laptop", fix_root_link=True)
        return laptop

    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        device = self.device
        with torch.device(device):
            self.table_scene.initialize(env_idx)
            b = len(env_idx)

            pos = torch.zeros((b, 3), device=device)
            pos[:, 0] = 0.12 + randomization.uniform(
                -0.02, 0.02, size=(b,), device=device
            )
            pos[:, 1] = randomization.uniform(-0.03, 0.03, size=(b,), device=device)
            # Raise laptop on a stand/platform so thin lip is at reachable height
            pos[:, 2] = self._base_size[2] / 2 + 0.06  # 6cm platform under laptop

            # Rotate 180 degrees so the opening faces the robot (hinge away from robot)
            base_yaw = np.pi
            yaw = base_yaw + randomization.uniform(
                -np.deg2rad(10.0), np.deg2rad(10.0), size=(b,), device=device
            )
            quat = torch.zeros((b, 4), device=device)
            quat[:, 0] = torch.cos(yaw / 2)
            quat[:, 1] = 0.0
            quat[:, 2] = 0.0
            quat[:, 3] = torch.sin(yaw / 2)

            base_pose = Pose.create_from_pq(p=pos, q=quat)
            self.laptop.set_pose(base_pose)

            initial_angle = -self._initial_open_slack + randomization.uniform(
                -np.deg2rad(2.0),
                np.deg2rad(2.0),
                size=(b,),
                device=device,
            )
            self.hinge_joint.qpos = initial_angle
            self.hinge_joint.set_drive_target(initial_angle)
            self.hinge_joint.set_drive_velocity_target(torch.zeros_like(initial_angle))

    def evaluate(self):
        hinge_angle = self.hinge_joint.qpos
        hinge_vel = torch.abs(self.hinge_joint.qvel)
        target_angle = torch.tensor(
            -self._target_open_angle, device=hinge_angle.device, dtype=hinge_angle.dtype
        )
        success_angle = torch.tensor(
            -self._success_open_angle,
            device=hinge_angle.device,
            dtype=hinge_angle.dtype,
        )
        open_ratio = torch.clamp(-hinge_angle / target_angle, 0.0, 1.0)
        almost_still = hinge_vel <= 0.15
        target_reached = hinge_angle <= target_angle
        success = (hinge_angle <= success_angle) & almost_still
        return {
            "success": success,
            "lid_open_ratio": open_ratio,
            "lid_angle": hinge_angle,
            "lid_velocity": hinge_vel,
            "target_reached": target_reached,
            "lid_is_still": almost_still,
        }

    def _get_obs_extra(self, info: Dict):
        hinge_angle = self.hinge_joint.qpos
        joint_pose = self.hinge_joint.get_global_pose()
        hinge_axis = quaternion_to_matrix(joint_pose.q)[..., 0]

        lid_pose = self.lid_link.pose
        handle_local = self._handle_local.to(self.device)
        handle_pos = quaternion_apply(lid_pose.q, handle_local[None, :]) + lid_pose.p

        obs = {
            "laptop_base_pos": self.laptop.pose.p,
            "laptop_base_quat": self.laptop.pose.q,
            "lid_pos": lid_pose.p,
            "lid_quat": lid_pose.q,
            "lid_angle": hinge_angle,
            "lid_axis": hinge_axis,
            "lid_handle_pos": handle_pos,
        }

        if "state" in self.obs_mode:
            obs.update(
                tcp_to_handle=handle_pos - self.agent.tcp_pose.p,
                target_open_angle=torch.full_like(
                    hinge_angle, -self._target_open_angle
                ),
            )
        return obs

    def compute_dense_reward(self, obs: Dict, action=None, info: Dict = None):
        hinge_angle = self.hinge_joint.qpos
        hinge_vel = torch.abs(self.hinge_joint.qvel)
        target_angle = torch.tensor(
            -self._target_open_angle, device=hinge_angle.device, dtype=hinge_angle.dtype
        )

        # Progress reward: how much the lid has been opened toward the target
        open_ratio = torch.clamp(-hinge_angle / target_angle, 0.0, 1.0)
        progress_reward = open_ratio

        # Bonus for reaching the target
        target_reached = hinge_angle <= target_angle
        target_bonus = target_reached.float() * 2.0

        # Penalty for moving too fast
        velocity_penalty = torch.clamp(hinge_vel / 2.0, 0.0, 0.5)

        # Gripper-to-handle distance reward (encourage getting close to handle)
        if "tcp_to_handle" in obs:
            tcp_to_handle_dist = torch.linalg.norm(obs["tcp_to_handle"], dim=-1)
            reach_reward = 1.0 - torch.tanh(5.0 * tcp_to_handle_dist)
            # Only apply reach reward when lid is not yet opened significantly
            reach_weight = torch.where(open_ratio < 0.3, 0.5, 0.1)
            reach_reward = reach_reward * reach_weight
        else:
            reach_reward = 0.0

        # Total reward
        reward = progress_reward + target_bonus - velocity_penalty + reach_reward

        # Success bonus
        success = info.get("success", torch.zeros_like(hinge_angle, dtype=torch.bool))
        reward = reward + success.float() * 5.0

        return reward
