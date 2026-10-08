import numpy as np
import sapien
from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.controllers import PDJointPosMimicControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots import Panda
from mani_skill.envs.mpcm.constants import FORMER_TABLE_BASE_X
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs.pose import Pose

WRISTCAM_QPOS = np.array([0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, np.pi / 4, 0.04, 0.04])


@register_agent()
class PandaWristCam(Panda):
    """Panda with no default wrist camera. Wrist camera is added by the env.

    Gripper commands are unnormalized finger targets in [-0.02, 0.04] (open=0.04).
    """

    uid = "panda_wristcam2"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_v3.urdf"
    gripper_stiffness = 400.0
    gripper_damping = 40.0
    gripper_force_limit = 40.0

    @property
    def _controller_configs(self):
        configs = super()._controller_configs
        gripper = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            lower=-0.02,
            upper=0.04,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            normalize_action=False,
            mimic={"panda_finger_joint2": {"joint": "panda_finger_joint1"}},
        )
        for mode, mode_cfg in configs.items():
            assert "gripper" in mode_cfg, f"{mode} missing gripper config, keys={sorted(mode_cfg)}"
            mode_cfg["gripper"] = gripper
        return configs


def wrist_camera_config(agent, width: int, height: int) -> CameraConfig:
    assert isinstance(width, int) and width > 0, f"width={width!r}"
    assert isinstance(height, int) and height > 0, f"height={height!r}"
    return CameraConfig(
        uid="camera_wrist",
        pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
        width=width,
        height=height,
        # fov=np.pi / 2,
        fov=np.deg2rad(100),
        # fov=2*np.pi / 3, # 120 degrees
        near=0.01,
        far=100,
        mount=agent.robot.links_map["camera_link"],
    )


def reset_wristcam_robot(env, n_envs: int, flip_joint_6: bool) -> None:
    """TableSceneBuilder only poses ``panda`` and ``panda_wristcam``, not ``panda_wristcam2``.

    ``flip_joint_6`` negates qpos index 6 (panda_joint7). The nominal pose is +π/4;
    PickCube uses the negation, which is the −π/4 pose from before Sep 29.
    """
    assert isinstance(flip_joint_6, bool), f"flip_joint_6={flip_joint_6!r}, expected bool"
    nominal = WRISTCAM_QPOS.copy()
    if flip_joint_6:
        nominal[6] *= -1
    qpos = env._episode_rng.normal(0, env.robot_init_qpos_noise, (n_envs, len(nominal))) + nominal
    qpos[:, -2:] = 0.04
    env.agent.reset(qpos)
    env.agent.robot.set_pose(sapien.Pose([0.0, 0, 0]))


def shift_table_to_origin_base(table_scene) -> None:
    """Slide the stock table by ``-FORMER_TABLE_BASE_X`` so it stays under the shifted objects."""
    pose = table_scene.table.pose
    p = pose.p.clone()
    p[..., 0] -= FORMER_TABLE_BASE_X
    table_scene.table.set_pose(Pose.create_from_pq(p=p, q=pose.q))
