import gymnasium as gym
import numpy as np
import sapien.core as sapien
import torch
import os

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.utils.registration import register_env
from mani_skill.agents.robots.panda.dual_panda import DualPanda 
from mani_skill.utils.structs import Pose
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.envs.tasks.tabletop.colosseum_v2.colosseum_v2_core import ColosseumV2Env, DisabledVariationFactors, PlacementRegion


@register_env("DualArmPourPot-v1", max_episode_steps=1000)
class DualArmPourPotEnv(ColosseumV2Env):
    """
    A minimal environment for Dual Panda motion planning.
    No cubes, no tasks, just the robot.
    """
    cube_half_size = 0.02
    tray_half_width = 0.2
    tray_half_length = 0.12
    # Explicitly tell ManiSkill to use the DualPanda agent
    SUPPORTED_ROBOTS = ["dual_panda"]
    agent: DualPanda # Type hinting for IDE support

    DisabledVariationFactors = DisabledVariationFactors(
        RO_color=True,
        RO_texture=True,
        RO_size=True,
    )

    def __init__(self, *args, robot_uids="dual_panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[-1.0, 0.0, 0.75 + 0.83], target=[0.0, 0, 0.2 + 0.83]) # 0.83: height of the table
        return self.update_camera_configs([
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 3,
                near=0.01,
                far=10,
            )
        ])

    @property
    def _default_human_render_camera_configs(self):
        """Configure camera for rendering videos and visualization"""
        pose = sapien_utils.look_at(eye=[0.6, 0.2, 0.4+0.83], target=[-0.1, 0, 0.1+0.83])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    
    def _load_scene(self, options: dict):
        ball_builder = lambda: self.get_glb_asset_builder(
                                           os.path.join(PACKAGE_ASSET_DIR, "pour_pot/tomato.glb"),
                                           initial_pose=sapien.Pose(p=[-0.2, -0.141, 0.83+self.cube_half_size]),
                                           object_type="MO",
                                           scale=(1,1,1),
                                           )
        pot_builder = lambda: self.get_glb_asset_builder( 
                                        os.path.join(PACKAGE_ASSET_DIR,"pour_pot/pot.glb"),
                                        initial_pose=sapien.Pose(p=[0.055, -0.158, 0.], q=[0.854,0.471,0.212,0.068]),
                                        object_type="MO",
                                        scale=(1,1,1),
                                        color=np.array([129/255, 133/255, 137/255, 1]))
        tray_builder = lambda: self.get_glb_asset_builder(
                                           os.path.join(PACKAGE_ASSET_DIR, "pour_pot/plastic_tray.glb"),
                                           initial_pose=sapien.Pose(),
                                           object_type="MO",
                                           scale=(0.4,0.4,0.4),
                                           color=np.array([48/255, 49/255, 51/255, 1]))
        self.ball = self.add_asset_to_scene(ball_builder, name="ball", physics_type="dynamic", object_type="BACKGROUND")
        self.pot = self.add_asset_to_scene(pot_builder, name="pot", physics_type="dynamic", object_type="MO")
        self.tray = self.add_asset_to_scene(tray_builder, name="tray", physics_type="dynamic", object_type="RO")
        self.load_scene_hook(manipulation_objects=[self.pot], receiving_objects=[self.tray])

        self._pot_region = self.update_placement_region(
            # Ground-truth from legacy sampling:
            # pot_xyz[..., 0] = 0.1
            # pot_xyz[..., 1] = torch.rand(b, device=self.device) * 0.2 - 0.1
            PlacementRegion(x_lims=(0.1, 0.1), y_lims=(-0.1, 0.1))
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            pot_xyz = torch.zeros((b, 3), device=self.device)
            # pot_xyz[..., 0] = 0.1
            # pot_xyz[..., 1] = torch.rand(b, device=self.device) * 0.2 - 0.1
            pot_xyz[..., 0:2] = self._pot_region.sample_xy(b, device=self.device)
            pot_xyz[..., 2] = self.cube_half_size+0.83

            pot_q = torch.tensor([0.5, 0.5, 0.5, 0.5], device=self.device).repeat(b, 1)
            self.pot.set_pose(Pose.create_from_pq(p=pot_xyz, q=pot_q))

            ball_xyz = pot_xyz.clone()
            ball_xyz[..., 2] = 0.9
            self.ball.set_pose(Pose.create_from_pq(p=ball_xyz,q=[1,0,0,0]))

            tray_xyz = pot_xyz.clone()
            tray_xyz[..., 0] = -0.2
            tray_xyz[..., 2] = 0.83 + self.cube_half_size
            self.tray.set_pose(Pose.create_from_pq(p=tray_xyz, q=[0.5, 0.5, 0.5, 0.5]))
            self.initialize_episode_hook(env_idx, mo_pose=self.pot.pose)
        self._initialize_agent()
        
    def _initialize_agent(self):
        # Reset the robot to a neutral position
        # Dual Panda has 14+ gripper joints. 
        # You can define a custom "qpos" (joint positions) here if you want.
        # 0-6: Left Arm, 7-8: Left Gripper, 9-15: Right Arm, 16-17: Right Gripper
        qpos = np.array([0.599, 2.358, 0.4, 0.442, -0.561, 0.697, -2.511, -2.513, 1.695, -1.775, 1.395, 1.347, -0.479, 2.031, 0.04, 0.04, 0.04, 0.04])
        # Example: Set arms to a ready position (optional)
        # qpos[0] = 0.5  # Move left shoulder
        # qpos[9] = -0.5 # Move right shoulder
        
        self.agent.reset(qpos)


    def evaluate(self):
        ball_pos = self.ball.pose.p
        tray_pos = self.tray.pose.p
        offset = tray_pos - ball_pos
        in_tray = (torch.abs(offset[..., 1]) < DualArmPourPotEnv.tray_half_width) * (torch.abs(offset[..., 0]) < DualArmPourPotEnv.tray_half_length)
        success = (ball_pos[..., 2] >= 0.83) * (ball_pos[..., 2] < 0.9) * in_tray
        return {"inside_tray": in_tray,"success": success}
    


# 2. Main Execution Block
if __name__ == "__main__":
    # Now you can load this safe environment
    env = gym.make(
        "DualArmPourPot-v1", 
        robot_uids="dual_panda", # Force the dual panda
        obs_mode="state_dict", 
        control_mode="pd_joint_delta_pos",
        render_mode="human"
    )

    print("Environment Created Successfully!")
    obs, _ = env.reset()
    
    print(f"Observation Keys: {obs.keys()}")
    if "agent" in obs:
        print(f"Joint Positions Shape: {obs['agent']['qpos'].shape}")
    
    # NOW you can run your IK loop here
    # 2. You MUST run a loop, or the window will close immediately
    while True:
        env.render()  # <--- Updates the GUI
    
    env.close()