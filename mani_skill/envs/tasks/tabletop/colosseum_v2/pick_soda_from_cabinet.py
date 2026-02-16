from typing import Any, Dict, Union
import numpy as np
import sapien
import torch
import trimesh
import os
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table.scene_builder import TableSceneBuilder
from mani_skill.utils.scene_builder.robocasa.fixtures.cabinet import OpenCabinet
from mani_skill.utils.structs.pose import Pose
from math import fabs
from mani_skill.utils.geometry import rotation_conversions
from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.envs.tasks.tabletop.colosseum_v2.colosseum_v2_core import ColosseumV2Env
import gymnasium as gym

@register_env("PickSodaFromCabinet-v1", max_episode_steps=50)
class PickSodaFromCabinetEnv(ColosseumV2Env):
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

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/StackCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs
    ):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)


    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[-0.3, 0, 0.6], target=[0, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([-0.6, -0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0])) # Loads the panda arm

    def _load_scene(self, options: dict):
        # Check if RoboCasa dataset is available
        self._check_robocasa_dataset()
                
        # If you previously built the full robocasa scene, skip it and use this:
        # programmatic open cabinet only:
        # size is width, depth, height (meters)
        cab_size = [0.6, 0.4, 0.9]  # adjust to taste
        open_cab = OpenCabinet(
            scene=self.scene,
            name="open_cabinet",
            size=cab_size,
            num_shelves=3,
            thickness=0.03,
            texture=None,
            pos=[-0.2, 0.5, 0.2],  # center pos in world coordinates (x,y,z)
            rng=np.random.default_rng(),  # or pass your env rng
        )
        # Build into the scene (for single env index, pass [0]; for batched envs use proper indices):
        open_cab.quat = sapien.Pose(q=[0.7071,0,0,-0.7071]).q  # default orientation
        open_cab.pos = np.array([0.3, -0.12, 0.456])
        # choose scene indices to build into; if you have a batch, build into all relevant indices
        scene_idxs = [i for i in range(self.num_envs)]
        built = open_cab.build(scene_idxs=scene_idxs)
        # If environment uses multiple envs, repeat build for each environment index you care about.
        # Optionally keep a handle:
        self.open_cabinet = built

        
        left_builder_fn = lambda: actors.build_box(
            self.scene,
            half_sizes=[0.38/2, 0.01, 0.272],
            color=np.array([141, 117, 105, 255]) / 255,
            name="left",
            body_type="kinematic",
            return_builder=True,
        )
        right_builder_fn = lambda: actors.build_box(
            self.scene,
            half_sizes=[0.38/2, 0.01, 0.272],
            color=np.array([141, 117, 105, 255]) / 255,
            name="right",
            body_type="kinematic",
            return_builder=True,
        )
        back_builder_fn = lambda: actors.build_box(
            self.scene,
            half_sizes=[0.58/2, 0.01, 0.272],
            color=np.array([141, 117, 105, 255]) / 255,
            name="back",
            body_type="kinematic",
            return_builder=True,
        )
        soda_builder_fn = lambda: self.get_glb_asset_builder(
            glb_filepath=os.path.join(PACKAGE_ASSET_DIR, "place_soda_in_cabinet/diet_soda.glb"),
            initial_pose=sapien.Pose(p=[0.055, -0.158, 0.1], q=[0.854,0.471,0.212,0.068]),
            object_type="MO",
            scale=(0.008,0.008,0.008),
        )
        self.left = self.add_asset_to_scene(left_builder_fn, name="left", physics_type="kinematic", object_type="RO")
        self.right = self.add_asset_to_scene(right_builder_fn, name="right", physics_type="kinematic", object_type="RO")
        self.back = self.add_asset_to_scene(back_builder_fn, name="back", physics_type="kinematic", object_type="RO")
        self.soda = self.add_asset_to_scene(soda_builder_fn, name="soda_can", physics_type="dynamic", object_type="MO")
        self.load_scene_hook(manipulation_objects=[self.soda], receiving_objects=[self.left, self.right, self.back])

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.405
            region = [[0.13, -0.3],[0.162, -0.1]]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001
            soda_xy = sampler.sample(radius, 100)

            xyz[:, :2] = soda_xy
            self.soda.set_pose(Pose.create_from_pq(p=xyz.clone(), q=torch.tensor([0.707, 0.707, 0, 0]).repeat(b,1)))
            self.left.set_pose(Pose.create_from_pq(p=torch.tensor([0.304005, 0.177265, 0.309642]), q=torch.tensor([1,0,0,0]).repeat(b,1)))
            self.right.set_pose(Pose.create_from_pq(p=torch.tensor([0.304005, -0.422210, 0.309642]), q=torch.tensor([1,0,0,0]).repeat(b,1)))
            self.back.set_pose(Pose.create_from_pq(p=torch.tensor([0.49, -0.120, 0.309642]), q=torch.tensor([0.7071,0,0,-0.7071]).repeat(b,1)))


            self.initialize_episode_hook(env_idx, mo_pose=xyz)
        self._initialize_agent()
            
    def _initialize_agent(self):
        # Reset the robot to a neutral position
        qpos = np.array([-0.387, -0.770, 0.395, -3.009, 2.978, 2.396, -1.963, 0.040, 0.04])
        self.agent.reset(qpos)
    
    def evaluate(self):
        is_soda_static = self.soda.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_soda_on_table = self.soda.pose.p[..., 2] < 0.1
        is_soda_in_x_lims = self.soda.pose.p[..., 0] > -0.3
        is_soda_in_y_lims = torch.logical_and(self.soda.pose.p[..., 1] > -0.3, self.soda.pose.p[..., 1] < -0)
        is_soda_within_bounds = torch.logical_and(is_soda_in_x_lims, is_soda_in_y_lims)
        success = torch.logical_and(is_soda_on_table, is_soda_within_bounds)
        return {
            "is_soda_on_table": is_soda_on_table,
            "is_soda_static": is_soda_static,
            "is_soda_within_bounds": is_soda_within_bounds,
            "success": success
        }
        
    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                soda_pose=self.soda.pose.raw_pose,
                tcp_to_soda_pos=self.soda.pose.p - self.agent.tcp.pose.p,
            )
        return obs


    def _check_robocasa_dataset(self):
        """
        Check if RoboCasa dataset is available.
        If not, provide helpful error message with download instructions.
        """
        import pathlib
        
        # Check for a key RoboCasa fixture file
        robocasa_data_path = pathlib.Path.home() / ".maniskill" / "data" / "scene_datasets" / "robocasa_dataset"
        cabinet_fixture_path = robocasa_data_path / "assets" / "fixtures" / "cabinets" / "cabinet_open.xml"
        
        if not cabinet_fixture_path.exists():
            error_msg = f"""
================================================================================
ERROR: RoboCasa dataset not found!
================================================================================

The PickSodaFromCabinet-v1 environment requires the RoboCasa dataset.

Expected location: {robocasa_data_path}
Missing file: {cabinet_fixture_path}

To download the dataset, run `python mani_skill/utils/download_asset.py RoboCasa`

After downloading, verify the path exists:
    ls -la {cabinet_fixture_path}

For more information, see:
    - https://github.com/haosulab/ManiSkill
    - https://github.com/robocasa/robocasa

================================================================================
"""
            raise FileNotFoundError(error_msg)

if __name__ == "__main__":
    # Now you can load this safe environment
    env = gym.make(
        "PickSodaFromCabinet-v1", 
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
        # Create a dummy action (stay still)
        # action = np.zeros(env.action_space.shape)
        
        # # Step the environment
        # obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the frame
        env.render()  # <--- Updates the GUI
        
    
