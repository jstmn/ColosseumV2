import os
from typing import Dict, Union

import numpy as np
import sapien
import torch
import gymnasium as gym

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.envs.tasks.tabletop.colosseum_v2.colosseum_v2_core import ColosseumV2Env


@register_env("PlaceBookInShelf-v1", max_episode_steps=50)
class PlaceBookEnv(ColosseumV2Env):
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
    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs
    ):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[-0.3, 0, 0.6], target=[-0.1, 0, -0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([-0.6, -0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0])) # Loads the panda arm


    def _load_scene(self, options: dict):

        def get_shelf_builder():
            return self.get_glb_asset_builder(
                glb_filepath=os.path.join(PACKAGE_ASSET_DIR, 'book_in_shelf/BookShelf.glb'),
                object_type="RO",
            )

        def get_book_builder():
            return self.get_glb_asset_builder(
                glb_filepath=os.path.join(PACKAGE_ASSET_DIR ,'book_in_shelf/simple_book_1.glb'),
                object_type="MO",
            )
        
        self.shelf = self.add_asset_to_scene(get_shelf_builder, name="shelf", type_="kinematic", object_type="RO")
        self.book_A = self.add_asset_to_scene(get_book_builder, name="book_A", type_="dynamic", object_type="MO")
        self.load_scene_hook(manipulation_objects=[self.book_A], receiving_objects=[self.shelf])


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            # self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.089
            region = [[0.03, -0.25],[0.09, 0]] 
            sampler = randomization.UniformPlacementSampler(bounds=region, batch_size=b, device=self.device)
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001
            bookA_xy = sampler.sample(radius, 100)

            xyz[:, :2] = bookA_xy
            self.book_A.set_pose(Pose.create_from_pq(p=xyz.clone(), q=torch.tensor([0.06, -0.162, -0.296, 0.940]).repeat(b,1)))

            xyz[..., 0] = 0.293 + torch.rand(b, device=self.device)*0.05
            xyz[..., 1] = -0.1 + torch.rand(b, device=self.device)*0.1            
            xyz[..., 2] = 0
            self.shelf.set_pose(Pose.create_from_pq(p=xyz, q=[-0.5, -0.5, 0.5, 0.5]))

            self.initialize_episode_hook(env_idx, mo_pose=xyz)
        self._initialize_agent()
        
    def _initialize_agent(self):
        qpos = np.array([-0.816, 0.109, 0.437, -3.005, 2.678, 1.626, -2.312, 0.04, 0.04])
        self.agent.reset(qpos)
        
    def evaluate(self):
        pos_shelf = self.shelf.pose.p
        pos_book = self.book_A.pose.p
        offset = pos_book
        offset[..., 0] = offset[..., 0] - pos_shelf[..., 0] + 0.293
        offset[..., 1] = offset[..., 1] - pos_shelf[..., 1] - 0.1
        x_flag = torch.logical_and((offset[..., 0]) <= 0.36, (offset[..., 0] >= 0.21))
        y_flag = torch.logical_and(-0.08 >= (offset[..., 1]), (offset[..., 1]) >= -0.17)
        z_flag = torch.logical_and(offset[..., 2] <= 0.16 + 0.005, offset[..., 2] >= 0.14)
        is_book_in_shelf = torch.logical_and(torch.logical_and(x_flag, y_flag),  z_flag)

        # NOTE (stao): GPU sim can be fast but unstable. Angular velocity is rather high despite it not really rotating
        is_book_static = self.book_A.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_book_grasped = self.agent.is_grasping(self.book_A)
        success = is_book_in_shelf * is_book_static * (~is_book_grasped)
        return {
            "is_book_grasped": is_book_grasped,
            "is_book_in_shelf": is_book_in_shelf,
            "is_book_static": is_book_static,
            "success": success
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                shelf_pose=self.shelf.pose.raw_pose,
                book_pose=self.book_A.pose.raw_pose,
                tcp_to_shelf_pos=self.shelf.pose.p - self.agent.tcp.pose.p,
                tcp_to_book_pos=self.book_A.pose.p - self.agent.tcp.pose.p,
                book_to_shelf_pos=self.shelf.pose.p - self.book_A.pose.p,
            )
        return obs



if __name__ == "__main__":
    # Now you can load this safe environment
    env = gym.make(
        "PlaceBookInShelf-v1", 
        # robot_uids="dual_panda", # Force the dual panda
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
        
        # if terminated or truncated:
        #     obs, _ = env.reset()
    
