import numpy as np
import torch

from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.envs.tasks.tabletop.colosseum_v2.colosseum_v2_core import ColosseumV2Env, DisabledPerturbationFactors, PlacementRegion

PICK_CUBE_DOC_STRING = """**Task Description:**
A simple task where the objective is to grasp a red cube with the {robot_id} robot and move it to a target goal position. This is also the *baseline* task to test whether a robot with manipulation
capabilities can be simulated and trained properly. Hence there is extra code for some robots to set them up properly in this environment as well as the table scene builder.

**Randomizations:**
- the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
- the cube's z-axis rotation is randomized to a random angle
- the target goal position (marked by a green sphere) of the cube has its xy position randomized in the region [0.1, 0.1] x [-0.1, -0.1] and z randomized in [0, 0.3]

**Success Conditions:**
- the cube position is within `goal_thresh` (default 0.025m) euclidean distance of the goal position
- the robot is static (q velocity < 0.2)
"""


@register_env("RaiseCube-v1", max_episode_steps=50)
class RaiseCubeEnv(ColosseumV2Env):

    """ This is a copy of the PickCube-v1 environment, but rather than reaching a goal position after grasping, the 
    cube simply needs to be raised above a target height.
    """

    GOAL_HEIGHT = 0.2

    DISABLED_PERTURBATION_FACTORS = DisabledPerturbationFactors(
        RO_color=True,
        RO_texture=True,
        RO_size=True,
    )

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs):

        # 
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if robot_uids in PICK_CUBE_CONFIGS:
            cfg = PICK_CUBE_CONFIGS[robot_uids]
        else:
            cfg = PICK_CUBE_CONFIGS["panda"]
        self.cube_half_size = cfg["cube_half_size"]
        self.goal_thresh = cfg["goal_thresh"]
        self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
        super().__init__(*args, robot_uids=robot_uids, **kwargs)


    def _load_scene(self, options: dict):

        cube_color = np.array([12, 42, 160, 255]) / 255
        def get_cube_builder():
            return self.get_box_asset_builder(half_size=[self.cube_half_size] * 3, color=cube_color, object_type="MO")

        self.cube = self.add_asset_to_scene(get_cube_builder, name="cube", physics_type="dynamic", object_type="MO")
        self.load_scene_hook(manipulation_objects=[self.cube])

        self._cube_region = self.update_placement_region(
            PlacementRegion(x_lims=(-0.1, 0.1), y_lims=(-0.1, 0.1))
        )

    @property
    def _default_human_render_camera_configs(self):
        return self._get_human_render_camera_config(eye=(0.5, 0.6, 0.5), target=(0.0, 0.0, 0.1))

    @property
    def _default_sensor_configs(self):
        pose1 = sapien_utils.look_at(eye=[0.2,  -0.3, 0.3], target=self.sensor_cam_target_pos)
        pose2 = sapien_utils.look_at(eye=[0.1,  0.3,  0.3], target=self.sensor_cam_target_pos)
        return self.update_camera_configs(
            [
                CameraConfig("external1_camera", pose1, 224, 224, np.pi / 2, 0.01, 100),
                CameraConfig("external2_camera", pose2, 224, 224, np.pi / 2, 0.01, 100)
            ])


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)

            xyz = torch.zeros((b, 3))
            # xyz[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[..., :2] = self._cube_region.sample_xy(b, device=self.device)
            xyz[..., 2] = self.cube_half_size
            q = [1, 0, 0, 0]

            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.cube.set_pose(obj_pose)

            #
            self.initialize_episode_hook(env_idx, mo_pose=xyz)


    def evaluate(self):
        is_obj_raised = self.cube.pose.p[..., 2] > self.GOAL_HEIGHT
        is_grasped = self.agent.is_grasping(self.cube)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_obj_raised & is_robot_static,
            "is_obj_raised": is_obj_raised,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

