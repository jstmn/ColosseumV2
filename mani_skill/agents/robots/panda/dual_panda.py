from copy import deepcopy
import numpy as np
import sapien
import torch
from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor

import sapien.physx as physx



@register_agent()
class DualPanda(BaseAgent):
    uid = "dual_panda"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/dual_panda_table.urdf"
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            panda_1_leftfinger=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            panda_1_rightfinger=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            panda_2_leftfinger=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            panda_2_rightfinger=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    np.pi/2,
                    np.pi/2,
                    0,
                    0,
                    0,
                    0,
                    -np.pi * 7 / 8,
                    -np.pi * 7 / 8,
                    0,
                    0,
                    np.pi * 3 / 4,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                    0.04,
                    0.04,
                ]
            ),
            pose=sapien.Pose(),
        )
    )

    arm_1_joint_names = [
        "panda_1_joint1",
        "panda_1_joint2",
        "panda_1_joint3",
        "panda_1_joint4",
        "panda_1_joint5",
        "panda_1_joint6",
        "panda_1_joint7",
    ]
    gripper_1_joint_names = [
        "panda_1_finger_joint1",
        "panda_1_finger_joint2",
    ]
    ee_1_link_name = "panda_1_hand_tcp"

    arm_2_joint_names = [
        "panda_2_joint1",
        "panda_2_joint2",
        "panda_2_joint3",
        "panda_2_joint4",
        "panda_2_joint5",
        "panda_2_joint6",
        "panda_2_joint7",
    ]
    gripper_2_joint_names = [
        "panda_2_finger_joint1",
        "panda_2_finger_joint2",
    ]
    ee_2_link_name = "panda_2_hand_tcp"

    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 100

    def get_proprioception(self):
        """
        Get the proprioceptive state of the agent, default is the qpos and qvel of the robot and any controller state.
        """
        obs = super().get_proprioception()
        def get_flat_pose(pose):
            # Convert CUDA tensors to CPU numpy arrays before stacking
            p = pose.p
            q = pose.q
            
            # Handle CUDA tensors
            if isinstance(p, torch.Tensor):
                p = p.cpu().numpy() if p.is_cuda else p.numpy()
            if isinstance(q, torch.Tensor):
                q = q.cpu().numpy() if q.is_cuda else q.numpy()
            
            return np.hstack([p, q])
        
        world__T__ee_1 = self.robot.links_map[self.ee_1_link_name].pose
        obs["world__T__ee_1"] = get_flat_pose(world__T__ee_1)

        world__T__ee_2 = self.robot.links_map[self.ee_2_link_name].pose
        obs["world__T__ee_2"] = get_flat_pose(world__T__ee_2)

        obs["world__T__root"] = get_flat_pose(self.robot.root.pose)

        return obs

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm 1
        # -------------------------------------------------------------------------- #
        arm_1_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_1_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_1_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_1_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        arm_1_pd_joint_target_delta_pos = deepcopy(arm_1_pd_joint_delta_pos)
        arm_1_pd_joint_target_delta_pos.use_target = True

        arm_1_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_1_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_1_link_name,
            urdf_path=self.urdf_path,
        )

        arm_1_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_1_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=3.0 * self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_1_link_name,
            urdf_path=self.urdf_path,
        )

        arm_1_pd_ee_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_1_joint_names,
            pos_lower=None,
            pos_upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_1_link_name,
            urdf_path=self.urdf_path,
            use_delta=False,
            normalize_action=False,
        )
        arm_1_pd_ee_target_delta_pos = deepcopy(arm_1_pd_ee_delta_pos)
        arm_1_pd_ee_target_delta_pos.use_target = True
        arm_1_pd_ee_target_delta_pose = deepcopy(arm_1_pd_ee_delta_pose)
        arm_1_pd_ee_target_delta_pose.use_target = True
        # -------------------------------------------------------------------------- #
        # Arm 2
        # -------------------------------------------------------------------------- #
        arm_2_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_2_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_2_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_2_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        
        arm_2_pd_joint_target_delta_pos = deepcopy(arm_2_pd_joint_delta_pos)
        arm_2_pd_joint_target_delta_pos.use_target = True
        
        arm_2_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_2_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_2_link_name,
            urdf_path=self.urdf_path,
        )

        arm_2_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_2_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=3.0 * self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_2_link_name,
            urdf_path=self.urdf_path,
        )

        arm_2_pd_ee_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_2_joint_names,
            pos_lower=None,
            pos_upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_2_link_name,
            urdf_path=self.urdf_path,
            use_delta=False,
            normalize_action=False,
        )
        arm_2_pd_ee_target_delta_pos = deepcopy(arm_2_pd_ee_delta_pos)
        arm_2_pd_ee_target_delta_pos.use_target = True
        arm_2_pd_ee_target_delta_pose = deepcopy(arm_2_pd_ee_delta_pose)
        arm_2_pd_ee_target_delta_pose.use_target = True
        # -------------------------------------------------------------------------- #
        # Grippers
        # -------------------------------------------------------------------------- #
        gripper_1_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_1_joint_names,
            lower=-0.01,
            upper=0.04,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            mimic={"panda_1_finger_joint2": {"joint": "panda_1_finger_joint1"}},
        )

        gripper_2_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_2_joint_names,
            lower=-0.01,
            upper=0.04,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            mimic={"panda_2_finger_joint2": {"joint": "panda_2_finger_joint1"}},
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm_1=arm_1_pd_joint_delta_pos,
                gripper_1=gripper_1_pd_joint_pos,
                arm_2=arm_2_pd_joint_delta_pos,
                gripper_2=gripper_2_pd_joint_pos,
            ),
            pd_joint_pos=dict(
                arm_1=arm_1_pd_joint_pos,
                gripper_1=gripper_1_pd_joint_pos,
                arm_2=arm_2_pd_joint_pos,
                gripper_2=gripper_2_pd_joint_pos,
            ),
            pd_ee_delta_pos=dict(
                arm_1=arm_1_pd_ee_delta_pos,
                gripper_1=gripper_1_pd_joint_pos,
                arm_2=arm_2_pd_ee_delta_pos,
                gripper_2=gripper_2_pd_joint_pos,
            ),
            pd_joint_target_delta_pos=dict(
              arm_1=arm_1_pd_joint_target_delta_pos,
              gripper_1=gripper_1_pd_joint_pos,
              arm_2=arm_2_pd_joint_target_delta_pos,
              gripper_2=gripper_2_pd_joint_pos,  
            ),
            pd_ee_delta_pose=dict(
                arm_1=arm_1_pd_ee_delta_pose,
                gripper_1=gripper_1_pd_joint_pos,
                arm_2=arm_2_pd_ee_delta_pose,
                gripper_2=gripper_2_pd_joint_pos,
            ),
            pd_ee_pose=dict(
                arm_1=arm_1_pd_ee_pose,
                gripper_1=gripper_1_pd_joint_pos,
                arm_2=arm_2_pd_ee_pose,
                gripper_2=gripper_2_pd_joint_pos,
            ),
            pd_ee_target_delta_pos=dict(
                arm_1=arm_1_pd_ee_target_delta_pos,
                gripper_1=gripper_1_pd_joint_pos,
                arm_2=arm_2_pd_ee_target_delta_pos,
                gripper_2=gripper_2_pd_joint_pos,
            ),
            pd_ee_target_delta_pose=dict(
                arm_1=arm_1_pd_ee_target_delta_pose,
                gripper_1=gripper_1_pd_joint_pos,
                arm_2=arm_2_pd_ee_target_delta_pose,
                gripper_2=gripper_2_pd_joint_pos,
            ),
        )

        return deepcopy(controller_configs)

    def _after_init(self):
        self.finger_1_1_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "panda_1_leftfinger"
        )
        self.finger_1_2_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "panda_1_rightfinger"
        )
        self.finger_2_1_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "panda_2_leftfinger"
        )
        self.finger_2_2_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "panda_2_rightfinger"
        )
        self.tcp_1 = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_1_link_name
        )
        self.tcp_2 = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_2_link_name
        )



    def is_grasping(
        self, object: Actor, arm_index: int = 1, min_force=0.5, max_angle=85
    ):
        """Check if a specific arm is grasping an object

        Args:
            object (Actor): The object to check if the robot is grasping
            arm_index (int): 1 or 2 to specify which arm. Defaults to 1.
            min_force (float, optional): Minimum force in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact. Defaults to 85.
        """
        if arm_index == 1:
            finger_1 = self.finger_1_1_link
            finger_2 = self.finger_1_2_link
        else:
            finger_1 = self.finger_2_1_link
            finger_2 = self.finger_2_2_link

        l_contact_forces = self.scene.get_pairwise_contact_forces(finger_1, object)
        r_contact_forces = self.scene.get_pairwise_contact_forces(finger_2, object)
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        ldirection = finger_1.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -finger_2.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag).to(self.device)

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-2]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @property
    def tcp_1_pos(self):
        return self.tcp_1.pose.p

    @property
    def tcp_1_pose(self):
        return self.tcp_1.pose

    @property
    def tcp_2_pos(self):
        return self.tcp_2.pose.p

    @property
    def tcp_2_pose(self):
        return self.tcp_2.pose

    # Even for a dual panda arm, it does not matter which arm you are calculating the build_grasp_pose for!
    # It is automatically taken care by the approaching and closing you calculate.
    # There you will be using tcp_1 or tcp_2!
    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        """Build a grasp pose for end effector."""
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return sapien.Pose(T)
    
    def reset(self, init_qpos=None):
        """Reset the agent to initial configuration"""
        if init_qpos is None:
            # Use the 'rest' keyframe by default
            init_qpos = self.keyframes["rest"].qpos
        super().reset(init_qpos)