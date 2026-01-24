import mplib
import numpy as np
import sapien
from typing import Tuple, Optional
from .bimanual_planner import BimanualPlanner
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.examples.motionplanning.base_motionplanner.motionplanner import BaseMotionPlanningSolver
from mani_skill.examples.motionplanning.two_finger_gripper.motionplanner import (
    TwoFingerGripperMotionPlanningSolver,
)
from scipy.spatial.transform import Rotation

from mani_skill.envs.scene import ManiSkillScene

from mani_skill.utils.structs.pose import to_sapien_pose
from transforms3d import quaternions

def pose_to_matrix(pose):
    """Convert Sapien Pose to 4x4 matrix"""
    T = np.eye(4)
    # Handle position
    p = pose.p
    if hasattr(p, 'cpu'):
        p = p.cpu().numpy().flatten()
    T[:3, 3] = p
    
    # Handle rotation  
    q = pose.q
    if hasattr(q, 'cpu'):
        q = q.cpu().numpy().flatten()
    # SAPIEN uses [qx, qy, qz, qw], scipy uses [qx, qy, qz, qw]
    R = Rotation.from_quat(q).as_matrix()
    T[:3, :3] = R
    return T

def matrix_to_pose_7d(T):
    """Convert 4x4 matrix to [x,y,z,qw,qx,qy,qz] for IK"""
    pos = T[:3, 3]
    R = T[:3, :3]
    quat_xyzw = Rotation.from_matrix(R).as_quat()  # [qx, qy, qz, qw]
    # Reorder to [qw, qx, qy, qz] for your IK
    return np.array([pos[0], pos[1], pos[2], quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

def build_two_finger_gripper_grasp_pose_visual(scene: ManiSkillScene, name="grasp_pose_visual"):
    builder = scene.create_actor_builder()
    grasp_pose_visual_width = 0.01
    grasp_width = 0.05

    builder.add_sphere_visual(
        pose=sapien.Pose(p=[0, 0, 0.0]),
        radius=grasp_pose_visual_width,
        material=sapien.render.RenderMaterial(base_color=[0.3, 0.4, 0.8, 0.7])
    )

    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, -0.08]),
        half_size=[grasp_pose_visual_width, grasp_pose_visual_width, 0.02],
        material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, -0.05]),
        half_size=[grasp_pose_visual_width, grasp_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[
                0.03 - grasp_pose_visual_width * 3,
                grasp_width + grasp_pose_visual_width,
                0.03 - 0.05,
            ],
            q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
        ),
        half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[
                0.03 - grasp_pose_visual_width * 3,
                -grasp_width - grasp_pose_visual_width,
                0.03 - 0.05,
            ],
            q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
        ),
        half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 0.7]),
    )
    # CRITICAL FIX: Use the passed 'name' argument here
    grasp_pose_visual = builder.build_kinematic(name=name)
    return grasp_pose_visual


class DualPandaMotionPlanningSolver(BaseMotionPlanningSolver):
    """
    Motion planning solver for dual Panda arms.
    
    This solver uses a custom bimanual planner that can handle:
    - Independent arm motion
    - Synchronized dual-arm grasping
    - Constrained motion (when both arms hold an object)
    """
    
    # Gripper states
    OPEN = 1
    CLOSED = -1
    
    # Move groups (end-effector links)
    MOVE_GROUP = ["panda_1_hand_tcp", "panda_2_hand_tcp"]
    
    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None,
        visualize_target_grasp_pose: bool = True,
        print_env_info: bool = False,
        joint_vel_limits: float = 0.9,
        joint_acc_limits: float = 0.9,
    ):
        if base_pose is None:
            base_pose = sapien.Pose()
        
        # Store dual-arm specific parameters before parent init
        self.visualize_target_grasp_pose = visualize_target_grasp_pose
        
        # Initialize gripper states for both arms
        self.gripper_1_state = self.OPEN
        self.gripper_2_state = self.OPEN
        self.debug = False
        # Visualization objects (will be set up after parent init)
        self.grasp_pose_visual_1 = None
        self.grasp_pose_visual_2 = None
        self.print_env_info = False
            
        # Call parent init (this calls setup_planner internally)
        super().__init__(
            env=env,
            debug=debug,
            vis=vis,
            base_pose=base_pose,
            print_env_info=print_env_info,
            joint_vel_limits=joint_vel_limits,
            joint_acc_limits=joint_acc_limits
        )
        # self.env_agent.reset()
        self.planner = self.setup_planner()
        # Now setup grasp pose visualizations after scene is ready
        if self.vis and self.visualize_target_grasp_pose:
            self._setup_grasp_visuals()
        # Verify joint ordering matches expectations
        if self.print_env_info:
            self._verify_joint_ordering()

    def _setup_grasp_visuals(self):
        """Create visual indicators for target grasp poses"""
        # Setup Visual 1 (Right Hand / Panda 1)
        if "grasp_pose_visual_1" not in self.base_env.scene.actors:
            self.grasp_pose_visual_1 = build_two_finger_gripper_grasp_pose_visual(
                self.base_env.scene, 
                name="grasp_pose_visual_1" # Explicit unique name
            )
        else:
            self.grasp_pose_visual_1 = self.base_env.scene.actors["grasp_pose_visual_1"]
        
        # Setup Visual 2 (Left Hand / Panda 2)
        if "grasp_pose_visual_2" not in self.base_env.scene.actors:
            self.grasp_pose_visual_2 = build_two_finger_gripper_grasp_pose_visual(
                self.base_env.scene, 
                name="grasp_pose_visual_2" # Explicit unique name
            )
        else:
            self.grasp_pose_visual_2 = self.base_env.scene.actors["grasp_pose_visual_2"]
            
        # Set initial poses
        self.grasp_pose_visual_1.set_pose(self.env_agent.tcp_1_pose)
        self.grasp_pose_visual_2.set_pose(self.env_agent.tcp_2_pose)
    
    def _verify_joint_ordering(self):
        """Verify that joint ordering is as expected"""
        
        sapien_joint_names = [j.get_name() for j in self.robot.get_active_joints()]
        planner_joint_names = [self.planner.user_joint_names[idx] 
                              for idx in self.planner.move_group_joint_indices]
        if self.debug:
            print("SAPIEN Joint Order:")
            for i, name in enumerate(sapien_joint_names):
                print(f"  {i}: {name}")
            
            print("\nPlanner Joint Order:")
            for i, name in enumerate(planner_joint_names):
                print(f"  {i}: {name}")
                
            print("\nPlanner move_group_joint_indices:")
            print(f"  {self.planner.move_group_joint_indices}")
            
            print("\nPlanner joint_name_2_idx:")
            for name in ["panda_1_joint1", "panda_2_joint1", "panda_1_joint2", "panda_2_joint2"]:
                if name in self.planner.joint_name_2_idx:
                    print(f"  {name} -> {self.planner.joint_name_2_idx[name]}")

        # Check if they match
        if sapien_joint_names == planner_joint_names:
            if self.debug:
                print("✓ Joint orderings match!")
        else:
            self._create_joint_mapping(sapien_joint_names, planner_joint_names)
    
    def _create_joint_mapping(self, sapien_names, planner_names):
        """Create mapping between SAPIEN and planner joint orders if needed"""
        
        # Map: planner_idx -> sapien_idx
        self.planner_to_sapien_map = []
        for p_name in planner_names:
            if p_name in sapien_names:
                self.planner_to_sapien_map.append(sapien_names.index(p_name))
            else:
                if self.debug:
                    print(f"WARNING: Planner joint {p_name} not found in SAPIEN!")
        
        # Map: sapien_idx -> planner_idx
        self.sapien_to_planner_map = [None] * len(sapien_names)
        for planner_idx, sapien_idx in enumerate(self.planner_to_sapien_map):
            self.sapien_to_planner_map[sapien_idx] = planner_idx
        
        self.needs_mapping = True
    
    def _convert_qpos_sapien_to_planner(self, qpos_sapien: np.ndarray) -> np.ndarray:
        """Convert qpos from SAPIEN ordering to planner ordering"""
        if not hasattr(self, 'needs_mapping') or not self.needs_mapping:
            return qpos_sapien[:len(self.planner.move_group_joint_indices)]
        
        qpos_planner = np.zeros(len(self.planner.move_group_joint_indices))
        for planner_idx, sapien_idx in enumerate(self.planner_to_sapien_map):
            if sapien_idx < len(qpos_sapien):
                qpos_planner[planner_idx] = qpos_sapien[sapien_idx]
        return qpos_planner
    
    def _convert_qpos_planner_to_sapien(self, qpos_planner: np.ndarray) -> np.ndarray:
        """Convert qpos from planner ordering to SAPIEN ordering"""
        if not hasattr(self, 'needs_mapping') or not self.needs_mapping:
            return qpos_planner
        
        # Get current full qpos from robot
        qpos_sapien = self.robot.get_qpos().cpu().numpy()[0].copy()
        
        # Update only the move group joints
        for planner_idx, sapien_idx in enumerate(self.planner_to_sapien_map):
            if sapien_idx < len(qpos_sapien) and planner_idx < len(qpos_planner):
                qpos_sapien[sapien_idx] = qpos_planner[planner_idx]
        
        return qpos_sapien
    
    def setup_planner(self):
        """Setup the bimanual planner using BimanualPlanner_v3"""
        
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        
        # Determine SRDF path
        srdf_path = self.env_agent.urdf_path.replace(".urdf", ".srdf")
        
        # Create bimanual planner with dual arm move group
        planner = BimanualPlanner(
            urdf=self.env_agent.urdf_path,
            srdf=srdf_path,
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="dual_arm",
            joint_vel_limits=np.ones(18) * self.joint_vel_limits,  # 18 DOF (9 per arm)
            joint_acc_limits=np.ones(18) * self.joint_acc_limits,
        )
        
        # Set base pose (usually identity for fixed-base robots)
        planner.set_base_pose(np.hstack([self.base_pose.p, self.base_pose.q]))
        
        # Override joint velocity and acceleration limits from planner
        planner.joint_vel_limits = np.asarray(planner.joint_vel_limits) * self.joint_vel_limits
        planner.joint_acc_limits = np.asarray(planner.joint_acc_limits) * self.joint_acc_limits
        
        # Print debug info about planner setup
        if self.debug:
            print(f"\n=== Dual Panda Planner Setup ===")
            print(f"URDF: {self.env_agent.urdf_path}")
            print(f"SRDF: {srdf_path}")
            print(f"Move group: dual_arm")
            print(f"Planning joints: {len(planner.move_group_joint_indices)}")
            print(f"Joint names: {[planner.user_joint_names[i] for i in planner.move_group_joint_indices]}")
        
        return planner
    
    def _update_grasp_visual(self, target_1: sapien.Pose, target_2: sapien.Pose = None) -> None:
        """Update visual indicators for target grasp poses"""
        if self.grasp_pose_visual_1 is not None and target_1 is not None:
            self.grasp_pose_visual_1.set_pose(target_1)
        if self.grasp_pose_visual_2 is not None and target_2 is not None:
            self.grasp_pose_visual_2.set_pose(target_2)
    
    def follow_path(self, result, refine_steps: int = 0, arm_index: int = None):
        import time
        """
        Execute a planned path.
        
        Args:
            result: Planning result containing position and velocity trajectories
            refine_steps: Additional steps to hold final position
            arm_index: If specified (1 or 2), only that arm moves. If None, both move.
        """
        n_step = result["position"].shape[0]
        current_qpos_sapien = self.robot.get_qpos().cpu().numpy()[0]
        current_qpos = self._convert_qpos_sapien_to_planner(current_qpos_sapien)
        if arm_index == 1:
            final_pos = current_qpos[9:16]
        elif arm_index == 2:
            final_pos = current_qpos[0:7]
        for i in range(n_step + refine_steps):
            qpos_18d = result["position"][min(i, n_step - 1)]
            if arm_index == 1:
                arm_1_pose = qpos_18d[0:7]
                arm_2_pose = final_pos
            elif arm_index == 2:
                arm_1_pose = final_pos
                arm_2_pose = qpos_18d[9:16]
            else:
                arm_1_pose = qpos_18d[0:7]
                arm_2_pose = qpos_18d[9:16]
                
            g1_val = 1 if self.gripper_1_state == self.OPEN else -1
            g2_val = 1 if self.gripper_2_state == self.OPEN else -1
                        
            action = np.hstack([arm_1_pose, g1_val, arm_2_pose, g2_val])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            
            if self.vis:
                self.base_env.render_human()
                
        return obs, reward, terminated, truncated, info
    
    def move_to_pose_pair_with_RRTConnect(
        self,
        left_pose: sapien.Pose,
        right_pose: sapien.Pose,
        dry_run: bool = False,
        refine_steps: int = 0
    ):
        """
        Move both arms to target poses using RRT Connect planning.
        
        Args:
            left_pose: Target pose for left arm (panda_2)
            right_pose: Target pose for right arm (panda_1)
            dry_run: If True, only plan without execution
            refine_steps: Additional steps to hold final position
        """
        left_pose = to_sapien_pose(left_pose)
        right_pose = to_sapien_pose(right_pose)
        
        self._update_grasp_visual(right_pose, left_pose)

        # Get current qpos and convert to planner space
        current_qpos_sapien = self.robot.get_qpos().cpu().numpy()[0]
        current_qpos = self._convert_qpos_sapien_to_planner(current_qpos_sapien)
        
        # Convert poses to [x, y, z, qw, qx, qy, qz] format
        left_pose_7d = np.concatenate([left_pose.p, left_pose.q])
        right_pose_7d = np.concatenate([right_pose.p, right_pose.q])
        
        # Solve IK for both arms
        ik_result = self.planner.IK(
            left_target_pose=left_pose_7d,
            right_target_pose=right_pose_7d,
            start_qpos=current_qpos,
            left_link_name="panda_2_hand_tcp",
            right_link_name="panda_1_hand_tcp",
        )
        
        if isinstance(ik_result, str):
            if self.debug:
                print(f"IK Failed: {ik_result}")
            self.render_wait()
            return -1
        
        # Check collisions
        collisions_self = self.planner.check_for_self_collision(qpos=ik_result)
        collisions_env = self.planner.check_for_env_collision(qpos=ik_result)
        
        if len(collisions_self) > 0 or len(collisions_env) > 0:
            if self.debug:
                print("IK solution is in collision")
                for c in collisions_self:
                    print(f"   Self-collision: {c.link_name1} <-> {c.link_name2}")
                for c in collisions_env:
                    print(f"   Environment collision: {c.link_name1} <-> {c.link_name2}")
            self.render_wait()
            return -1
        
        # Plan path
        result = self.planner.plan_qpos_to_qpos(
            goal_qposes=[ik_result],
            current_qpos=current_qpos,
            time_step=self.base_env.control_timestep,
            planning_time=10.0,
            planner_name="RRTConnect",
        )
        
        if result["status"] != "Success":
            if self.debug:
                print(f"Planning failed: {result['status']}")
            self.render_wait()
            return -1
        
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)
    
    def move_arm_to_pose_with_RRTConnect(
        self,
        pose: sapien.Pose,
        arm_index: int,
        dry_run: bool = False,
        refine_steps: int = 0
    ):
        """
        Move a single arm to target pose while keeping the other arm fixed.
        
        Args:
            pose: Target pose for the specified arm
            arm_index: 1 for right arm (panda_1), 2 for left arm (panda_2)
            dry_run: If True, only plan without execution
            refine_steps: Additional steps to hold final position
        """
        pose = to_sapien_pose(pose)
        
        # Get current qpos
        current_qpos = self.robot.get_qpos().cpu().numpy()[0]
        
        # Helper function to convert SAPIEN pose to IK format
        def sapien_pose_to_ik_format(sapien_pose):
            """Convert SAPIEN pose to [x,y,z,qw,qx,qy,qz] format"""
            # Handle position
            p = sapien_pose.p
            if hasattr(p, 'cpu'):
                p = p.cpu().numpy().flatten()
            else:
                p = np.array(p).flatten()
            
            # Handle quaternion [qx,qy,qz,qw] -> [qw,qx,qy,qz]
            q = sapien_pose.q
            if hasattr(q, 'cpu'):
                q = q.cpu().numpy().flatten()
            else:
                q = np.array(q).flatten()
            
            return np.array([p[0], p[1], p[2], q[0], q[1], q[2], q[3]])
        
        # Convert target pose
        pose_7d = sapien_pose_to_ik_format(pose)
        
        # Determine which arm to move
        if arm_index == 1:
            # Move right arm (panda_1), keep left arm (panda_2) fixed
            self._update_grasp_visual(pose, None)
            left_pose = sapien_pose_to_ik_format(self.env_agent.tcp_2.pose)
            # left_pose = None
            right_pose = pose_7d
            link_name = "panda_1_hand_tcp"
        else:
            # Move left arm (panda_2), keep right arm (panda_1) fixed
            self._update_grasp_visual(None, pose)
            left_pose = pose_7d
            right_pose = sapien_pose_to_ik_format(self.env_agent.tcp_1.pose)
            # right_pose = None
            link_name = "panda_2_hand_tcp"
        
        # Solve IK
        ik_result = self.planner.IK(
            left_target_pose=left_pose,
            right_target_pose=right_pose,
            start_qpos=current_qpos,
            left_link_name="panda_2_hand_tcp",
            right_link_name="panda_1_hand_tcp",
        )
        
        if isinstance(ik_result, str):
            if self.debug:
                print(f"IK Failed: {ik_result}")
            self.render_wait()
            return -1
        # Plan with fixed joints for the other arm
        # (The planner will automatically fix gripper joints)
        result = self.planner.plan_qpos_to_qpos(
            goal_qposes=[ik_result],
            current_qpos=current_qpos,
            time_step=self.base_env.control_timestep,
            planning_time=20.0,
            planner_name="RRTConnect",
        )
        
        if result["status"] != "Success":
            if self.debug:
                print(f"Planning failed: {result['status']}")
            self.render_wait()
            return -1
                
        if dry_run:
            return result
        
        return self.follow_path(result, refine_steps=refine_steps, arm_index=arm_index)
    
    def open_gripper(self, arm_index: int = None, t: int = 6):
        """
        Open gripper(s).
        
        Args:
            arm_index: 1 for arm_1, 2 for arm_2, None for both
            t: Number of timesteps to execute
        """
        if arm_index is None:
            self.gripper_1_state = self.OPEN
            self.gripper_2_state = self.OPEN
        elif arm_index == 1:
            self.gripper_1_state = self.OPEN
        else:
            self.gripper_2_state = self.OPEN
            
        g1_val = 1 if self.gripper_1_state == self.OPEN else -1
        g2_val = 1 if self.gripper_2_state == self.OPEN else -1

        # Assuming sapien order differs from planner order
        qpos = self.robot.get_qpos()[0, :18].cpu().numpy()
        qpos = self._convert_qpos_sapien_to_planner(qpos)
        
        for i in range(t):
            action = np.hstack([qpos[0:7], g1_val, qpos[9:16], g2_val])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            
            if self.debug:
                print(f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}")
            
            if self.vis:
                self.base_env.render_human()
        
        return obs, reward, terminated, truncated, info
    
    def close_gripper(self, arm_index: int = None, t: int = 6):
        """
        Close gripper(s).
        
        Args:
            arm_index: 1 for arm_1, 2 for arm_2, None for both
            t: Number of timesteps to execute
        """
        if arm_index is None:
            self.gripper_1_state = self.CLOSED
            self.gripper_2_state = self.CLOSED
        elif arm_index == 1:
            self.gripper_1_state = self.CLOSED
        else:
            self.gripper_2_state = self.CLOSED
        
        g1_val = 1 if self.gripper_1_state == self.OPEN else -1
        g2_val = 1 if self.gripper_2_state == self.OPEN else -1
        
        qpos = self.robot.get_qpos()[0, :18].cpu().numpy()
        qpos = self._convert_qpos_sapien_to_planner(qpos)
        
        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos[0:7], g1_val, qpos[9:16], g2_val])
            else:
                action = np.hstack([qpos, qpos * 0, self.gripper_1_state, self.gripper_2_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            
            if self.debug:
                print(f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}")
            
            if self.vis:
                self.base_env.render_human()
        
        return obs, reward, terminated, truncated, info
    
    def plan_dual_arm_constrained_motion(
        self,
        obj_goal_pos: np.ndarray,
        current_object_pose,
        current_left_pose,
        current_right_pose,
        dry_run: bool = False,
        refine_steps: int = 0
    ):
        """
        Plan motion where both arms hold an object (constrained motion).
        
        Args:
            goal_qpos: Target joint configuration
            left_grasp_transform: 4x4 transform from left hand to object
            right_grasp_transform: 4x4 transform from right hand to object
            dry_run: If True, only plan without execution
            refine_steps: Additional steps to hold final position
        """

        current_qpos_sapien = self.robot.get_qpos().cpu().numpy()[0]
        current_qpos = self._convert_qpos_sapien_to_planner(current_qpos_sapien)

        
        current_object_pose = pose_to_matrix(current_object_pose)
        current_left_pose = pose_to_matrix(current_left_pose)
        current_right_pose = pose_to_matrix(current_right_pose)
        
        left_grasp_T = np.linalg.inv(current_left_pose) @ current_object_pose
        right_grasp_T = np.linalg.inv(current_right_pose) @ current_object_pose
        
        goal_result = self.planner.plan_dual_arm_grasp(
            target_object_pose=obj_goal_pos,
            left_grasp_T=left_grasp_T,
            right_grasp_T=right_grasp_T,
            start_qpos=current_qpos
        )

        if goal_result["status"] != "Success":
            return -1

        goal_qpos = goal_result["qpos"]
                
        result = self.planner.plan_dual_arm_constrained(
            start_qpos=current_qpos,
            goal_qpos=goal_qpos,
            left_grasp_T=left_grasp_T,
            right_grasp_T=right_grasp_T,
            time_step=self.base_env.control_timestep,
            planning_time=10.0
        )
        
        if result["status"] != "Success":
            if self.debug:
                print(f"Constrained planning failed: {result['status']}")
            self.render_wait()
            return -1
        
        if dry_run:
            return result
        
        return self.follow_path(result, refine_steps=refine_steps)


    def move_dual_arm_screw_constrained(
        self,
        obj_goal_pose: sapien.Pose,
        current_object_pose: sapien.Pose,
        current_left_pose: sapien.Pose,
        current_right_pose: sapien.Pose,
        dry_run: bool = False,
        step_size: float = 0.01
    ):
        """Coordinated Dual-Arm Screw Motion."""
        
        # 1. Sync Planner
        current_qpos_sapien = self.robot.get_qpos().cpu().numpy()[0]
        current_qpos = self._convert_qpos_sapien_to_planner(current_qpos_sapien)
        self.planner.robot.set_qpos(current_qpos, True)

        # 2. Calculate Grasp Transforms
        T_obj = pose_to_matrix(current_object_pose)
        T_L = pose_to_matrix(current_left_pose)
        T_R = pose_to_matrix(current_right_pose)
        
        # T_L_O = inv(T_L) * T_O (object in left hand frame)
        left_grasp_T = np.linalg.inv(T_L) @ T_obj
        right_grasp_T = np.linalg.inv(T_R) @ T_obj
        
        T_obj_from_L = T_L @ left_grasp_T
        T_obj_from_R = T_R @ right_grasp_T
        if self.debug:
            print("\n=== Grasp Transform Verification ===")
            print(f"Original object pos: {T_obj[:3, 3]}")
            print(f"From left hand:      {T_obj_from_L[:3, 3]}")
            print(f"From right hand:     {T_obj_from_R[:3, 3]}")
            print(f"Reconstruction error (L): {np.linalg.norm(T_obj[:3, 3] - T_obj_from_L[:3, 3]):.6f}")
            print(f"Reconstruction error (R): {np.linalg.norm(T_obj[:3, 3] - T_obj_from_R[:3, 3]):.6f}")
        
        # If error is large, something is wrong with the input poses
        if np.linalg.norm(T_obj[:3, 3] - T_obj_from_L[:3, 3]) > 0.01:
            if self.debug:
                print("WARNING: Large grasp transform error! Check input poses.")
        
        
        # 3. Prepare Target
        obj_goal_7d = np.concatenate([obj_goal_pose.p, obj_goal_pose.q])
        
        # 4. Viz Callback
        def screw_vis_callback(q_planner):
             if self.vis:
                q_sapien = self._convert_qpos_planner_to_sapien(q_planner)
                self.robot.set_qpos(q_sapien)
                self.base_env.render_human()
        
        # 5. Execute Coordinated Planner
        result = self.planner.plan_dual_arm_screw_constrained(
            start_qpos=current_qpos,
            obj_goal_pose=obj_goal_7d,
            left_grasp_T=left_grasp_T,
            right_grasp_T=right_grasp_T,
            step_size=step_size,
            visualize_callback=None
        )
        
        if result["status"] != "Success":
            return -1
                
        if dry_run: return result
        return self.follow_path(result)
    
    def move_to_pose_pair_with_screw(
        self,
        left_pose: sapien.Pose,
        right_pose: sapien.Pose,
        dry_run: bool = False
    ):
        """
        Move both arms linearly (screw motion).
        """
        left_pose = to_sapien_pose(left_pose)
        right_pose = to_sapien_pose(right_pose)
        
        self._update_grasp_visual(right_pose, left_pose)

        # Get current qpos
        current_qpos_sapien = self.robot.get_qpos().cpu().numpy()[0]
        current_qpos = self._convert_qpos_sapien_to_planner(current_qpos_sapien)
        self.planner.robot.set_qpos(current_qpos, True)
        
        # Prepare targets
        left_target = np.concatenate([left_pose.p, left_pose.q])
        right_target = np.concatenate([right_pose.p, right_pose.q])
        
        def screw_vis_callback(q_planner):
            if self.vis:
                # Convert planner qpos back to SAPIEN qpos
                q_sapien = self._convert_qpos_planner_to_sapien(q_planner)
                # Update the actual robot in the scene
                self.robot.set_qpos(q_sapien)
                # Render the frame
                self.base_env.render_human()
                
        # Call Planner
        result = self.planner.plan_screw(
            target_pose_L=left_target,
            target_pose_R=right_target,
            current_qpos=current_qpos,
            step_size=0.01, # 1cm/step roughly
            visualize_callback=None
        )
        if result["status"] != "Success":
            return -1
            
        if dry_run: return result
        return self.follow_path(result)
    
    def move_to_pose_with_screw(
        self,
        pose: sapien.Pose,
        arm_index: int, 
        dry_run: bool = False
    ):
        """
        Move a single arm linearly (screw motion).
        """
        pose = to_sapien_pose(pose)
        
        # Get current qpos
        current_qpos_sapien = self.robot.get_qpos().cpu().numpy()[0]
        current_qpos = self._convert_qpos_sapien_to_planner(current_qpos_sapien)
        self.planner.robot.set_qpos(current_qpos, True)
        # Prepare targets
        # Convert pose to [x, y, z, qw, qx, qy, qz]
        target_7d = np.concatenate([pose.p, pose.q])
        left_target = None
        right_target = None
        
        if arm_index == 1: # Right Arm (Panda 1)
            right_target = target_7d
            self._update_grasp_visual(target_7d, None)
            curr_pose_L = self.planner.pinocchio_model.get_link_pose(self.planner.link_name_2_idx["panda_2_hand_tcp"])
            left_target = np.concatenate([curr_pose_L[:3], curr_pose_L[3:]])
        else: # Left Arm (Panda 2)
            left_target = target_7d
            self._update_grasp_visual(None, target_7d)
            curr_pose_R = self.planner.pinocchio_model.get_link_pose(self.planner.link_name_2_idx["panda_1_hand_tcp"])
            right_target = np.concatenate([curr_pose_R[:3], curr_pose_R[3:]])
        
        
        def screw_vis_callback(q_planner):
            if self.vis:
                # Convert planner qpos back to SAPIEN qpos
                q_sapien = self._convert_qpos_planner_to_sapien(q_planner)
                # Update the actual robot in the scene
                self.robot.set_qpos(q_sapien)
                # Render the frame
                self.base_env.render_human()
        
        # Call Planner
        result = self.planner.plan_screw(
            target_pose_L=left_target,
            target_pose_R=right_target,
            current_qpos=current_qpos,
            step_size=0.005, # 1cm/step roughly
            visualize_callback=None
        )
        if result["status"] != "Success":
            return -1
        if dry_run: return result
        return self.follow_path(result, arm_index=arm_index)
    
    def _sync_planner_robot_state(self):
            """
            Force MPLib to adopt the current SAPIEN joint angles.
            This prevents the robot from snapping to 'Zero' at the start of a plan.
            """
            # 1. Get the current actual angles from SAPIEN
            current_qpos = self.robot.get_qpos().cpu().numpy()[0]
            
            # 2. Force the Planner's internal robot to match
            self.planner.robot.set_qpos(current_qpos, True)