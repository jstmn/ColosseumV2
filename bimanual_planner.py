from __future__ import annotations

import os
from typing import Optional, Sequence
import numpy as np
import toppra as ta
import toppra.algorithm as algo
import toppra.constraint as constraint
from transforms3d.quaternions import mat2quat, quat2mat
import pinocchio
from mplib.pymp import ArticulatedModel, PlanningWorld
from mplib.pymp.planning import ompl
# from mplib.planning.ompl import OMPLPlanner
from pathlib import Path

class BimanualPlanner:
    def __init__(
            self,
            urdf: str,
            move_group: str | Sequence[str], 
            srdf: str = "",
            package_keyword_replacement: str = "",
            user_link_names: Sequence[str] = [],
            user_joint_names: Sequence[str] = [],
            joint_vel_limits: Optional[Sequence[float] | np.ndarray] = None,
            joint_acc_limits: Optional[Sequence[float] | np.ndarray] = None,
            include_mobile_base: bool = False, 
            **kwargs
            ):
        if joint_vel_limits is None:
            joint_vel_limits = []
        if joint_acc_limits is None:
            joint_acc_limits = []
        
        print("Initializing BimanualPlanner for Dual Panda Table...")
        self.urdf = str(urdf)
        urdf = self.replace_package_keyword(package_keyword_replacement)
        self.urdf = str(urdf)
        print(self.urdf)
        # Handle SRDF
        self.srdf = urdf.replace(".urdf", ".srdf")
        if srdf == "":
            potential_srdf = self.urdf.replace(".urdf", ".srdf")
            if os.path.exists(potential_srdf):
                self.srdf = potential_srdf
            else:
                print("Generating dummy SRDF to prevent crash...")
                dummy_srdf_path = self.urdf.replace(".urdf", "_dummy.srdf")
                with open(dummy_srdf_path, "w") as f:
                    f.write(f'<?xml version="1.0"?><robot name="dummy_robot"></robot>')
                self.srdf = dummy_srdf_path
        else:
            self.srdf = srdf
        
        # Build Pinocchio model to get full list of joints/links
        temp_robot = pinocchio.buildModelFromUrdf(urdf)
        all_joint_names = list(temp_robot.names[1:])
        print("\n=== Pinocchio Joint Order ===")
        for i, name in enumerate(all_joint_names):
            print(f"  {i}: {name}")

        all_link_names = [
            f.name for f in temp_robot.frames 
            if f.type == pinocchio.FrameType.BODY
        ]

        # --- UPDATED: No mobile base in this URDF ---
        # The robot is fixed to the table, so we don't need to filter out root joints
        if not user_joint_names:
            user_joint_names = all_joint_names

        self.user_joint_names = user_joint_names
        
        # Initialize ArticulatedModel
        self.robot = ArticulatedModel(
            urdf_filename=urdf,
            srdf_filename=self.srdf,
            gravity=np.array([0, 0, -9.81], dtype=np.float64).reshape(3, 1),
            link_names=all_link_names,    
            joint_names=all_joint_names,  
            convex=False,                 
            verbose=False
        )
        
        self.pinocchio_model = self.robot.get_pinocchio_model()
        self.user_link_names = self.pinocchio_model.get_link_names()
        self.user_joint_names = self.pinocchio_model.get_joint_names()
        # self.planning_world = PlanningWorld(
        #     [self.robot],
        #     kwargs.get("normal_objects", []),
        # )
        normal_objects = kwargs.get("normal_objects", [])
        # Generate dummy names for objects if they don't have a 'name' attribute
        normal_object_names = [getattr(o, "name", f"obj_{i}") for i, o in enumerate(normal_objects)]

        self.planning_world = PlanningWorld(
            [self.robot],          # articulations
            ["panda_dual_arm"],    # articulation_names (Must be a list of strings)
            normal_objects,        # normal_objects
            normal_object_names    # normal_object_names (Must be a list of strings)
        )
        # Map names to indices
        self.joint_name_2_idx = {j: i for i, j in enumerate(self.user_joint_names)}
        self.link_name_2_idx = {l: i for i, l in enumerate(self.user_link_names)}
    
        if srdf == "":
            self.generate_collision_pair() 
            self.robot.update_SRDF(self.srdf)

        # --- UPDATED: Move Group Logic for New URDF ---
        self.move_group = move_group
        self.move_group_joint_indices = []
        target_links = []
        
        # Mapped to the new URDF link names
        if move_group == "dual_arm":
            target_links = ["panda_1_hand", "panda_2_hand"]
        elif isinstance(move_group, list):
            target_links = move_group
        else:
            target_links = [move_group]

        # Validate links exist
        for link in target_links:
            assert link in self.user_link_names, f"Link {link} not found in robot model."

        # --- UPDATED: Joint Names for dual_panda_table.urdf ---
        # 14 DOF total (7 for each arm). Mobile base joints removed.
        target_joint_names = [
            # Panda 1 (7 arm + 2 gripper)
            "panda_1_joint1", "panda_1_joint2", "panda_1_joint3", "panda_1_joint4", 
            "panda_1_joint5", "panda_1_joint6", "panda_1_joint7",
            "panda_1_finger_joint1", "panda_1_finger_joint2",
            # Panda 2 (7 arm + 2 gripper)
            "panda_2_joint1", "panda_2_joint2", "panda_2_joint3", "panda_2_joint4", 
            "panda_2_joint5", "panda_2_joint6", "panda_2_joint7",
            "panda_2_finger_joint1", "panda_2_finger_joint2"
        ]

        # Find indices
        for name in target_joint_names:
            if name in self.joint_name_2_idx:
                self.move_group_joint_indices.append(self.joint_name_2_idx[name])
            else:
                print(f"Warning: Joint {name} not found in robot model!")
        
        self.move_group_joint_indices = sorted(self.move_group_joint_indices)
        print(f"Manually compiled {len(self.move_group_joint_indices)} active joints.")
        
        self.joint_types = self.pinocchio_model.get_joint_types()
        self.joint_limits = np.concatenate(self.pinocchio_model.get_joint_limits())
        
        self.joint_vel_limits = (
            joint_vel_limits
            if len(joint_vel_limits)
            else np.ones(len(self.move_group_joint_indices))
        )
        self.joint_acc_limits = (
            joint_acc_limits
            if len(joint_acc_limits)
            else np.ones(len(self.move_group_joint_indices))
        )
        
        if len(target_links) == 1:
            self.move_group_link_id = self.link_name_2_idx[target_links[0]]
        else:
            self.move_group_link_id = [self.link_name_2_idx[l] for l in target_links]

        # self.planning_world = PlanningWorld([self.robot], [])
        normal_objects = kwargs.get("normal_objects", [])
        # Generate dummy names for objects if they don't have a 'name' attribute
        normal_object_names = [getattr(o, "name", f"obj_{i}") for i, o in enumerate(normal_objects)]

        self.planning_world = PlanningWorld(
            [self.robot],          # articulations
            ["panda_dual_arm"],    # articulation_names (Must be a list of strings)
            normal_objects,        # normal_objects
            normal_object_names    # normal_object_names (Must be a list of strings)
        )
        self.planner = ompl.OMPLPlanner(world=self.planning_world)

    def replace_package_keyword(self, package_keyword_replacement):
        rtn_urdf = self.urdf
        with open(self.urdf) as in_f:
            content = in_f.read()
            if "package://" in content:
                rtn_urdf = self.urdf.replace(".urdf", "_package_keyword_replaced.urdf")
                content = content.replace("package://", package_keyword_replacement)
                # if not os.path.exists(rtn_urdf):
                with open(rtn_urdf, "w") as out_f:
                    out_f.write(content)
        return rtn_urdf

    def generate_collision_pair(self, sample_time=1000, echo_freq=100000):
        print("Generating collision pairs (no SRDF provided)...")
        n_link = len(self.user_link_names)
        cnt = np.zeros((n_link, n_link), dtype=np.int32)
        for i in range(sample_time):
            qpos = self.pinocchio_model.get_random_configuration()
            self.robot.set_qpos(qpos, True)
            collisions = self.planning_world.collide_full()
            for collision in collisions:
                u = self.link_name_2_idx[collision.link_name1]
                v = self.link_name_2_idx[collision.link_name2]
                cnt[u][v] += 1
            if i % echo_freq == 0:
                print("Finish %.1f%%!" % (i * 100 / sample_time))

        import xml.etree.ElementTree as ET
        from xml.dom import minidom

        root = ET.Element("robot")
        robot_name = self.urdf.split("/")[-1].split(".")[0]
        root.set("name", robot_name)
        self.srdf = self.urdf.replace(".urdf", ".srdf")

        for i in range(n_link):
            for j in range(n_link):
                if cnt[i][j] == sample_time:
                    link1 = self.user_link_names[i]
                    link2 = self.user_link_names[j]
                    collision = ET.SubElement(root, "disable_collisions")
                    collision.set("link1", link1)
                    collision.set("link2", link2)
                    collision.set("reason", "Default")
        with open(self.srdf, "w") as srdf_file:
            srdf_file.write(
                minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
            )
            srdf_file.close()
        print("Saving the SRDF file to %s" % self.srdf)

    def wrap_joint_limit(self, q) -> bool:
        n = len(q)
        flag = True
        for i in range(n):
            if self.joint_types[i].startswith("JointModelR"):
                if -1e-3 <= q[i] - self.joint_limits[i][0] < 0:
                    continue
                q[i] -= (
                    2 * np.pi * np.floor((q[i] - self.joint_limits[i][0]) / (2 * np.pi))
                )
                if q[i] > self.joint_limits[i][1] + 1e-3:
                    flag = False
            else:
                if (
                    q[i] < self.joint_limits[i][0] - 1e-3
                    or q[i] > self.joint_limits[i][1] + 1e-3
                ):
                    flag = False
        return flag

    def pad_qpos(self, qpos, articulation=None):
        if len(qpos) == len(self.move_group_joint_indices):
            tmp = (
                articulation.get_qpos()
                if articulation is not None
                else self.robot.get_qpos()
            )
            for k, idx in enumerate(self.move_group_joint_indices):
                tmp[idx] = qpos[k]            
            qpos = tmp
        return qpos

    def check_for_collision(
        self,
        collision_function,
        articulation: Optional[ArticulatedModel] = None,
        qpos: Optional[np.ndarray] = None,
    ) -> list:
        if articulation is None:
            articulation = self.robot
        if qpos is None:
            qpos = articulation.get_qpos()
        qpos = self.pad_qpos(qpos, articulation)
        old_qpos = articulation.get_qpos()
        articulation.set_qpos(qpos, True)
        collisions = collision_function()
        articulation.set_qpos(old_qpos, True)
        return collisions
    
    def check_for_self_collision(
        self,
        articulation: Optional[ArticulatedModel] = None,
        qpos: Optional[np.ndarray] = None,
    ) -> list:
        return self.check_for_collision(
            self.planning_world.self_collide, articulation, qpos
        )

    def check_for_env_collision(
        self,
        articulation: Optional[ArticulatedModel] = None,
        qpos: Optional[np.ndarray] = None,
        with_point_cloud=False,
        use_attach=False,
    ) -> list:
        return self.check_for_collision(
            self.planning_world.collide_with_others, articulation, qpos
        )

    # --- UPDATED: Default link names for new URDF ---
    def IK(self, left_target_pose=None, right_target_pose=None, start_qpos=None, left_link_name="panda_1_hand_tcp", right_link_name="panda_2_hand_tcp", threshold=1e-3, max_iter=100, step_size=0.1, attempts=200):
        print("HI FROM IK")
        if start_qpos is None:
            start_qpos = self.robot.get_qpos()

        left_idx = self.link_name_2_idx.get(left_link_name, -1)
        right_idx = self.link_name_2_idx.get(right_link_name, -1)
        
        if left_idx == -1 or right_idx == -1:
            print(f"Error: Could not find link names {left_link_name} or {right_link_name}")
            return "Failed"

        # Use all active indices for IK (Mobile base joints are gone, so just use move_group indices)
        active_indices = self.move_group_joint_indices
        
        # def to_SE3(pose_7d):
        #     if hasattr(pose_7d, 'rotation'): return pose_7d
        #     R = np.array(pose_7d[3:])
        #     pos = np.array(pose_7d[:3])
        #     R = R/np.linalg.norm(R)
        #     qx,qy,qz,qw = pose_7d[3], pose_7d[4], pose_7d[5], pose_7d[6]
        #     quat = pinocchio.Quaternion(qw, qx, qy, qz)
        #     return pinocchio.SE3(quat, pos)
        # --- HELPER: ROBUST CONVERSION ---
        # Converts List (User) OR mplib.Pose (Wrapper) -> pin.SE3
        def to_pin_SE3(pose_input):
            if pose_input is None: return None
            
            # Case A: Input is a list [x, y, z, qw, qx, qy, qz] (from User)
            if isinstance(pose_input, (list, np.ndarray)):
                pos = np.array(pose_input[:3])
                # Explicitly construct quaternion from (w, x, y, z)
                # Pinocchio constructor signature is (w, x, y, z)
                quat = pinocchio.Quaternion(
                    float(pose_input[3]), 
                    float(pose_input[4]), 
                    float(pose_input[5]), 
                    float(pose_input[6])
                )
                quat.normalize()
                return pinocchio.SE3(quat, pos)
            
            # Case B: Input is mplib.pymp.Pose (from Wrapper)
            # SAFE METHOD: Use Transformation Matrix to avoid quaternion order confusion
            else:
                mat = pose_input.to_transformation_matrix() # Returns 4x4 numpy array
                R = mat[:3, :3]
                t = mat[:3, 3]
                return pinocchio.SE3(R, t)
        
        target_L_se3 = to_pin_SE3(left_target_pose)
        target_R_se3 = to_pin_SE3(right_target_pose)
        print(target_L_se3, target_R_se3)
        for attempt in range(attempts):
            if attempt == 0:
                q = np.copy(start_qpos)
            else:
                random_full_q = self.pinocchio_model.get_random_configuration()
                q = np.copy(start_qpos)
                q[active_indices] = random_full_q[active_indices]
                print(f"  [IK] Attempt {attempt+1}: Restarting with random configuration...")
            for i in range(max_iter):
                q = np.array(q, dtype=np.float64)
                # print(len(q), self.pinocchio_model.nq)
                self.pinocchio_model.compute_forward_kinematics(q)
                self.pinocchio_model.compute_full_jacobian(q)
                error_stack = []
                J_stack = []
                
                if target_L_se3 is not None:
                    mplib_pose_L = self.pinocchio_model.get_link_pose(left_idx)
                    curr_L_se3 = to_pin_SE3(mplib_pose_L)
                    dMf = curr_L_se3.actInv(target_L_se3)
                    err_L = pinocchio.log(dMf).vector
                    error_stack.append(err_L)
                    J_L = self.pinocchio_model.get_link_jacobian(left_idx, local=True)
                    J_stack.append(J_L[:, active_indices])
                
                if target_R_se3 is not None:
                    mplib_pose_R = self.pinocchio_model.get_link_pose(right_idx)
                    curr_R_se3 = to_pin_SE3(mplib_pose_R)
                    dMf = curr_R_se3.actInv(target_R_se3)
                    err_R = pinocchio.log(dMf).vector
                    error_stack.append(err_R)
                    J_R = self.pinocchio_model.get_link_jacobian(right_idx, local=True)
                    J_stack.append(J_R[:, active_indices])
                if not error_stack:
                    
                    return "Failed", q
                err_total = np.concatenate(error_stack)
                
                if np.linalg.norm(error_stack) < threshold:
                    # if left_target_pose is None:
                    #     q = np.hstack([q[0:9], start_qpos[9:]])
                    # elif right_target_pose is None:
                    #     q = np.hstack([start_qpos[0:9], q[9:]])
                    return q
                
                J_total = np.vstack(J_stack)
                damp = 1e-3
                JJt = J_total @ J_total.T
                dq = J_total.T @ np.linalg.inv(JJt + damp * np.eye(len(err_total))) @ err_total
                
                q[active_indices] += step_size * dq
                if i == 0:
                    limits = self.pinocchio_model.get_joint_limits()
                    limits_stack = np.vstack(limits)
                    lower_lim = limits_stack[:, 0]
                    upper_lim = limits_stack[:, 1]
                
                q = np.clip(q, lower_lim, upper_lim)

        print("❌ IK Failed to converge.")
        return "Failed"

    def plan_dual_arm_grasp(
        self,
        target_object_pose,
        left_grasp_T,
        right_grasp_T,
        start_qpos,
        treshold=1e-3,
    ):
        T_obj_world = np.eye(4)
        T_obj_world[:3, :3] = quat2mat(target_object_pose[3:])
        T_obj_world[:3, 3] = target_object_pose[:3]
        
        T_left_target = T_obj_world @ np.linalg.inv(left_grasp_T)
        left_target_7d = np.zeros(7)
        left_target_7d[:3] = T_left_target[:3, 3]
        left_target_7d[3:] = mat2quat(T_left_target[:3, :3])
        
        T_right_target = T_obj_world @ np.linalg.inv(right_grasp_T)
        right_target_7d = np.zeros(7)
        right_target_7d[:3] = T_right_target[:3, 3]
        right_target_7d[3:] = mat2quat(T_right_target[:3, :3])
        
        status, q_result = self.IK(
            left_target_7d,
            right_target_7d,
            start_qpos,
            threshold=treshold,
        )
        
        if status != "Failed":
            return {
                "status": "Success",
                "qpos": q_result,
                "left_target": left_target_7d,
                "right_target": right_target_7d
            }
        else:
            return {"status": "IK Failed"}

    # --- UPDATED: Link names for new URDF ---
    def create_dual_arm_constraint_function(self, left_grasp_T, right_grasp_T):
        left_idx = self.link_name_2_idx["panda_1_hand"]
        right_idx = self.link_name_2_idx["panda_2_hand"]
        
        def to_SE3(mat):
            return pinocchio.SE3(mat[:3, :3], mat[:3, 3])
        
        se3_grasp_L = to_SE3(left_grasp_T)
        se3_grasp_R = to_SE3(right_grasp_T)
        
        def constraint_function(q):
            full_q = self.pad_qpos(q)
            self.pinocchio_model.compute_forward_kinematics(full_q)
            p_L = self.pinocchio_model.get_link_pose(left_idx)
            T_L = pinocchio.SE3(quat2mat(p_L[3:]), np.array(p_L[:3]))
            
            p_R = self.pinocchio_model.get_link_pose(right_idx)
            T_R = pinocchio.SE3(quat2mat(p_R[3:]), np.array(p_R[:3]))
            
            T_Obj_L = T_L.act(se3_grasp_L)
            T_Obj_R = T_R.act(se3_grasp_R)
            
            return pinocchio.log(T_Obj_L.actInv(T_Obj_R)).vector
        
        def numerical_jacobian(q):
            eps = 1e-4
            n = len(q)
            J = np.zeros((6, n))
            f0 = constraint_function(q)
            
            for i in range(n):
                q_plus = q.copy()
                q_plus[i] += eps
                f_plus = constraint_function(q_plus)
                J[:, i] = (f_plus - f0) / eps 
            return J
        
        return constraint_function, numerical_jacobian
    
    def TOPP(self, path, step=0.1, verbose=False):
        N_samples = path.shape[0]
        dof = path.shape[1]
        assert dof == len(self.joint_vel_limits)
        assert dof == len(self.joint_acc_limits)
        ss = np.linspace(0, 1, N_samples)
        path = ta.SplineInterpolator(ss, path)
        pc_vel = constraint.JointVelocityConstraint(self.joint_vel_limits)
        pc_acc = constraint.JointAccelerationConstraint(self.joint_acc_limits)
        instance = algo.TOPPRA(
            [pc_vel, pc_acc], path, parametrizer="ParametrizeConstAccel"
        )
        jnt_traj = instance.compute_trajectory()
        if jnt_traj is None:
            raise RuntimeError("Fail to parameterize path")
        ts_sample = np.linspace(0, jnt_traj.duration, int(jnt_traj.duration / step))
        qs_sample = jnt_traj(ts_sample)
        qds_sample = jnt_traj(ts_sample, 1)
        qdds_sample = jnt_traj(ts_sample, 2)
        return ts_sample, qs_sample, qds_sample, qdds_sample, jnt_traj.duration

    def plan_dual_arm_constrained(
        self,
        start_qpos,
        goal_qpos,
        left_grasp_T,
        right_grasp_T,
        time_step=0.1,
        rrt_range=0.01,
        planning_time=5,
        verbose=False):
        
        constr_func, constr_jac = self.create_dual_arm_constraint_function(left_grasp_T, right_grasp_T)
        
        return self.plan_qpos_to_qpos(
            goal_qposes=[goal_qpos],
            current_qpos=start_qpos,
            time_step=time_step,
            rrt_range=rrt_range,
            planning_time=planning_time,
            planner_name="RRTConnect",
            no_simplification=True,
            constraint_function=constr_func,
            constraint_jacobian=constr_jac,
            constraint_tolerance=1e-3,
            verbose=verbose,
        )

    def _smooth_path(self, path, simplification_time=2.0, fixed_joints=None):
        """
        Smooth the path using shortcut smoothing.
        Tries to connect distant waypoints directly if collision-free.
        """
        if len(path) <= 2:
            return path
        
        import time
        start_time = time.time()
        smoothed = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
            # Try to shortcut as far as possible
            best_next = current_idx + 1
            
            for test_idx in range(len(path) - 1, current_idx, -1):
                if time.time() - start_time > simplification_time:
                    break
                
                # Check if we can go directly from current to test_idx
                if self._is_path_segment_valid(
                    path[current_idx], 
                    path[test_idx],
                    fixed_joints
                ):
                    best_next = test_idx
                    break
            
            smoothed.append(path[best_next])
            current_idx = best_next
        
        return np.array(smoothed)

    def _is_path_segment_valid(self, start, end, fixed_joints, num_checks=10):
        """Check if linear interpolation between two configs is collision-free"""
        for alpha in np.linspace(0, 1, num_checks):
            q = (1 - alpha) * start + alpha * end
            
            # Set robot to this config
            self.robot.set_qpos(q, True)
            
            # Check collisions
            if self.planning_world.collide():
                return False
        return True

    
    def plan_qpos_to_qpos(
        self,
        goal_qposes: list,
        current_qpos,
        time_step=0.1,
        rrt_range=0.25,
        planning_time=10,
        fix_joint_limits=True,
        use_point_cloud=False,
        use_attach=False,
        planner_name="RRTConnect",
        no_simplification=False,
        constraint_function=None,
        constraint_jacobian=None,
        constraint_tolerance=1e-3,
        fixed_joint_indices=None,
        verbose=False,
        simplify_path = True,
        simplification_time=3.0
        
    ):
        if fixed_joint_indices is None:
            fixed_joint_indices = []
        
        n = current_qpos.shape[0]
        if fix_joint_limits:
            for i in range(n):
                if current_qpos[i] < self.joint_limits[i][0]:
                    current_qpos[i] = self.joint_limits[i][0] + 1e-3
                if current_qpos[i] > self.joint_limits[i][1]:
                    current_qpos[i] = self.joint_limits[i][1] - 1e-3

        current_qpos = self.pad_qpos(current_qpos)
        
        self.robot.set_qpos(current_qpos, True)
        # if self.planning_world.collide():
        #     print("Invalid start state (Collision)")
        #     return {"status": "Invalid start state"}
        
        goal_qpos_ = goal_qposes
        fixed_joints = set()
        for joint_idx in fixed_joint_indices:
            fixed_joints.add(ompl.FixedJoint(0, joint_idx, current_qpos[joint_idx]))
        
        start_state = current_qpos.astype(np.float64)
        try:
            goal_states = [gq.astype(np.float64) for gq in goal_qpos_]
        except AttributeError:
            return {"status": "IK Failed"}

        # Fix Grippers (Indices updated for new URDF)
        # Panda 1 Hand: panda_1_finger_joint1/2
        # Panda 2 Hand: panda_2_finger_joint1/2
        # We need indices for them.
        gripper_names = ["panda_1_finger_joint1", "panda_1_finger_joint2", "panda_2_finger_joint1", "panda_2_finger_joint2"]
        for name in gripper_names:
            if name in self.joint_name_2_idx:
                idx = self.joint_name_2_idx[name]
                fixed_joints.add(ompl.FixedJoint(0, idx, current_qpos[idx]))

        status, path = self.planner.plan(
            start_state=start_state,
            goal_states=goal_states,
            range=rrt_range,
            time=planning_time,
            fixed_joints=fixed_joints,
            constraint_function=constraint_function,
            constraint_jacobian=constraint_jacobian,
            constraint_tolerance=constraint_tolerance,
            verbose=True,
            no_simplification=False
        )
        # print(path)
        if status == "Exact solution":
            # print("PATH:",path)
            if simplify_path and len(path) > 2:
                print("Smoothing path...")
                path = self._smooth_path(
                    path, 
                    simplification_time=simplification_time,
                    fixed_joints=fixed_joints
                )
                print(f"✓ Smoothed to {len(path)} waypoints")

            points_per_segment = 12 
            new_path_segments = []

            for i in range(len(path) - 1):
                # Interpolate between point i and i+1
                # endpoint=False ensures we don't duplicate the connection point
                segment = np.linspace(path[i], path[i+1], points_per_segment, endpoint=False)
                new_path_segments.append(segment)

            # Don't forget to add the very last point of the original path
            new_path_segments.append([path[-1]])

            # Combine into one array
            path = np.vstack(new_path_segments)
            try:
                times, pos, vel, acc, duration = self.TOPP(path, time_step)
                return {
                    "status": "Success",
                    "time": times,
                    "position": pos,
                    "velocity": vel,
                    "acceleration": acc,
                    "duration": duration,
                }
            except Exception as e:
                print(f"TOPP Parameterization Failed: {e}")
                return {"status": "Success (No TOPP)", "position": path}
        else:
            return {"status": "RRT Failed. %s" % status}

    # Helper methods for collision objects
    def update_point_cloud(self, pc, radius=1e-3):
        self.planning_world.update_point_cloud(pc, radius)

    def update_attached_tool(self, fcl_collision_geometry, pose, link_id=-1):
        if link_id == -1:
            link_id = self.move_group_link_id
        self.planning_world.update_attached_tool(fcl_collision_geometry, link_id, pose)

    def update_attached_sphere(self, radius, pose, link_id=-1):
        if link_id == -1:
            link_id = self.move_group_link_id
        self.planning_world.update_attached_sphere(radius, link_id, pose)

    def update_attached_box(self, size, pose, link_id=-1):
        if link_id == -1:
            link_id = self.move_group_link_id
        self.planning_world.update_attached_box(size, link_id, pose)

    def update_attached_mesh(self, mesh_path, pose, link_id=-1):
        if link_id == -1:
            link_id = self.move_group_link_id
        self.planning_world.update_attached_mesh(mesh_path, link_id, pose)

    def set_base_pose(self, pose):
            # Import Pose if it's not globally available, or ensure it's imported at top
            # from mplib.pymp import Pose 
            # import numpy as np

            # # Convert numpy array/list to mplib.pymp.Pose
            # if isinstance(pose, (np.ndarray, list)):
            #     # Helper to handle the conversion. 
            #     # modifying based on the array passed: [x, y, z, qw, qx, qy, qz]
            #     pose = np.array(pose) # Ensure it's numpy for slicing
            
            #     # Case 1: Flat 7D Array [x, y, z, qw, qx, qy, qz]
            #     if pose.flatten().shape[0] == 7:
            #         pose = Pose(p=pose[:3], q=pose[3:])
                    
            #     # Case 2: 4x4 Transformation Matrix
            #     elif pose.shape == (4, 4):
            #         pose = Pose(pose)
            #     else:
            #         raise ValueError(f"Invalid pose shape: {pose.shape}. Expected 7 (flat) or 4x4.")
                    
            # self.robot.set_base_pose(pose)
            self.robot.set_base_pose(pose)
            
    def set_normal_object(self, name, collision_object):
        self.planning_world.set_normal_object(name, collision_object)

    def remove_normal_object(self, name):
        return self.planning_world.remove_normal_object(name)
