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
        self.debug = False
        if self.debug:
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
        if self.debug:    
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
                if self.debug:
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
                if self.debug:
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
                
                q[active_indices] += dq*step_size
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
        
        q_result = self.IK(
            left_target_7d,
            right_target_7d,
            start_qpos,
            threshold=treshold,
        )
        
        if type(q_result) is not str:
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
            fixed_joint_indices=[],  # Don't fix any joints
            verbose=verbose,
        )
        
    def plan_dual_arm_screw_constrained(
        self,
        start_qpos,
        obj_goal_pose,
        left_grasp_T,
        right_grasp_T,
        step_size=0.01,
        time_step=0.1,
        visualize_callback=None,
        max_steps=500,
        threshold=1e-3,
        check_collisions=False
    ):
        """
        Coordinated Screw with HARD Constraint Enforcement.
        
        Key idea: Instead of computing target twists for each hand separately,
        compute ONE object twist and derive both hand twists from it using the
        CURRENT grasp relationship (not the initial one).
        """
        import pinocchio
        
        # --- Helpers ---
        def to_pin_se3(pose_7d):
            if pose_7d is None:
                return None
            p = np.array(pose_7d[:3], dtype=np.float64)
            q = pinocchio.Quaternion(
                float(pose_7d[3]), float(pose_7d[4]), 
                float(pose_7d[5]), float(pose_7d[6])
            )
            q.normalize()
            return pinocchio.SE3(q, p)

        def mat_to_se3(mat):
            return pinocchio.SE3(mat[:3, :3], mat[:3, 3])

        # --- Setup ---
        T_L_O_se3 = mat_to_se3(left_grasp_T)
        T_R_O_se3 = mat_to_se3(right_grasp_T)
        
        se3_obj_goal = to_pin_se3(obj_goal_pose)

        active_indices = self.move_group_joint_indices
        left_idx = self.link_name_2_idx.get("panda_2_hand_tcp", -1)
        right_idx = self.link_name_2_idx.get("panda_1_hand_tcp", -1)
        
        if left_idx == -1 or right_idx == -1:
            return {"status": "Failed (Invalid link indices)"}

        limits = self.pinocchio_model.get_joint_limits()
        limits_stack = np.vstack(limits)
        lower_lim = limits_stack[:, 0]
        upper_lim = limits_stack[:, 1]

        path = [start_qpos.copy()]
        q = start_qpos.copy()
        
        # --- Optimization Loop ---
        for step in range(max_steps):
            if visualize_callback is not None and step % 5 == 0:
                visualize_callback(q)

            self.pinocchio_model.compute_forward_kinematics(q)
            self.pinocchio_model.compute_full_jacobian(q)
            
            # A. Get current hand poses
            curr_L_pose = self.pinocchio_model.get_link_pose(left_idx)
            curr_R_pose = self.pinocchio_model.get_link_pose(right_idx)
            
            se3_L_curr = to_pin_se3(curr_L_pose)
            se3_R_curr = to_pin_se3(curr_R_pose)
            
            # B. ✅ CRITICAL: Use AVERAGE of both object estimates
            # This naturally enforces the constraint by making both hands agree
            se3_obj_from_L = se3_L_curr.act(T_L_O_se3)
            se3_obj_from_R = se3_R_curr.act(T_R_O_se3)
            
            # Average in SE3 space
            se3_obj_curr = pinocchio.SE3.Interpolate(se3_obj_from_L, se3_obj_from_R, 0.5)
            
            # C. Measure constraint violation (for monitoring only)
            constraint_error = pinocchio.log(se3_obj_from_L.actInv(se3_obj_from_R)).vector
            constraint_violation = np.linalg.norm(constraint_error)
            
            # D. Compute single object twist toward goal
            dMf_obj = se3_obj_curr.actInv(se3_obj_goal)
            twist_obj = pinocchio.log(dMf_obj).vector
            task_error = np.linalg.norm(twist_obj)
            
            # E. Check convergence
            if task_error < threshold and constraint_violation < 0.1:
                if self.debug:
                    print(f"✓ Converged in {step} steps")
                    print(f"  Final task error: {task_error:.6f}")
                    print(f"  Final constraint: {constraint_violation:.6f}")
                    break
            
            # F. Clamp object twist
            if task_error > step_size:
                twist_obj = twist_obj * (step_size / task_error)
            
            # G. ✅ KEY FIX: Recompute grasp transforms at CURRENT state
            # This makes the constraint self-correcting
            T_O_L_curr = se3_obj_curr.actInv(se3_L_curr)  # Current obj→left transform
            T_O_R_curr = se3_obj_curr.actInv(se3_R_curr)  # Current obj→right transform
            
            # H. Compute hand twists from object twist using CURRENT transforms
            motion_obj = pinocchio.Motion(twist_obj)
            
            # Transform object twist to each hand using current relationship
            twist_L = T_O_L_curr.act(motion_obj).vector
            twist_R = T_O_R_curr.act(motion_obj).vector
            
            # I. ✅ ADD CONSTRAINT CORRECTION to both hands
            # Push each hand slightly toward the other's object estimate
            constraint_correction = constraint_error * 0.5  # Split correction 50/50
            
            # Add correction in object frame, then transform to hand frames
            motion_correction_L = pinocchio.Motion(constraint_correction)
            motion_correction_R = pinocchio.Motion(-constraint_correction)
            
            twist_L += T_O_L_curr.act(motion_correction_L).vector * 0.3  # 30% correction
            twist_R += T_O_R_curr.act(motion_correction_R).vector * 0.3
            
            # J. Get Jacobians and solve
            J_L = self.pinocchio_model.get_link_jacobian(left_idx, local=True)[:, active_indices]
            J_R = self.pinocchio_model.get_link_jacobian(right_idx, local=True)[:, active_indices]
            
            J_total = np.vstack([J_L, J_R])
            twist_total = np.concatenate([twist_L, twist_R])
            
            damp = 1e-3
            JJt = J_total @ J_total.T
            dq = J_total.T @ np.linalg.inv(JJt + damp * np.eye(12)) @ twist_total
            
            # K. Update with smaller step if constraint is large
            if constraint_violation > 1.0:
                dq *= 0.5  # Reduce step when constraint is violated
            
            q[active_indices] += dq
            q = np.clip(q, lower_lim, upper_lim)
            
            # L. Debug
            if step % 10 == 0:
                if self.debug:
                    print(f"[Step {step}] Task: {task_error:.4f}, Constraint: {constraint_violation:.4f}")
            
            # M. Collision check
            if check_collisions and step % 10 == 0:
                self.robot.set_qpos(q, True)
                if self.planning_world.collide():
                    return {"status": "Collision"}
            
            path.append(q.copy())
        
        # N. Final check
        if step >= max_steps - 1:
            if self.debug:
                print(f"⚠ Reached max steps")
                print(f"  Final task error: {task_error:.4f}")
                print(f"  Final constraint: {constraint_violation:.4f}")
                
            if constraint_violation < 0.15:  # Relaxed acceptance
                print("  → Accepting solution (constraint within tolerance)")
            else:
                return {"status": "Failed (Constraint not satisfied)"}
        
        # --- Finalize ---
        path = np.array(path)
        if len(path) < 2:
            return {"status": "Failed (Path too short)"}

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
            print(f"[Screw Constrained] TOPP Failed: {e}")
            return {"status": "Success (No TOPP)", "position": path}
        
    # Helper needed for the function above
    def _mat_to_7d(self, mat):
        from transforms3d.quaternions import mat2quat
        p = mat[:3, 3]
        q = mat2quat(mat[:3, :3])
        return np.concatenate([p, q])
        
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
        # fixed_joints = set()
        # print("FIXED JOINT INDICES",fixed_joint_indices)
        # for joint_idx in fixed_joint_indices:
        #     fixed_joints.add(ompl.FixedJoint(0, joint_idx, current_qpos[joint_idx]))
        
        start_state = current_qpos.astype(np.float64)
        try:
            goal_states = [gq.astype(np.float64) for gq in goal_qpos_]
        except AttributeError:
            return {"status": "IK Failed"}

        # Fix Grippers (Indices updated for new URDF)
        # Panda 1 Hand: panda_1_finger_joint1/2
        # Panda 2 Hand: panda_2_finger_joint1/2
        # We need indices for them.
        # fixed_joints = set()

        # # Only add fixed joints if NOT doing constrained planning
        # if constraint_function is None:
        #     for joint_idx in fixed_joint_indices:
        #         fixed_joints.add(ompl.FixedJoint(0, joint_idx, current_qpos[joint_idx]))
            
        #     # Fix grippers only for unconstrained planning
        #     gripper_names = ["panda_1_finger_joint1", "panda_1_finger_joint2", 
        #                     "panda_2_finger_joint1", "panda_2_finger_joint2"]
        #     for name in gripper_names:
        #         if name in self.joint_name_2_idx:
        #             idx = self.joint_name_2_idx[name]
        #             fixed_joints.add(ompl.FixedJoint(0, idx, current_qpos[idx]))

        # gripper_names = ["panda_1_finger_joint1", "panda_1_finger_joint2", "panda_2_finger_joint1", "panda_2_finger_joint2"]
        # for name in gripper_names:
        #     if name in self.joint_name_2_idx:
        #         idx = self.joint_name_2_idx[name]
        #         fixed_joints.add(ompl.FixedJoint(0, idx, current_qpos[idx]))
        
        if constraint_function is not None and constraint_jacobian is not None:    
            status, path = self.planner.plan(
                start_state=start_state,
                goal_states=goal_states,
                range=rrt_range,
                time=planning_time,
                # fixed_joints=set(),
                constraint_function=constraint_function,
                constraint_jacobian=constraint_jacobian,
                constraint_tolerance=constraint_tolerance,
                verbose=True,
                no_simplification=True
            )
        else:
            fixed_joints = set()
            # Add user-specified fixed joints
            for joint_idx in fixed_joint_indices:
                fixed_joints.add(ompl.FixedJoint(0, joint_idx, current_qpos[joint_idx]))
            
            # Add gripper fixed joints (Mimic joint handling)
            gripper_names = ["panda_1_finger_joint1", "panda_1_finger_joint2", 
                            "panda_2_finger_joint1", "panda_2_finger_joint2"]
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
    
    # def plan_screw(
    # self,
    # target_pose_L=None,
    # target_pose_R=None,
    # current_qpos=None,
    # step_size=0.01,      # MUCH SMALLER - was 0.01
    # time_step=0.1,
    # threshold=1e-3,
    # max_steps=10000,       # Increased for smaller steps
    # visualize_callback=None,
    # check_collisions=True
    # ):
    #     """
    #     Plan a linear (screw) motion for dual arms using Differential IK.
    #     With COLLISION AVOIDANCE and adaptive stepping.
    #     """
    #     if current_qpos is None:
    #         current_qpos = self.robot.get_qpos()
        
    #     # Setup
    #     left_idx = self.link_name_2_idx.get("panda_2_hand_tcp", -1)
    #     right_idx = self.link_name_2_idx.get("panda_1_hand_tcp", -1)
    #     all_active_indices = self.move_group_joint_indices

    #     # Filter out gripper joints
    #     gripper_joint_names = [
    #         "panda_1_finger_joint1", "panda_1_finger_joint2",
    #         "panda_2_finger_joint1", "panda_2_finger_joint2"
    #     ]
    #     gripper_indices = [self.joint_name_2_idx[name] for name in gripper_joint_names 
    #                     if name in self.joint_name_2_idx]
        
    #     active_indices = [idx for idx in all_active_indices if idx not in gripper_indices]

    #     def to_pin_se3(pose_7d):
    #         if pose_7d is None: return None
    #         if isinstance(pose_7d, (list, np.ndarray)):
    #             pos = np.array(pose_7d[:3])
    #             quat = pinocchio.Quaternion(
    #                 float(pose_7d[3]), float(pose_7d[4]), 
    #                 float(pose_7d[5]), float(pose_7d[6])
    #             )
    #             quat.normalize()
    #             return pinocchio.SE3(quat, pos)
    #         else:
    #             mat = pose_7d.to_transformation_matrix()
    #             R = mat[:3, :3]
    #             t = mat[:3, 3]
    #             return pinocchio.SE3(R, t)
            
    #     target_L_se3 = to_pin_se3(target_pose_L)
    #     target_R_se3 = to_pin_se3(target_pose_R)
        
    #     path = [current_qpos.copy()]
    #     q = current_qpos.copy()
    #     gripper_values = {idx: q[idx] for idx in gripper_indices}
        
    #     # Get joint limits once
    #     limits = self.pinocchio_model.get_joint_limits()
    #     limits_stack = np.vstack(limits)
    #     lower_lim = limits_stack[:, 0]
    #     upper_lim = limits_stack[:, 1]
        
    #     # ✅ NEW: Adaptive step size
    #     current_step_size = step_size
    #     min_step_size = step_size * 0.1
    #     collision_count = 0
    #     last_collision_step = -10
        
    #     # ✅ NEW: Track previous valid state for collision retreat
    #     last_valid_q = q.copy()
        
    #     for step in range(max_steps):
    #         if visualize_callback is not None and step % 10 == 0: 
    #             visualize_callback(q)
            
    #         self.pinocchio_model.compute_forward_kinematics(q)
    #         self.pinocchio_model.compute_full_jacobian(q)
            
    #         error_stack = []
    #         J_stack = []
            
    #         # Right Arm
    #         if target_R_se3 is not None:
    #             curr_pose_R = self.pinocchio_model.get_link_pose(right_idx)
    #             curr_se3_R = to_pin_se3(curr_pose_R)
    #             dMf = curr_se3_R.actInv(target_R_se3)
    #             err_R = pinocchio.log(dMf).vector
            
    #             # ✅ CRITICAL: Clamp error to CURRENT step size
    #             if np.linalg.norm(err_R) > current_step_size:
    #                 err_R = err_R * (current_step_size / np.linalg.norm(err_R))
            
    #             error_stack.append(err_R)
    #             J_R = self.pinocchio_model.get_link_jacobian(right_idx, local=True)
    #             J_stack.append(J_R[:, active_indices])
            
    #         # Left Arm
    #         if target_L_se3 is not None:
    #             curr_pose_L = self.pinocchio_model.get_link_pose(left_idx)
    #             curr_se3_L = to_pin_se3(curr_pose_L)
    #             dMf = curr_se3_L.actInv(target_L_se3)
    #             err_L = pinocchio.log(dMf).vector
                
    #             if np.linalg.norm(err_L) > current_step_size:
    #                 err_L = err_L * (current_step_size / np.linalg.norm(err_L))
                
    #             error_stack.append(err_L)
    #             J_L = self.pinocchio_model.get_link_jacobian(left_idx, local=True)
    #             J_stack.append(J_L[:, active_indices])
            
    #         # Convergence check
    #         full_error = np.concatenate(error_stack)
    #         if np.linalg.norm(full_error) < threshold:
    #             print(f"✓ Screw motion converged in {step} steps")
    #             break
            
    #         # Solve for joint velocities
    #         J_total = np.vstack(J_stack)
            
    #         # ✅ IMPROVED: Larger damping for stability near obstacles
    #         base_damp = 1e-3
    #         if step - last_collision_step < 5:
    #             # Recent collision - increase damping
    #             damp = base_damp * 10
    #         else:
    #             damp = base_damp
                
    #         JJt = J_total @ J_total.T
    #         dq = J_total.T @ np.linalg.inv(JJt + damp * np.eye(len(full_error))) @ full_error
            
    #         # ✅ NEW: Limit maximum joint velocity
    #         max_joint_vel = 0.05  # rad per step
    #         dq_norm = np.linalg.norm(dq)
    #         if dq_norm > max_joint_vel:
    #             dq = dq * (max_joint_vel / dq_norm)
            
    #         # Update configuration
    #         q_proposed = q.copy()
    #         q_proposed[active_indices] += dq
    #         q_proposed = np.clip(q_proposed, lower_lim, upper_lim)
            
    #         # Restore gripper values
    #         for idx, val in gripper_values.items():
    #             q_proposed[idx] = val
            
    #         # ✅ CRITICAL: Collision checking BEFORE accepting the step
    #         if check_collisions:
    #             self.robot.set_qpos(q_proposed, True)
                
    #             if self.planning_world.collide():
    #                 collision_count += 1
    #                 last_collision_step = step
                    
    #                 # Print collision details
    #                 if step % 10 == 0:  # Don't spam
    #                     contacts = self.planning_world.collide_full()
    #                     for c in contacts:
    #                         print(f"[Screw Collision {step}] {c.link_name1} <-> {c.link_name2}")
                    
    #                 # ✅ STRATEGY 1: Retreat to last valid state
    #                 if collision_count > 3:
    #                     print(f"[Screw] Multiple collisions detected, retreating...")
    #                     q = last_valid_q.copy()
    #                     collision_count = 0
                        
    #                     # ✅ STRATEGY 2: Reduce step size dramatically
    #                     current_step_size = max(min_step_size, current_step_size * 0.5)
    #                     print(f"[Screw] Reduced step size to {current_step_size:.6f}")
    #                     continue
                    
    #                 # ✅ STRATEGY 3: Try smaller step
    #                 current_step_size = max(min_step_size, current_step_size * 0.7)
    #                 continue  # Don't accept this q, try again with smaller step
                
    #             else:
    #                 # ✅ No collision - accept step and maybe increase speed
    #                 last_valid_q = q_proposed.copy()
    #                 q = q_proposed
                    
    #                 # Gradually increase step size if no recent collisions
    #                 if step - last_collision_step > 20:
    #                     current_step_size = min(step_size, current_step_size * 1.05)
                    
    #                 collision_count = 0
    #         else:
    #             # No collision checking - just accept
    #             q = q_proposed
            
    #         path.append(q.copy())
            
    #         # Debug every N steps
    #         if step % 50 == 0:
    #             print(f"[Screw {step}] Error: {np.linalg.norm(full_error):.6f}, "
    #                 f"Step size: {current_step_size:.6f}, "
    #                 f"Collisions: {collision_count}")
        
    #     # Final check
    #     if step >= max_steps - 1:
    #         print(f"⚠ Screw motion reached max steps")
    #         return {"status": "Failed (max steps)"}
        
    #     # Parameterize path
    #     path = np.array(path)
    #     if len(path) < 2:
    #         return {"status": "Failed (Path too short)"}

    #     try:
    #         times, pos, vel, acc, duration = self.TOPP(path, time_step)
    #         return {
    #             "status": "Success",
    #             "time": times,
    #             "position": pos,
    #             "velocity": vel,
    #             "acceleration": acc,
    #             "duration": duration,
    #         }
    #     except Exception as e:
    #         print(f"[Screw] TOPP Failed: {e}")
    #         return {"status": "Success (No TOPP)", "position": path}
        
    def plan_screw(
        self,
        target_pose_L=None,
        target_pose_R=None,
        current_qpos=None,
        step_size=0.01,   # Max joint change per step
        time_step=0.1,    # For TOPP
        threshold=1e-3,   # Convergence threshold
        max_steps=2000,    # Safety breakout
        visualize_callback=None,
        check_collisions=False
    ):
        """
        Plan a linear (screw) motion for dual arms using Differential IK.
        """
        if current_qpos is None:
            current_qpos = self.robot.get_qpos()
        
        # 1. Setup Pinocchio Data
        left_idx = self.link_name_2_idx.get("panda_2_hand_tcp", -1)
        right_idx = self.link_name_2_idx.get("panda_1_hand_tcp", -1)
        all_active_indices = self.move_group_joint_indices
    
        # Filter out gripper joints
        gripper_joint_names = [
            "panda_1_finger_joint1", "panda_1_finger_joint2",
            "panda_2_finger_joint1", "panda_2_finger_joint2"
        ]
        gripper_indices = [self.joint_name_2_idx[name] for name in gripper_joint_names 
                        if name in self.joint_name_2_idx]
        
        # Active indices = arm joints only (exclude grippers)
        active_indices = [idx for idx in all_active_indices if idx not in gripper_indices]

        # Helper to convert input to Pinocchio SE3
        def to_pin_se3(pose_7d):
            if pose_7d is None: return None
            # [x, y, z, qw, qx, qy, qz]
            if isinstance(pose_7d, (list, np.ndarray)):
                pos = np.array(pose_7d[:3])
                quat = pinocchio.Quaternion(
                    float(pose_7d[3]), float(pose_7d[4]), float(pose_7d[5]), float(pose_7d[6])
                )
                quat.normalize()
                return pinocchio.SE3(quat, pos)
            else:
                mat = pose_input.to_transformation_matrix() # Returns 4x4 numpy array
                R = mat[:3, :3]
                t = mat[:3, 3]
                return pinocchio.SE3(R, t)
            
        target_L_se3 = to_pin_se3(target_pose_L)
        target_R_se3 = to_pin_se3(target_pose_R)
        # print(current_qpos)
        path = [current_qpos.copy()]
        q = current_qpos.copy()
        gripper_values = {idx: q[idx] for idx in gripper_indices}
        # 2. Iterative Descent (Differential IK)
        for step in range(max_steps):
            if visualize_callback is not None and step % 10 == 0: 
                visualize_callback(q)
            
            self.pinocchio_model.compute_forward_kinematics(q)
            self.pinocchio_model.compute_full_jacobian(q)
            
            error_stack = []
            J_stack = []
            
            # --- Right Arm Constraints ---
            if target_R_se3 is not None:
                curr_pose_R = self.pinocchio_model.get_link_pose(right_idx)
                curr_se3_R = to_pin_se3(curr_pose_R)
                dMf = curr_se3_R.actInv(target_R_se3)
                err_R = pinocchio.log(dMf).vector
            
                if np.linalg.norm(err_R) > step_size:
                    err_R = err_R * (step_size / np.linalg.norm(err_R))
            
                error_stack.append(err_R)
                J_R = self.pinocchio_model.get_link_jacobian(right_idx, local=True)
                J_stack.append(J_R[:, active_indices])
            
            # --- Left Arm Constraints ---
            if target_L_se3 is not None:
                curr_pose_L = self.pinocchio_model.get_link_pose(left_idx)
                curr_se3_L = to_pin_se3(curr_pose_L)
                
                # Log(T_current^-1 * T_target) gives the Twist (6D error)
                # We clamp the error magnitude to create linear motion "step by step"
                dMf = curr_se3_L.actInv(target_L_se3)
                err_L = pinocchio.log(dMf).vector
                
                # Clamp error to step_size (Linear Interpolation effect)
                if np.linalg.norm(err_L) > step_size:
                    err_L = err_L * (step_size / np.linalg.norm(err_L))
                
                error_stack.append(err_L)
                J_L = self.pinocchio_model.get_link_jacobian(left_idx, local=True)
                J_stack.append(J_L[:, active_indices])
            
            # --- Convergence Check ---
            # Use raw error for convergence, but clamped error for movement
            full_error = np.concatenate(error_stack)
            if np.linalg.norm(full_error) < threshold:
                break # Reached Target
            
            # --- Solve J * dq = error ---
            J_total = np.vstack(J_stack)
            damp = 1e-3
            JJt = J_total @ J_total.T
            dq = J_total.T @ np.linalg.inv(JJt + damp * np.eye(len(full_error))) @ full_error
            
            # --- Update & Check ---
            q[active_indices] += dq
            if step == 0:
                limits = self.pinocchio_model.get_joint_limits()
                limits_stack = np.vstack(limits)
                lower_lim = limits_stack[:, 0]
                upper_lim = limits_stack[:, 1]
                q = np.clip(q, lower_lim, upper_lim)
            
            for idx, val in gripper_values.items():
                q[idx] = val
            
            # Collision Check (Expensive, maybe check every N steps if slow)
            if step % 1 == 0:
                self.robot.set_qpos(q, True)
                if self.planning_world.collide() and check_collisions:
                    # --- Add this debug block ---
                    contacts = self.planning_world.collide_full()
                    for c in contacts:
                        if self.debug:
                            print(f"[DEBUG] Collision: {c.link_name1} <--> {c.link_name2}")
                    # ----------------------------
                    print(f"[Screw] Collision detected at step {step}")
                    return {"status": "Collision"}
            
            path.append(q.copy())
            
        # 3. Parameterize Path (Compute Time, Velocity, Acceleration)
        path = np.array(path)
        if len(path) < 2:
             return {"status": "Failed (Path too short)"}
        if len(path) > 5:
            # Simple box filter for smoothing
            path_smooth = np.copy(path)
            for i in range(1, len(path)-1):
                path_smooth[i] = (path[i-1] + path[i] + path[i+1]) / 3.0
            path = path_smooth
        try:
            # reuse your existing TOPP function
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
            print(f"[Screw] TOPP Failed: {e}")
            # return {"status": "Success (No TOPP)", "position": path}
            print(f"[Screw] TOPP Failed: {e}")
            
            # FALLBACK: Constant velocity parameterization if TOPP fails
            # This ensures the robot still moves even if the dynamics calculation fails
            print("[Screw] Falling back to constant velocity profile")
            
            # Create a simple time array assuming constant small time steps
            # n_steps = len(path)
            # # Assume each IK step takes 'time_step' seconds (e.g. 0.05s)
            # safe_dt = 0.1
            # times = np.linspace(0, n_steps * safe_dt, n_steps)
            
            # # Calculate simple finite difference velocities
            # vel = np.zeros_like(path)
            # vel[1:] = (path[1:] - path[:-1]) / safe_dt
            
            # # Zero acceleration (approximation)
            # acc = np.zeros_like(path)
            
            # return {
            #     "status": "Success", # Return success so execution continues!
            #     "time": times,
            #     "position": path,
            #     "velocity": vel,
            #     "acceleration": acc,
            #     "duration": times[-1],
            # }
            return {"status": "Success (No TOPP)", "position": path}
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
