import gymnasium as gym
import numpy as np
import sapien
import time
from mani_skill.examples.motionplanning.dual_panda.motionplanner import DualPandaMotionPlanningSolver
from mani_skill.envs.tasks import DualArmCabinetOpenEnv
from scipy.spatial.transform import Rotation as R
import sapien.core as sapien
import sapien.physx as physx  # REQUIRED for SAPIEN 3 Physics classes

def main():
    """
    Test the dual panda motion planner with various scenarios.
    """
    env:DualArmCabinetOpenEnv = gym.make(
        'DualArmCabinetOpen-v0',
        obs_mode='none',
        control_mode="pd_joint_pos",  # Use pd_joint_pos for motion planning
        render_mode='human',  # Use 'human' for visualization
    )
    # debug_collision_properties(env)
    for seed in range(10):  # Test with 3 different seeds
        print(f"\n--- Seed {seed} ---")
        success = solve(env, seed=seed, debug=True, vis=True)            
        print(f"Result: {'Success' if success else 'Failure'}")
        env.reset(seed=seed, options={"reconfigure": True})
    env.close()

def solve(env, seed, debug=False, vis=False):
    env.reset(seed=seed)
    if vis: 
        env.unwrapped.render_human()
    planner = DualPandaMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        print_env_info=True
    )
    # --------------------------------------------------------------------------
    # 1. Identify Target (Right Door Handle)
    # --------------------------------------------------------------------------
    print("1. Identifying Right Handle...")
    
    # Get the cabinet and right door link
    cabinet = env.unwrapped.open_cabinet
    right_door_link = None
    hinge_joint = None
    
    for link in cabinet.articulation.get_links():
        if "hingerightdoor" in link.name:
            right_door_link = link
            break
            
    # Find the hinge joint (usually connected to the door link)
    for joint in cabinet.articulation.get_joints():
        if "rightdoorhinge" in joint.name:
            hinge_joint = joint
            break

    if right_door_link is None:
        print("Error: Could not find right door link.")
        return False

    # Calculate Handle Position (Approximate from site or bounds)
    # Based on XML: Handle is roughly at +0.225 (x), -0.20 (y) in link frame
    # We verify this by taking the center of p1 and p2 sites if available, 
    # or just transforming the known local offset.
    handle_local_offset = np.array([0.225, -0.20, 0.0]) 
    # handle_pose_world = right_door_link.pose.transform(sapien.Pose(p=handle_local_offset))
    try:
        tcp_1_pose = env.unwrapped.agent.tcp_1_pose
        tcp_2_pose = env.unwrapped.agent.tcp_2_pose
        if hasattr(tcp_1_pose.p, 'cpu'):
            # PyTorch tensors
            tcp_1_p = tcp_1_pose.p.cpu().numpy().copy().flatten()
            tcp_1_q = tcp_1_pose.q.cpu().numpy().copy().flatten()
            tcp_2_p = tcp_2_pose.p.cpu().numpy().copy().flatten()
            tcp_2_q = tcp_2_pose.q.cpu().numpy().copy().flatten()
        else:
            # Already numpy
            tcp_1_p = np.array(tcp_1_pose.p).flatten().copy()
            tcp_1_q = np.array(tcp_1_pose.q).flatten().copy()
            tcp_2_p = np.array(tcp_2_pose.p).flatten().copy()
            tcp_2_q = np.array(tcp_2_pose.q).flatten().copy()
        
        print(f"Initial TCP1 (right) p={tcp_1_p}, q={tcp_1_q}")
        print(f"Initial TCP2 (left) p={tcp_2_p}, q={tcp_2_q}")
        
        # Open both grippers
        planner.open_gripper(arm_index=None)
        lift_1 = sapien.Pose(
            p=np.array([-0.0, -0.017, 1.382]),
            q=np.array([0.5,0.5,0.5,0.5])
        )
        
        lift_2 = sapien.Pose(
            p=np.array([0.03, 0.06, 0.89]),
            q=np.array([0, -0.707, 0, -0.707])
        )
        
        result = planner.move_arm_to_pose_with_RRTConnect(
            lift_1,  # left
            arm_index=1
        )
        
        if result == -1:
            print("Failed to lift")
            return False
        # move_2 = lift_2*sapien.Pose(p=[0,0,-0.4])
        # result = planner.move_arm_to_pose_with_RRTConnect(
        #     move_2,
        #     arm_index=2
        # )
        
        # if result == -1:
        #     print("Failed to lift")
        #     return False
        
        # result = planner.move_to_pose_with_screw(
        #     lift_2,
        #     arm_index=2
        # )
        
        # if result == -1:
        #     print("Failed to lift")
        #     return False
        
        planner.close_gripper(arm_index=1)
        # --------------------------------------------------------------------------
        # 3. Magic Grasp (Physics Attachment)
        # --------------------------------------------------------------------------
        print("3. Attaching Handle (Magic Grasp)...")
        
        # Verify this is the correct link index for the Right Hand (Panda Hand)
        # You can print [l.name for l in env.unwrapped.agent.robot.get_links()] to check
        robot_hand_link = env.unwrapped.agent.robot.get_links()[18] 
        
        # Create the Drive (This returns the ManiSkill 'Drive' wrapper you showed me)
        env.magic_drive = env.scene.create_drive(
            robot_hand_link, 
            sapien.Pose(), 
            right_door_link,
            right_door_link.pose.inv() * robot_hand_link.pose
        )
        
        # Define rigid parameters
        stiff = 1e8   # Very strong spring
        damp = 1e6    # High damping to stop oscillation
        
        # --- A. Lock Linear Motion (Supported by Wrapper) ---
        # The wrapper exposes set_limit_x/y/z, so we use them directly.
        env.magic_drive.set_limit_x(0, 0, stiffness=stiff, damping=damp)
        env.magic_drive.set_limit_y(0, 0, stiffness=stiff, damping=damp)
        env.magic_drive.set_limit_z(0, 0, stiffness=stiff, damping=damp)
        
        # --- B. Lock Angular Motion (Bypass Wrapper) ---
        # Since set_twist_limit/set_limit_cone are missing in the wrapper,
        # we iterate over the underlying PhysX objects in ._objs
        for drive in env.magic_drive._objs:
            # Lock Twist (Rotation around X axis)
            if hasattr(drive, "set_limit_twist"):
                drive.set_limit_twist(0, 0, stiffness=stiff, damping=damp)
                
            # Lock Swing (Rotation around Y/Z axes) - SAPIEN usually uses 'Cone' limits
            if hasattr(drive, "set_limit_cone"):
                # limit_cone(angle_y, angle_z, stiffness, damping) -> 0 angle = locked
                drive.set_limit_cone(0, 0, stiffness=stiff, damping=damp)
            
            # Fallback: Some versions use pyramid
            elif hasattr(drive, "set_limit_pyramid"):
                drive.set_limit_pyramid(0, 0, 0, 0, stiffness=stiff, damping=damp)

        print("   Magic Grasp created (Linear+Angular locked).")
        # --------------------------------------------------------------------------
        # 4. Calculate Open Goal (Arc)
        # --------------------------------------------------------------------------
        print("4. Calculating Arc Goal...")
        import torch # Ensure torch is imported

        # Helper function to convert Tensor/Array -> Flat Numpy
        def to_flat_np(data):
            if isinstance(data, torch.Tensor):
                return data.cpu().detach().numpy().flatten()
            return np.array(data).flatten()

        hinge_pose = right_door_link.pose 
        
        # 1. Get current handle pose relative to hinge
        # Note: If .p or .q are tensors, we create a temporary SAP Pose for math
        # Constructing SAPIEN Pose from tensors requires conversion first
        current_p = to_flat_np(lift_1.p)
        current_q = to_flat_np(lift_1.q)
                
        # Create clean SAPIEN poses for math
        T_handle_world_sapien = sapien.Pose(p=current_p, q=current_q)
        
        # Handle hinge pose (might be a ManiSkill Pose wrapper with tensors)
        hinge_p = to_flat_np(hinge_pose.p)
        hinge_q = to_flat_np(hinge_pose.q)
        T_hinge_world_sapien = sapien.Pose(p=hinge_p, q=hinge_q)

        # Calculate offset
        T_handle_in_hinge = T_hinge_world_sapien.inv() * T_handle_world_sapien
        
        # 2. Define Rotation
        open_angle = 1*np.pi/4 # ~45 degrees
        rot_quat = R.from_euler('z', open_angle).as_quat() # [x, y, z, w]
        # SAPIEN Pose expects [w, x, y, z]
        rotation_arc = sapien.Pose(q=[rot_quat[3], rot_quat[0], rot_quat[1], rot_quat[2]]) 
        
        # 3. Calculate Final Goal in World Frame
        T_handle_goal_sapien = T_hinge_world_sapien * rotation_arc * T_handle_in_hinge
        
        # 4. Create the final Goal Pose object (This is now a valid sapien.Pose)
        goal_pose = T_handle_goal_sapien
        
        print(f"   Goal Pose: {goal_pose}")
        # --------------------------------------------------------------------------
        # 5. Plan to Open (RRTConnect)
        # --------------------------------------------------------------------------
        # print(f"5. Planning Arc Motion to {goal_pose_array[:3]}...")
        
        # We use RRTConnect to find a path to the END state of the arc.
        # The Magic Grasp + Physics Engine will handle the constraint guidance 
        # as long as the planner doesn't generate a path that rips the arm off.
        res = planner.move_arm_to_pose_with_RRTConnect(
            goal_pose, 
            arm_index=1, 
            # planner_name="RRTConnect",
            # Optional: reduce step size for finer motion
            # rrt_step=0.05 
        )
        
        if res == -1: 
            print("Failed to plan arc motion.")
            return False
        
        # CLEANUP: Remove the magic grasp
        if hasattr(env, "magic_drive") and env.magic_drive is not None:
            print("Removing Magic Grasp...")
            
            # env.magic_drive._objs contains the SAPIEN PhysxDriveComponents
            for i, sapien_drive_component in enumerate(env.magic_drive._objs):
                # 1. Get the Entity that holds this drive component
                drive_entity = sapien_drive_component.entity
                
                # 2. Remove that Entity from the corresponding SAPIEN scene
                env.scene.sub_scenes[i].remove_entity(drive_entity)
                
            env.magic_drive = None
        print("Success!")
        return True
        
    except Exception as e:
        print(f"✗ Error during synchronized motion test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
# import gymnasium as gym
# import numpy as np
# import sapien
# import time
# from mani_skill.examples.motionplanning.dual_panda.motionplanner import DualPandaMotionPlanningSolver
# from mani_skill.envs.tasks import DualArmCabinetOpenEnv

# def main():
#     """
#     Test the dual panda motion planner with various scenarios.
#     """
#     env: DualArmCabinetOpenEnv = gym.make(
#         'DualArmCabinetOpen-v0',
#         obs_mode='none',
#         control_mode="pd_joint_pos",
#         render_mode='human',
#     )
        
#     for seed in range(3):
#         print(f"\n--- Seed {seed} ---")
#         success = solve(env, seed=seed, debug=True, vis=True)            
#         print(f"Result: {'Success' if success else 'Failure'}")
#         time.sleep(2)
#         env.reset()
#     env.close()

# def solve(env, seed, debug=False, vis=False):
#     env.reset(seed=seed)
#     if vis: 
#         env.unwrapped.render_human()
    
#     planner = DualPandaMotionPlanningSolver(
#         env,
#         debug=debug,
#         vis=vis,
#         print_env_info=True
#     )
    
#     try:
#         # Get cabinet info
#         cabinet = env.unwrapped.open_cabinet
#         joints = cabinet.get_active_joints()
        
#         if len(joints) == 0:
#             print("No movable joints found!")
#             return False
        
#         # Find the revolute joint (door hinge)
#         revolute_joint = None
#         for joint in joints:
#             if joint.type == "revolute":
#                 revolute_joint = joint
#                 break
        
#         if revolute_joint is None:
#             print("No revolute joint found!")
#             return False
        
#         print(f"Found revolute joint: {revolute_joint.name}")
#         print(f"Joint limits: {revolute_joint.get_limit()}")
        
#         # Get the door link (child of the hinge joint)
#         door_link = revolute_joint.get_child_link()
        
#         # Find handle position on the door
#         # The handle is typically on the edge of the door, away from the hinge
#         # We need to find it by inspecting the door's collision shapes
#         handle_pos_local = find_handle_on_door(door_link)
        
#         if handle_pos_local is None:
#             print("Could not find handle position, using default")
#             # Default: assume handle is at edge of door
#             handle_pos_local = np.array([0.3, 0, 0.5])  # Adjust based on your model
        
#         print(f"Handle position (local to door): {handle_pos_local}")
        
#         # Get hinge axis and position
#         hinge_parent_link = revolute_joint.get_parent_link()
#         hinge_pose_in_parent = revolute_joint.get_pose_in_parent()
#         hinge_axis_local = revolute_joint.get_axis()  # Usually [0,0,1] for vertical hinge
        
#         # Transform hinge to world coordinates
#         parent_pose = hinge_parent_link.get_pose()
#         hinge_pose_world = parent_pose * hinge_pose_in_parent
#         hinge_pos_world = hinge_pose_world.p
#         hinge_axis_world = hinge_pose_world.to_transformation_matrix()[:3, :3] @ hinge_axis_local
        
#         print(f"Hinge position (world): {hinge_pos_world}")
#         print(f"Hinge axis (world): {hinge_axis_world}")
        
#         # Calculate handle position in world coordinates
#         door_pose = door_link.get_pose()
#         handle_pos_world = door_pose.transform_point(handle_pos_local)
        
#         print(f"Handle position (world): {handle_pos_world}")
        
#         # Open grippers
#         planner.open_gripper(arm_index=None)
#         time.sleep(0.5)
        
#         # Calculate approach pose for RIGHT arm (arm_index=1)
#         # Approach from the side of the handle
#         approach_offset = np.array([0, -0.1, 0])  # Approach from -Y direction
#         approach_pos = handle_pos_world + approach_offset
        
#         # Gripper orientation: fingers pointing toward handle
#         # For a vertical door, this is typically pointing in +Y direction
#         approach_quat = np.array([0, 0.707, 0, 0.707])  # Adjust based on your setup
        
#         approach_pose = sapien.Pose(p=approach_pos, q=approach_quat)
        
#         print(f"Moving RIGHT arm to approach pose: {approach_pos}")
#         result = planner.move_arm_to_pose_with_RRTConnect(
#             approach_pose,
#             arm_index=1
#         )
        
#         if result == -1:
#             print("Failed to reach approach pose")
#             return False
        
#         time.sleep(0.5)
        
#         # Move to grasp pose (closer to handle)
#         grasp_pos = handle_pos_world + np.array([0, -0.02, 0])  # Close to handle
#         grasp_pose = sapien.Pose(p=grasp_pos, q=approach_quat)
        
#         print(f"Moving RIGHT arm to grasp pose: {grasp_pos}")
#         result = planner.move_to_pose_with_screw(
#             grasp_pose,
#             arm_index=1
#         )
        
#         if result == -1:
#             print("Failed to reach grasp pose")
#             return False
        
#         time.sleep(0.5)
        
#         # Close gripper to grasp handle
#         print("Closing gripper...")
#         planner.close_gripper(arm_index=1)
#         time.sleep(1.0)
        
#         # Verify grasp by checking gripper joint positions
#         gripper_qpos = env.unwrapped.agent.robot.get_qpos()[7:9]  # Right gripper joints
#         print(f"Gripper qpos after closing: {gripper_qpos}")
        
#         if np.all(gripper_qpos < 0.01):  # Gripper fully closed = no grasp
#             print("Failed to grasp handle (gripper fully closed)")
#             return False
        
#         print(" Grasp successful!")
        
#         # Now pull the door open by following a circular arc
#         print("\nPulling door open...")
        
#         initial_joint_pos = revolute_joint.get_drive_target()
#         joint_limits = revolute_joint.get_limit()
#         target_joint_pos = min(initial_joint_pos + np.pi/3, joint_limits[0, 1])  # Open 60 degrees or max
        
#         num_steps = 30
#         for step in range(num_steps):
#             # Interpolate joint angle
#             alpha = (step + 1) / num_steps
#             current_joint_angle = initial_joint_pos + alpha * (target_joint_pos - initial_joint_pos)
            
#             # Calculate new handle position following circular arc
#             # Handle rotates around hinge axis
#             new_handle_pos = rotate_point_around_axis(
#                 handle_pos_world,
#                 hinge_pos_world,
#                 hinge_axis_world,
#                 current_joint_angle - initial_joint_pos
#             )
            
#             # Calculate new gripper orientation (tangent to the arc)
#             # Gripper should rotate with the door
#             tangent_direction = np.cross(hinge_axis_world, new_handle_pos - hinge_pos_world)
#             tangent_direction = tangent_direction / np.linalg.norm(tangent_direction)
            
#             # Create new pose
#             # Keep gripper perpendicular to the door surface
#             new_quat = calculate_gripper_orientation(hinge_axis_world, tangent_direction)
            
#             target_pose = sapien.Pose(p=new_handle_pos, q=new_quat)
            
#             # Move to new pose with screw motion (small incremental motion)
#             result = planner.move_to_pose_with_screw(
#                 target_pose,
#                 arm_index=1
#             )
            
#             if result == -1:
#                 print(f"Failed at step {step}/{num_steps}")
#                 # Try to continue anyway
#                 continue
            
#             # Small delay for visualization
#             if vis:
#                 time.sleep(0.05)
            
#             # Print progress
#             if step % 5 == 0:
#                 current_angle_deg = np.degrees(current_joint_angle - initial_joint_pos)
#                 print(f"Step {step}/{num_steps}, Door opened: {current_angle_deg:.1f}°")
        
#         # Check final door state
#         final_joint_pos = revolute_joint.get_drive_target()
#         final_angle_deg = np.degrees(final_joint_pos - initial_joint_pos)
#         print(f"\nFinal door angle: {final_angle_deg:.1f}°")
        
#         # Release gripper
#         planner.open_gripper(arm_index=1)
#         time.sleep(0.5)
        
#         # Consider success if door opened at least 20 degrees
#         success = final_angle_deg > 20.0
        
#         if success:
#             print(" Successfully opened door!")
#         else:
#             print(f"Door only opened {final_angle_deg:.1f}°")
        
#         return success
        
#     except Exception as e:
#         print(f"Error: {e}")
#         import traceback
#         traceback.print_exc()
#         return False

# def find_handle_on_door(door_link):
#     """
#     Try to find the handle position on the door by inspecting collision shapes.
#     Returns position in door's local frame.
#     """
#     collision_shapes = door_link.get_collision_shapes()
    
#     if len(collision_shapes) == 0:
#         return None
    
#     # Simple heuristic: find the shape furthest from origin in X direction
#     # (assuming handle is on the edge of door away from hinge)
#     max_x = -np.inf
#     handle_pos = None
    
#     for shape in collision_shapes:
#         pose = shape.get_local_pose()
#         if pose.p[0] > max_x:
#             max_x = pose.p[0]
#             handle_pos = pose.p
    
#     return handle_pos

# def rotate_point_around_axis(point, axis_point, axis_direction, angle):
#     """
#     Rotate a point around an axis by a given angle.
    
#     Args:
#         point: Point to rotate (3D)
#         axis_point: A point on the rotation axis
#         axis_direction: Direction vector of the axis (will be normalized)
#         angle: Rotation angle in radians
    
#     Returns:
#         Rotated point
#     """
#     # Normalize axis
#     axis = axis_direction / np.linalg.norm(axis_direction)
    
#     # Translate point to origin
#     p = point - axis_point
    
#     # Rodrigues' rotation formula
#     cos_angle = np.cos(angle)
#     sin_angle = np.sin(angle)
    
#     rotated = (p * cos_angle + 
#                np.cross(axis, p) * sin_angle + 
#                axis * np.dot(axis, p) * (1 - cos_angle))
    
#     # Translate back
#     return rotated + axis_point

# def calculate_gripper_orientation(hinge_axis, tangent_direction):
#     """
#     Calculate gripper orientation based on hinge axis and tangent direction.
    
#     Args:
#         hinge_axis: The axis of the door hinge (typically [0,0,1])
#         tangent_direction: Direction tangent to the circular arc
    
#     Returns:
#         Quaternion for gripper orientation
#     """
#     # Gripper Z-axis should point in tangent direction (direction of pull)
#     z_axis = tangent_direction / np.linalg.norm(tangent_direction)
    
#     # Gripper Y-axis should be perpendicular to hinge axis
#     y_axis = np.cross(hinge_axis, z_axis)
#     y_axis = y_axis / np.linalg.norm(y_axis)
    
#     # Gripper X-axis completes the right-handed frame
#     x_axis = np.cross(y_axis, z_axis)
    
#     # Construct rotation matrix
#     rot_mat = np.column_stack([x_axis, y_axis, z_axis])
    
#     # Convert to quaternion
#     from scipy.spatial.transform import Rotation
#     quat = Rotation.from_matrix(rot_mat).as_quat()  # [x, y, z, w]
    
#     # SAPIEN uses [w, x, y, z] format
#     return np.array([quat[3], quat[0], quat[1], quat[2]])

# if __name__ == "__main__":
#     main()