import numpy as np
import sapien

from mani_skill.envs.tasks.tabletop.pick_dish_from_rack import PickDishFromRackEnv
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)


def solve(env: PickDishFromRackEnv, seed=None, debug=False, vis=False):
    """Pick plate from rack and place it flat on the table."""
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode

    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    env_sim = env.unwrapped

    # Get plate position and orientation (plate is vertical in rack)
    plate_pose = env_sim.plate.pose
    plate_pos = plate_pose.p[0].cpu().numpy()

    if debug:
        print(f"\n=== PLATE POSITION IN RACK ===")
        print(f"Plate position: {plate_pos}")

    # Plate geometry parameters
    plate_outer_radius = env_sim._plate_outer_radius

    # Grasp at the top of the vertical plate
    grasp_center = plate_pos.copy()
    grasp_center[2] = plate_pos[2] + plate_outer_radius * 0.8  # Near the top rim

    # Get current EE pose and orientation
    current_tcp = env_sim.agent.tcp_pose.sp
    current_pos = np.array(current_tcp.p)
    current_quat = np.array(current_tcp.q)

    if debug:
        print(f"Current EE position: {current_pos}")
        print(f"Grasp center: {grasp_center}")

    # -------------------------------------------------------------------------- #
    # Step 1: Move laterally to be above the plate (keep current orientation)
    # -------------------------------------------------------------------------- #
    if debug:
        print(f"\n=== STEP 1: MOVE LATERALLY ABOVE PLATE ===")

    # Move to same X, Y as grasp center, keep current Z and orientation
    lateral_pos = np.array([grasp_center[0], grasp_center[1], current_pos[2]+ 0.2])
    lateral_pose = sapien.Pose(lateral_pos, current_quat)
    result = planner.move_to_pose_with_screw(lateral_pose)
    if result == -1:
        if debug:
            print("Failed to move laterally")
        planner.close()
        return result

    if debug:
        print("Moved laterally above plate")

    # -------------------------------------------------------------------------- #
    # Step 2: Descend to reach position (keep current orientation)
    # -------------------------------------------------------------------------- #
    if debug:
        print(f"\n=== STEP 2: DESCEND TO REACH POSITION ===")

    # Get updated position after lateral move
    current_tcp = env_sim.agent.tcp_pose.sp
    current_pos = np.array(current_tcp.p)
    current_quat = np.array(current_tcp.q)

    # Descend to 15cm above grasp center
    reach_pos = np.array([grasp_center[0], grasp_center[1], grasp_center[2] + 0.15])
    reach_pose = sapien.Pose(reach_pos, current_quat)
    result = planner.move_to_pose_with_screw(reach_pose)
    if result == -1:
        if debug:
            print("Failed to descend to reach position")
        planner.close()
        return result

    if debug:
        print("Descended to reach position")

    # -------------------------------------------------------------------------- #
    # Step 3: Descend to grasp position (keep current orientation)
    # -------------------------------------------------------------------------- #
    if debug:
        print(f"\n=== STEP 3: DESCEND TO GRASP ===")

    grasp_pose = sapien.Pose(grasp_center, current_quat)
    result = planner.move_to_pose_with_screw(grasp_pose)
    if result == -1:
        if debug:
            print("Failed to descend to grasp")
        planner.close()
        return result

    # Close gripper with maximum force
    planner.close_gripper(t=40, gripper_state=-1.0)

    # Let physics settle after grasping
    qpos = env_sim.agent.robot.get_qpos()[0, : len(planner.planner.joint_vel_limits)].cpu().numpy()
    for i in range(20):
        if planner.control_mode == "pd_joint_pos":
            action = np.hstack([qpos, -1.0])
        else:
            action = np.hstack([qpos, qpos * 0, -1.0])
        env_sim.step(action)

    is_grasped = env_sim.agent.is_grasping(env_sim.plate)[0].item()

    if debug:
        gripper_qpos = env_sim.agent.robot.get_qpos()[0, 7:9].cpu().numpy()
        print(f"Gripper closed: {gripper_qpos}")
        print(f"Is grasped: {is_grasped}")

    # -------------------------------------------------------------------------- #
    # Step 4: Pull out of rack (lift straight up, keep orientation)
    # -------------------------------------------------------------------------- #
    if debug:
        print(f"\n=== STEP 4: PULL OUT ===")

    current_tcp = env_sim.agent.tcp_pose.sp
    current_pos = np.array(current_tcp.p)
    current_quat = np.array(current_tcp.q)

    pullout_pos = np.array([current_pos[0], current_pos[1], current_pos[2] + 0.20])
    pullout_pose = sapien.Pose(pullout_pos, current_quat)
    res = planner.move_to_pose_with_screw(pullout_pose)
    if res == -1:
        if debug:
            print("Failed to pull out")
        planner.close()
        return res

    if debug:
        plate_after_pull = env_sim.plate.pose.p[0].cpu().numpy()
        print(f"Pulled plate out to: {plate_after_pull}")

    # -------------------------------------------------------------------------- #
    # Step 5: Move away from rack (keep orientation)
    # -------------------------------------------------------------------------- #
    if debug:
        print(f"\n=== STEP 5: MOVE AWAY FROM RACK ===")

    current_tcp = env_sim.agent.tcp_pose.sp
    current_pos = np.array(current_tcp.p)
    current_quat = np.array(current_tcp.q)

    # Move toward center Y and forward
    target_pos = np.array([-0.2, 0.0, current_pos[2]])
    target_pose = sapien.Pose(target_pos, current_quat)
    res = planner.move_to_pose_with_screw(target_pose)
    if res == -1:
        if debug:
            print("Failed to move away from rack")
        planner.close()
        return res

    if debug:
        plate_after_move = env_sim.plate.pose.p[0].cpu().numpy()
        print(f"Moved plate to: {plate_after_move}")

    # -------------------------------------------------------------------------- #
    # Wait for physics to settle
    # -------------------------------------------------------------------------- #
    arm_dof = len(env_sim.agent.arm_joint_names)
    home_qpos = env_sim.agent.robot.get_qpos()[0, :arm_dof].cpu().numpy()
    for _ in range(60):
        if env_sim.control_mode == "pd_joint_pos":
            action = np.hstack([home_qpos, planner.gripper_state])
        else:
            action = np.hstack([home_qpos, np.zeros(arm_dof), planner.gripper_state])
        obs, reward, terminated, truncated, info = env.step(action)
        if vis:
            env_sim.render_human()

    planner.close()
    return obs, reward, terminated, truncated, info
