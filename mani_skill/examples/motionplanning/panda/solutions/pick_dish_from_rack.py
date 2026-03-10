import numpy as np
import sapien
import gymnasium as gym
from mani_skill.envs.tasks.tabletop.colosseum_v2.pick_dish_from_rack import PickDishFromRackEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver


def main():
    env = gym.make(
        "PickDishFromRack-v1",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="none",
        _env_id="PickDishFromRack-v1",
        )
    for seed in range(100):
        res = solve(env, seed=seed, debug=True, vis=True)
        print(res)
    env.close()

# Quaternion multiply: q_new = q * rot_90 (note scipy uses (x,y,z,w) but sapien uses (w,x,y,z))
def quat_multiply(q1, q2):
    # q format: (w, x, y, z)
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])



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
    grasp_center = plate_pos.copy()
    grasp_center[1] += 0.025
    grasp_center[2] = plate_pos[2] + plate_outer_radius - 0.05

    # Get current EE pose and orientation
    current_tcp = env_sim.agent.tcp_pose.sp
    current_pos = np.array(current_tcp.p)
    quat_0 = np.array(current_tcp.q)

    # Rotate +90deg about z (yaw)
    # Quaternion for +90deg about z: [cos(theta/2), 0, 0, sin(theta/2)] where theta=pi/2
    # In (w,x,y,z) order: [sqrt(0.5), 0, 0, sqrt(0.5)]
    rot_90_about_z = np.array([np.sqrt(0.5), 0, 0, np.sqrt(0.5)])
    grasping_quat = quat_multiply(quat_0, rot_90_about_z)

    if debug:
        print(f"Current EE position: {current_pos}")

    # -------------------------------------------------------------------------- #
    # Step 1: Move laterally to be above the plate (keep current orientation)
    # -------------------------------------------------------------------------- #
    if debug:
        print(f"\n=== STEP 1: MOVE LATERALLY ABOVE PLATE ===")

    # Move to same X, Y as grasp center, keep current Z and orientation
    lateral_pos = np.array([grasp_center[0], grasp_center[1], current_pos[2]])
    lateral_pose = sapien.Pose(lateral_pos, grasping_quat)
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
    # if debug:
    #     print(f"\n=== STEP 2: DESCEND TO REACH POSITION ===")

    # # Get updated position after lateral move
    # current_tcp = env_sim.agent.tcp_pose.sp
    # current_pos = np.array(current_tcp.p)
    # current_quat = np.array(current_tcp.q)

    # # Descend to 15cm above grasp center
    # reach_pos = np.array([grasp_center[0], grasp_center[1], grasp_center[2] + 0.15])
    # reach_pose = sapien.Pose(reach_pos, current_quat)
    # result = planner.move_to_pose_with_screw(reach_pose)
    # if result == -1:
    #     if debug:
    #         print("Failed to descend to reach position")
    #     planner.close()
    #     return result

    # if debug:
    #     print("Descended to reach position")

    # -------------------------------------------------------------------------- #
    # Step 3: Descend to grasp position (keep current orientation)
    # -------------------------------------------------------------------------- #
    if debug:
        print(f"\n=== STEP 3: DESCEND TO GRASP ===")

    grasp_pose = sapien.Pose(grasp_center, np.array(env_sim.agent.tcp_pose.sp.q))
    result = planner.move_to_pose_with_screw(grasp_pose)
    if result == -1:
        if debug:
            print("Failed to descend to grasp")
        planner.close()
        return result

    # Close gripper with maximum force
    planner.close_gripper(t=4, gripper_state=-1.0)

    # Let physics settle after grasping
    # qpos = env_sim.agent.robot.get_qpos()[0, : len(planner.planner.joint_vel_limits)].cpu().numpy()
    # for i in range(20):
    #     if planner.control_mode == "pd_joint_pos":
    #         action = np.hstack([qpos, -1.0])
    #     else:
    #         action = np.hstack([qpos, qpos * 0, -1.0])
    #     env_sim.step(action)

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
    # if debug:
    #     print(f"\n=== STEP 5: MOVE AWAY FROM RACK ===")

    # current_tcp = env_sim.agent.tcp_pose.sp
    # current_pos = np.array(current_tcp.p)
    # current_quat = np.array(current_tcp.q)

    # # Move toward center Y and forward
    # target_pos = np.array([-0.2, 0.0, current_pos[2]])
    # target_pose = sapien.Pose(target_pos, current_quat)
    # res = planner.move_to_pose_with_screw(target_pose)
    # if res == -1:
    #     if debug:
    #         print("Failed to move away from rack")
    #     planner.close()
    #     return res

    # if debug:
    #     plate_after_move = env_sim.plate.pose.p[0].cpu().numpy()
    #     print(f"Moved plate to: {plate_after_move}")

    # -------------------------------------------------------------------------- #
    # Wait for physics to settle
    # -------------------------------------------------------------------------- #
    # arm_dof = len(env_sim.agent.arm_joint_names)
    # home_qpos = env_sim.agent.robot.get_qpos()[0, :arm_dof].cpu().numpy()
    # for _ in range(60):
    #     if env_sim.control_mode == "pd_joint_pos":
    #         action = np.hstack([home_qpos, planner.gripper_state])
    #     else:
    #         action = np.hstack([home_qpos, np.zeros(arm_dof), planner.gripper_state])
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     if vis:
    #         env_sim.render_human()

    planner.close()
    return res


if __name__ == "__main__":
    main()