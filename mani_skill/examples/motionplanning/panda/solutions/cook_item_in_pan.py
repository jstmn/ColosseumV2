import numpy as np
import sapien

from mani_skill.envs.tasks.tabletop.cook_item_in_pan import CookItemInPanEnv
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)


def solve(env: CookItemInPanEnv, seed=None, debug=False, vis=False):
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
    arm_dof = len(env_sim.agent.arm_joint_names)
    home_arm_qpos = (
        env_sim.agent.robot.get_qpos()[0, :arm_dof].cpu().numpy()
    )
    home_arm_qvel = np.zeros(arm_dof)

    def move_arm_to_qpos(target_qpos, steps=60):
        for _ in range(steps):
            if env_sim.control_mode == "pd_joint_pos":
                action = np.hstack([target_qpos, planner.gripper_state])
            else:
                action = np.hstack([target_qpos, home_arm_qvel, planner.gripper_state])
            env.step(action)
            if vis:
                env_sim.render_human()

    def move_to_pose(target_pose, use_screw=False, max_retries=3):
        for attempt in range(max_retries):
            if use_screw:
                res = planner.move_to_pose_with_screw(target_pose)
            else:
                res = planner.move_to_pose_with_RRTConnect(target_pose)
            if res != -1:
                return res
            # Retry with screw as fallback on RRT failure
            if not use_screw and attempt < max_retries - 1:
                print(f"RRT failed, retrying with screw motion (attempt {attempt + 2})")
                res = planner.move_to_pose_with_screw(target_pose)
                if res != -1:
                    return res
        return -1

    # Get positions at start
    pan_pos = env_sim.pan.pose.p[0].cpu().numpy()
    apple_pos_initial = env_sim.food.pose.p[0].cpu().numpy()
    print(f"Pan pos: {pan_pos}")
    print(f"Apple pos (initial): {apple_pos_initial}")

    # Top-down grasp on the rim - approach from above, close along Y
    approaching = np.array([0.0, 0.0, -1.0])
    closing = np.array([0.0, 1.0, 0.0])

    # The mesh is not centered - account for mesh offset after 90deg rotation
    # After rotation: local X->world Y, local Y->world -X
    # Mesh body center is at roughly [-0.10, 0.35] relative to spawn
    # Target the near rim for easy grasp
    rim_pos = np.array([-0.10, 0.17, 0.15])  # near rim, above

    # Robot already starts at pregrasp pose (above rim), skip first motion

    # Lower straight down to grasp the rim
    grasp_pos = rim_pos.copy()
    grasp_pos[2] = 0.01  # rim height
    grasp_pose = env_sim.agent.build_grasp_pose(approaching, closing, grasp_pos)
    res = move_to_pose(grasp_pose)
    if res == -1:
        planner.close()
        return res

    # Close gripper
    planner.close_gripper()

    # Lift up
    lift_pos = grasp_pos.copy()
    lift_pos[2] = 0.30  # lift to 30cm
    lift_pose = env_sim.agent.build_grasp_pose(approaching, closing, lift_pos)
    res = move_to_pose(lift_pose)
    if res == -1:
        planner.close()
        return res

    # Check if grasping
    is_grasping = env_sim.agent.is_grasping(env_sim.pan)
    print(f"Is grasping pan: {is_grasping}")

    # Move to above stove
    stove_pos = env_sim.stove.pose.p[0].cpu().numpy()
    print(f"Stove pos: {stove_pos}")

    above_stove = lift_pos.copy()
    above_stove[0] = stove_pos[0]  # center on stove X
    above_stove[1] = stove_pos[1] - 0.2  # center on stove Y
    above_stove[2] = 0.25  # keep high
    stove_pose = env_sim.agent.build_grasp_pose(approaching, closing, above_stove)
    res = move_to_pose(stove_pose)
    if res == -1:
        planner.close()
        return res

    # Lower pan onto stove
    lower_pos = above_stove.copy()
    lower_pos[2] = stove_pos[2] + env_sim.stove_top_offset + 0.03
    lower_pose = env_sim.agent.build_grasp_pose(approaching, closing, lower_pos)
    res = move_to_pose(lower_pose)
    if res == -1:
        planner.close()
        return res

    # Release pan
    planner.open_gripper()

    # Lift up after releasing pan
    retreat_pos = lower_pos.copy()
    retreat_pos[2] = 0.25
    retreat_pose = env_sim.agent.build_grasp_pose(approaching, closing, retreat_pos)
    res = move_to_pose(retreat_pose)
    if res == -1:
        planner.close()
        return res

    # Use initial apple position (saved at start)
    apple_pos = apple_pos_initial.copy()
    print(f"Apple pos (using initial): {apple_pos}")

    # Move above apple
    above_apple = apple_pos.copy()
    above_apple[2] = 0.15
    above_apple_pose = env_sim.agent.build_grasp_pose(approaching, closing, above_apple)
    res = move_to_pose(above_apple_pose)
    if res == -1:
        planner.close()
        return res

    # Lower to grasp apple
    grasp_apple_pos = apple_pos.copy()
    grasp_apple_pos[2] = apple_pos[2] + 0.02
    grasp_apple_pose = env_sim.agent.build_grasp_pose(approaching, closing, grasp_apple_pos)
    res = move_to_pose(grasp_apple_pose)
    if res == -1:
        planner.close()
        return res

    # Grasp apple
    planner.close_gripper()

    # Lift apple
    lift_apple_pos = grasp_apple_pos.copy()
    lift_apple_pos[2] = 0.25
    lift_apple_pose = env_sim.agent.build_grasp_pose(approaching, closing, lift_apple_pos)
    res = move_to_pose(lift_apple_pose)
    if res == -1:
        planner.close()
        return res

    # Move apple above pan (now on stove)
    pan_on_stove = env_sim.pan.pose.p[0].cpu().numpy()
    above_pan_pos = lower_pos.copy()
    above_pan_pos[1] = lower_pos[1] + 0.1
    above_pan_pos[2] = lower_pos[2] + 0.15  # above pan
    above_pan_pose = env_sim.agent.build_grasp_pose(approaching, closing, above_pan_pos)
    res = move_to_pose(above_pan_pose)
    if res == -1:
        planner.close()
        return res

    # Lower apple into pan - use higher z to avoid collision with pan rim
    place_apple_pos = above_pan_pos.copy()
    place_apple_pos[2] = 0.12  # higher to clear pan rim
    place_apple_pose = env_sim.agent.build_grasp_pose(approaching, closing, place_apple_pos)
    res = move_to_pose(place_apple_pose)
    if res == -1:
        planner.close()
        return res

    # Release apple
    planner.open_gripper()

    # Rise straight up using screw motion (smoother than RRT)
    rise_up = place_apple_pos.copy()
    rise_up[2] += 0.15
    rise_pose = env_sim.agent.build_grasp_pose(approaching, closing, rise_up)
    res = move_to_pose(rise_pose, use_screw=True)
    if res == -1:
        planner.close()
        return res

    # Stay at rise position - no return to home to avoid jerky motion
    planner.open_gripper()

    # Get current qpos to hold position during settling
    current_arm_qpos = env_sim.agent.robot.get_qpos()[0, :arm_dof].cpu().numpy()

    # Wait for objects to settle (hold current position)
    for _ in range(120):
        if env_sim.control_mode == "pd_joint_pos":
            action = np.hstack([current_arm_qpos, planner.gripper_state])
        else:
            action = np.hstack([current_arm_qpos, home_arm_qvel, planner.gripper_state])
        obs, reward, terminated, truncated, info = env.step(action)
        if vis:
            env_sim.render_human()

    planner.close()
    return obs, reward, terminated, truncated, info
