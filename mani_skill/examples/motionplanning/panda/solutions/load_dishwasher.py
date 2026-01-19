import numpy as np

from mani_skill.envs.tasks.tabletop.load_dishwasher import LoadDishwasherEnv
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)


def solve(env: LoadDishwasherEnv, seed=None, debug=False, vis=False):
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

    # Get drawer link position as reference
    drawer_link = env_sim.dishwasher._objs[0].links[2]
    drawer_pos = np.array(drawer_link.pose.p)

    # Grasp orientation: approach from top, fingers close front-to-back
    approaching = np.array([0, 0, -1])  # approach from above
    closing = np.array([1, 0, 0])        # fingers close in X direction

    # Pre-grasp position: above the lip
    pregrasp_pos = drawer_pos.copy()
    pregrasp_pos[0] -= 0.015
    pregrasp_pos[1] -= 0.03  # at the lip edge
    pregrasp_pos[2] += 0.25  # high above
    pregrasp_pose = env_sim.agent.build_grasp_pose(approaching, closing, pregrasp_pos)

    # Grasp position: relative to pre-grasp, move down to grip the lip
    grasp_pos = pregrasp_pos.copy()
    grasp_pos[2] -= 0.04  # move down 2cm from pre-grasp
    grasp_pose = env_sim.agent.build_grasp_pose(approaching, closing, grasp_pos)

    if debug:
        print(f"Drawer pos: {drawer_pos}")
        print(f"Pregrasp pos: {pregrasp_pos}")
        print(f"Grasp pos: {grasp_pos}")
        print(f"Distance (Z): {pregrasp_pos[2] - grasp_pos[2]:.2f}m")

    # Step 1: Move to pre-grasp position
    planner.open_gripper(t=30)
    res = planner.move_to_pose_with_RRTConnect(pregrasp_pose)
    if res == -1:
        planner.close()
        return res

    if debug:
        print("Step 1: At pre-grasp position")

    # Step 2: Move forward so open gripper wraps around the lip
    planner.open_gripper(t=10)
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1:
        planner.close()
        return res

    if debug:
        print("Step 2: Gripper wrapped around lip")

    # Step 3: Close gripper hard
    planner.close_gripper(t=30, gripper_state=-1.0)

    if debug:
        print("Step 3: Gripper closed")

    # Step 4: Pull back toward robot base (-X direction)
    pull_pos = grasp_pos.copy()
    pull_pos[0] -= 0.20  # pull back 20cm toward robot
    pull_pose = env_sim.agent.build_grasp_pose(approaching, closing, pull_pos)

    res = planner.move_to_pose_with_screw(pull_pose)
    if res == -1:
        if debug:
            print("Pull failed")
        planner.close()
        return res

    if debug:
        print("Step 4: Pulled back")

    # Hold position
    robot_qpos = env_sim.agent.robot.get_qpos()[0, : len(planner.planner.joint_vel_limits)].cpu().numpy()
    for _ in range(60):
        if planner.control_mode == "pd_joint_pos":
            action = np.hstack([robot_qpos, -1.0])
        else:
            action = np.hstack([robot_qpos, robot_qpos * 0, -1.0])
        env_sim.step(action)

    planner.close()
    return res
