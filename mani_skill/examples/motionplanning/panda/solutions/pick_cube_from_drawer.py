import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks.tabletop.colosseum_v2_versions.pick_cube_from_drawer import PickCubeFromDrawerEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver


def solve(env: PickCubeFromDrawerEnv, seed=None, debug=False, vis=False):
    """
    Solution for PickCubeFromDrawer task.
    Opens the drawer fully.
    """
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
        joint_vel_limits=1.5,
        joint_acc_limits=1.5,
    )

    env_inner = env.unwrapped
    handle_obj = env_inner.handle_link_goal

    if debug:
        print("Opening drawer...")

    handle_pos = handle_obj.pose.p.cpu().numpy()[0]

    if debug:
        print(f"Handle position: {handle_pos}")

    planner.open_gripper()

    # Gripper faces -Y, close on Z, pull in +Y
    handle_approaching = np.array([0, -1, 0], dtype=np.float32)
    handle_closing = np.array([0, 0, -1], dtype=np.float32)

    grasp_pose = env_inner.agent.build_grasp_pose(handle_approaching, handle_closing, handle_pos)

    # Move to pre-grasp (5cm back from handle, in +Y direction)
    pre_grasp_pos = handle_pos.copy()
    pre_grasp_pos[1] += 0.05
    pre_grasp_pose = env_inner.agent.build_grasp_pose(handle_approaching, handle_closing, pre_grasp_pos)

    res = planner.move_to_pose_with_screw(pre_grasp_pose)
    if res == -1:
        res = planner.move_to_pose_with_RRTStar(pre_grasp_pose)
        if res == -1:
            if debug:
                print("Failed to reach pre-grasp pose")
            planner.close()
            return -1

    # Move forward to grasp handle
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1:
        planner.move_to_pose_with_RRTStar(grasp_pose)

    planner.close_gripper()

    # Get drawer joint limits
    qlimits = env_inner.handle_link.joint.limits
    qmin, qmax = qlimits[0, 0].item(), qlimits[0, 1].item()
    target_qpos = qmax  # Target fully open

    # Set drive target to open the drawer fully
    env_inner.handle_link.joint.set_drive_target(np.array([target_qpos]))

    # Pull drawer in +Y direction
    current_pos = grasp_pose.p.copy()
    step_size = 0.15  # 15cm steps
    max_steps = 3  # Up to 45cm total

    for i in range(max_steps):
        current_pos[1] += step_size
        pull_pose = env_inner.agent.build_grasp_pose(handle_approaching, handle_closing, current_pos)

        res = planner.move_to_pose_with_screw(pull_pose)
        if res == -1:
            if debug:
                print(f"Pull step {i+1} failed at Y={current_pos[1]:.3f}")
            planner.close()
            return -1

        # Check if drawer is fully open (90%+)
        qpos = env_inner.handle_link.joint.qpos[0].item()
        pct = (qpos - qmin) / (qmax - qmin) * 100
        if debug:
            print(f"Drawer open: {pct:.1f}%")
        if pct >= 90:
            break

    planner.open_gripper()

    if debug:
        qpos = env_inner.handle_link.joint.qpos[0].item()
        pct = (qpos - qmin) / (qmax - qmin) * 100
        print(f"Drawer opened: {pct:.1f}%")

    # Retreat from handle (move back in +Y)
    tcp_pos = env_inner.agent.tcp.pose.p.cpu().numpy()[0]
    retreat_pos = tcp_pos.copy()
    retreat_pos[1] += 0.10  # 10cm back
    retreat_pose = env_inner.agent.build_grasp_pose(handle_approaching, handle_closing, retreat_pos)
    planner.move_to_pose_with_screw(retreat_pose)

    # Lift to 20cm height
    lift_pos = retreat_pos.copy()
    lift_pos[2] = 0.70  # 20cm height
    lift_pose = env_inner.agent.build_grasp_pose(handle_approaching, handle_closing, lift_pos)
    res = planner.move_to_pose_with_screw(lift_pose)
    if res == -1:
        planner.move_to_pose_with_RRTStar(lift_pose)

    # Get cube position and hover above it
    cube = env_inner.cube
    cube_pos = cube.pose.p[0].cpu().numpy()

    if debug:
        print(f"Cube position: {cube_pos}")

    # Hover above cube at same XY, keeping 20cm height
    hover_pos = np.array([cube_pos[0], cube_pos[1], 0.40])
    # Use top-down orientation for hovering above cube
    hover_approaching = np.array([0, 0, -1], dtype=np.float32)
    hover_closing = np.array([1, 0, 0], dtype=np.float32)
    hover_pose = env_inner.agent.build_grasp_pose(hover_approaching, hover_closing, hover_pos)

    res = planner.move_to_pose_with_RRTStar(hover_pose)
    if res == -1:
        planner.move_to_pose_with_screw(hover_pose)

    # Go down to cube center (bigger cube is easier to grasp)
    cube_pos = cube.pose.p[0].cpu().numpy()
    grasp_pos = np.array([cube_pos[0], cube_pos[1], cube_pos[2]])
    grasp_pose = env_inner.agent.build_grasp_pose(hover_approaching, hover_closing, grasp_pos)

    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1:
        planner.move_to_pose_with_RRTStar(grasp_pose)

    # Grasp the cube
    planner.close_gripper()

    # Lift straight up 15cm from current position
    tcp = env_inner.agent.tcp.pose
    lift_pose = sapien.Pose([tcp.sp.p[0], tcp.sp.p[1], tcp.sp.p[2] + 0.25], tcp.sp.q)
    planner.move_to_pose_with_screw(lift_pose)

    # Wait for settle
    for _ in range(2):
        obs, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape))

    if debug:
        cube_height = cube.pose.p[0, 2].item()
        is_grasped = env_inner.agent.is_grasping(cube)
        print(f"Final cube height: {cube_height:.3f}")
        print(f"Cube grasped: {is_grasped}")

    planner.close()
    return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    from mani_skill.envs.distraction_set import DistractionSet
    import gymnasium as gym

    env = gym.make(
        "PickCubeFromDrawer-v1",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
        distraction_set=DistractionSet(),
    )
    for seed in range(10):
        res = solve(env, seed=seed, debug=True, vis=False)
        if res != -1:
            print(f"Seed {seed}: Success={res[-1]['success']}")
        else:
            print(f"Seed {seed}: Failed")
    env.close()
