import numpy as np
import sapien

from mani_skill.envs.tasks.tabletop.place_cube_in_drawer import PlaceCubeInDrawerEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver


def solve(env: PlaceCubeInDrawerEnv, seed=None, debug=False, vis=False):
    """
    Solution for PlaceCubeInDrawer task.
    1. Open the drawer
    2. Pick cube from table
    3. Place cube inside drawer
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
    cube = env_inner.cube

    # ========== PHASE 1: Open the drawer ==========
    # (Copied from pick_cube_from_drawer.py)
    if debug:
        print("Phase 1: Opening drawer...")

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
            break

        # Check if drawer is open enough (70%+)
        qpos = env_inner.handle_link.joint.qpos[0].item()
        pct = (qpos - qmin) / (qmax - qmin) * 100
        if debug:
            print(f"Drawer open: {pct:.1f}%")
        if pct >= 70:
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

    # Lift up
    lift_pos = retreat_pos.copy()
    lift_pos[2] = 0.50
    lift_pose = env_inner.agent.build_grasp_pose(handle_approaching, handle_closing, lift_pos)
    res = planner.move_to_pose_with_screw(lift_pose)
    if res == -1:
        planner.move_to_pose_with_RRTStar(lift_pose)

    # ========== PHASE 2: Pick cube from table ==========
    if debug:
        print("Phase 2: Picking cube from table...")

    cube_pos = cube.pose.p[0].cpu().numpy()

    if debug:
        print(f"Cube position: {cube_pos}")

    # Use top-down approach for cube
    cube_approaching = np.array([0, 0, -1], dtype=np.float32)
    cube_closing = np.array([1, 0, 0], dtype=np.float32)

    # Hover above cube
    hover_pos = np.array([cube_pos[0], cube_pos[1], cube_pos[2] + 0.15])
    hover_pose = env_inner.agent.build_grasp_pose(cube_approaching, cube_closing, hover_pos)

    res = planner.move_to_pose_with_RRTStar(hover_pose)
    if res == -1:
        res = planner.move_to_pose_with_screw(hover_pose)
        if res == -1:
            if debug:
                print("Failed to reach hover position above cube")
            planner.close()
            return -1

    # Go down to cube
    cube_pos = cube.pose.p[0].cpu().numpy()  # Refresh position
    grasp_pos = np.array([cube_pos[0], cube_pos[1], cube_pos[2]])
    cube_grasp_pose = env_inner.agent.build_grasp_pose(cube_approaching, cube_closing, grasp_pos)

    res = planner.move_to_pose_with_screw(cube_grasp_pose)
    if res == -1:
        planner.move_to_pose_with_RRTStar(cube_grasp_pose)

    planner.close_gripper()

    if debug:
        print("Cube grasped")

    # Lift cube
    tcp = env_inner.agent.tcp.pose
    lift_pos = np.array([tcp.sp.p[0], tcp.sp.p[1], tcp.sp.p[2] + 0.25])
    lift_pose = env_inner.agent.build_grasp_pose(cube_approaching, cube_closing, lift_pos)
    planner.move_to_pose_with_screw(lift_pose)

    # ========== PHASE 3: Place cube in drawer ==========
    if debug:
        print("Phase 3: Placing cube in drawer...")

    # Get current handle position (drawer is now open)
    handle_pos = handle_obj.pose.p.cpu().numpy()[0]

    # Drawer interior is behind the handle (in +Y direction since drawer faces -Y)
    drawer_interior = handle_pos.copy()
    drawer_interior[1] -= 0.12  # Into the drawer

    if debug:
        print(f"Drawer interior target: {drawer_interior}")

    # Move above drawer interior
    above_drawer = drawer_interior.copy()
    above_drawer[1] += 0.05
    above_drawer[2] += 0.25  # Above drawer
    above_drawer_pose = env_inner.agent.build_grasp_pose(cube_approaching, cube_closing, above_drawer)

    res = planner.move_to_pose_with_RRTStar(above_drawer_pose)
    if res == -1:
        res = planner.move_to_pose_with_screw(above_drawer_pose)
        if res == -1:
            if debug:
                print("Failed to reach above drawer position")
            planner.close()
            return -1

    # Lower into drawer
    place_pos = drawer_interior.copy()
    place_pos[2] += 0.02  # Slightly above drawer floor
    place_pose = env_inner.agent.build_grasp_pose(cube_approaching, cube_closing, place_pos)

    res = planner.move_to_pose_with_screw(place_pose)
    if res == -1:
        planner.move_to_pose_with_RRTStar(place_pose)

    # Release cube
    planner.open_gripper()

    if debug:
        print("Cube released in drawer")

    # Retreat upward
    retreat_pos = place_pos.copy()
    retreat_pos[2] += 0.15
    retreat_pose = env_inner.agent.build_grasp_pose(cube_approaching, cube_closing, retreat_pos)
    planner.move_to_pose_with_screw(retreat_pose)

    # Wait for settle
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape))

    if debug:
        print(f"Success: {info['success']}")
        print(f"Drawer open: {info['is_drawer_open']}")
        print(f"Cube in drawer: {info['is_cube_in_drawer']}")
        print(f"Cube static: {info['is_cube_static']}")
        print(f"Cube grasped: {info['is_cube_grasped']}")

    planner.close()
    return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    import gymnasium as gym

    env = gym.make(
        "PlaceCubeInDrawer-v1",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
    )
    for seed in range(10):
        res = solve(env, seed=seed, debug=True, vis=False)
        if res != -1:
            print(f"Seed {seed}: Success={res[-1]['success']}")
        else:
            print(f"Seed {seed}: Failed")
    env.close()
