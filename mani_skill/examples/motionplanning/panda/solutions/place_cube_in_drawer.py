import numpy as np
import sapien

from mani_skill.envs.tasks.tabletop.colosseum_v2.place_cube_in_drawer import PlaceCubeInDrawerEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver



def move_to_pose(planner: PandaArmMotionPlanningSolver, pose: sapien.Pose, pose_name: str, debug: bool =False):
    def print_(s, *args, **kwargs):
        if debug:
            print(s, *args, **kwargs)
    res = planner.move_to_pose_with_screw(pose)
    if res != -1:
        print_(f"✅ {pose_name} | reached")
    else:
        print_(f"❌ {pose_name} | failed to reach: {pose}")
    return res


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

    res = move_to_pose(planner, pre_grasp_pose, "pre_grasp_pose", debug)
    if res == -1:
        return res

    # Move forward to grasp handle
    res = move_to_pose(planner, grasp_pose, "grasp_pose", debug)
    if res == -1:
        return res

    planner.close_gripper()

    # Get drawer joint limits
    qlimits = env_inner.handle_link.joint.limits
    qmin, qmax = qlimits[0, 0].item(), qlimits[0, 1].item()
    target_qpos = qmax  # Target fully open

    # Set drive target to open the drawer fully
    env_inner.handle_link.joint.set_drive_target(np.array([target_qpos]))

    # Pull drawer in +Y direction
    current_pos = grasp_pose.p.copy()
    step_size = 0.3  # 2cm steps
    max_steps = 23  # Up to 46cm total

    for _ in range(max_steps):
        current_pos[1] += step_size
        pull_pose = env_inner.agent.build_grasp_pose(handle_approaching, handle_closing, current_pos)

        res = move_to_pose(planner, pull_pose, "pull_pose", debug)
        if res == -1:
            return res

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

    # ========== PHASE 2: Pick cube from table ==========

    # Use top-down approach for cube
    cube_approaching = np.array([0, 0, -1], dtype=np.float32)
    cube_closing = np.array([1, 0, 0], dtype=np.float32)

    # Hover above cube
    cube_pos = cube.pose.p[0].cpu().numpy()
    hover_pos = np.array([cube_pos[0], cube_pos[1], cube_pos[2] + 0.15])
    hover_pose = env_inner.agent.build_grasp_pose(cube_approaching, cube_closing, hover_pos)

    res = move_to_pose(planner, hover_pose, "hover_pose", debug)
    if res == -1:
        return res
    
    # Go down to cube
    cube_pos = cube.pose.p[0].cpu().numpy()  # Refresh position
    grasp_pos = np.array([cube_pos[0], cube_pos[1], cube_pos[2]])
    cube_grasp_pose = env_inner.agent.build_grasp_pose(cube_approaching, cube_closing, grasp_pos)

    res = move_to_pose(planner, cube_grasp_pose, "cube_grasp_pose", debug)
    if res == -1:
        return res
    planner.close_gripper()

    # Lift cube
    tcp = env_inner.agent.tcp.pose
    lift_pos = np.array([tcp.sp.p[0], tcp.sp.p[1] - 0.1, tcp.sp.p[2] + 0.4])
    # lift_pos = np.array([tcp.sp.p[0], tcp.sp.p[1], tcp.sp.p[2] + 0.25])
    lift_pose = env_inner.agent.build_grasp_pose(cube_approaching, cube_closing, lift_pos)
    res = move_to_pose(planner, lift_pose, "lift_pose", debug)
    if res == -1:
        return res

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

    res = move_to_pose(planner, above_drawer_pose, "above_drawer_pose", debug)
    if res == -1:
        return res

    # Lower into drawer
    place_pos = drawer_interior.copy()
    place_pos[2] += 0.02  # Slightly above drawer floor
    place_pose = env_inner.agent.build_grasp_pose(cube_approaching, cube_closing, place_pos)

    res = move_to_pose(planner, place_pose, "place_pose", debug)
    if res == -1:
        return res

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
        reward_mode="none",
    )
    for seed in range(10):
        # res = solve(env, seed=seed, debug=True, vis=True)
        res = solve(env, seed=seed, debug=False, vis=True)
        if res != -1:
            print(f"Seed {seed}: success={res[-1]['success']}")
        else:
            print(f"Seed {seed}: Failed")
    env.close()
