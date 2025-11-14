import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks.tabletop.open_laptop import OpenLaptopEnv
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float32)
    identity = np.eye(3, dtype=np.float32)
    return identity + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _hold(planner: PandaArmMotionPlanningSolver, seconds: float) -> None:
    zero_action = np.zeros(planner.env.action_space.shape, dtype=np.float32)
    steps = max(1, int(seconds / planner.base_env.control_timestep))
    for _ in range(steps):
        planner.env.step(zero_action)
        if planner.vis:
            planner.base_env.render_human()


def _move_to_position(planner: PandaArmMotionPlanningSolver, target_pos: np.ndarray, keep_orientation: bool = True) -> bool:
    """Move TCP to target position while optionally keeping current orientation. Returns True if successful."""
    for _ in range(100):
        current_pose = planner.base_env.agent.tcp_pose.sp
        current_pos = current_pose.p

        # Position error
        pos_error = target_pos - current_pos
        error_norm = np.linalg.norm(pos_error)

        if error_norm < 0.01:
            return True

        # Build action with stronger control
        action = np.zeros(planner.env.action_space.shape, dtype=np.float32)
        # Clip error to prevent huge actions
        pos_error_clipped = np.clip(pos_error, -0.05, 0.05)
        action[:3] = pos_error_clipped * 20.0  # Much stronger position control

        if keep_orientation:
            action[3:6] = 0  # Keep current orientation

        planner.env.step(action)
        if planner.vis:
            planner.base_env.render_human()

    return False


def solve(env: OpenLaptopEnv, seed=None, debug: bool = False, vis: bool = False):
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

    base_env = env.unwrapped

    # STEP 1: Get the grasp point - center of lip extending from front of lid
    lid_pose = base_env.lid_link.pose.sp
    lid_tf = lid_pose.to_transformation_matrix()
    R_lid = lid_tf[:3, :3]
    handle_local = base_env._handle_local.cpu().numpy()
    grasp_point = R_lid @ handle_local + lid_pose.p

    if debug:
        print(f"\n=== STEP 1: Identify grasp point ===")
        print(f"Lip grasp point: {grasp_point}")
        print(f"Robot TCP: {base_env.agent.tcp_pose.sp.p}")
        print(f"Distance: {np.linalg.norm(grasp_point - base_env.agent.tcp_pose.sp.p):.3f}m")

    # STEP 2: Approach from above with gripper angled slightly to grip lip
    approach_pos = grasp_point.copy()
    approach_pos[2] += 0.08  # Above lip

    # Rotate gripper: gentle 15-degree angle to approach thin lip
    # This allows the fingers to grip the thin lip edge better while remaining reachable
    target_rotation = euler2quat(0, np.deg2rad(75), 0)  # 75 degrees from horizontal = 15 degrees down from vertical
    approach_pose = sapien.Pose(q=target_rotation, p=approach_pos)

    if debug:
        print(f"\n=== STEP 2: Move to approach position ===")
        print(f"Approach position: {approach_pos}")

    planner.open_gripper()
    result = planner.move_to_pose_with_RRTConnect(approach_pose)
    if result == -1:
        if debug:
            print(f"FAILED to reach approach pose")
        planner.close()
        return -1

    if debug:
        print(f"SUCCESS - reached approach pose")

    # STEP 3: Move to grasp position - approach lip at an angle for better grip
    grasp_pos_adjusted = grasp_point.copy()

    if debug:
        print(f"\n=== STEP 3: Move to grasp position ===")
        print(f"Grasp position: {grasp_pos_adjusted}")
        print(f"Gripper angle: 15 degrees down for thin lip grip")

    grasp_pose = sapien.Pose(q=target_rotation, p=grasp_pos_adjusted)
    result = planner.move_to_pose_with_screw(grasp_pose)
    if result == -1:
        if debug:
            print("FAILED with screw motion, trying RRTConnect...")
        # Fallback to RRTConnect
        result = planner.move_to_pose_with_RRTConnect(grasp_pose)
        if result == -1:
            if debug:
                print("FAILED to reach grasp position")
            planner.close()
            return -1

    if debug:
        tcp_pos = base_env.agent.tcp_pose.sp.p
        print(f"SUCCESS - moved to grasp position")
        print(f"TCP now at: {tcp_pos}")
        print(f"Distance to target: {np.linalg.norm(grasp_pos_adjusted - tcp_pos):.4f}m")

    # STEP 4: Close gripper firmly
    if debug:
        print(f"\n=== STEP 4: Close gripper ===")

    _hold(planner, 0.5)
    planner.close_gripper(t=30)  # More time to close fully
    _hold(planner, 1.0)  # More time to stabilize grasp

    if debug:
        gripper_qpos = base_env.agent.robot.get_qpos()[0, 7:9].cpu().numpy()
        print(f"Gripper width: {gripper_qpos.sum():.4f}m")

    # STEP 5: Lift upward to open the lid
    # Maintain the 15-degree gripper angle for secure grip
    if debug:
        print(f"\n=== STEP 5: Lift up 20cm while maintaining angled grip ===")

    lift_pos = grasp_pos_adjusted.copy()
    lift_pos[2] += 0.20  # Move up 20cm to open lid
    lift_pose = sapien.Pose(q=target_rotation, p=lift_pos)

    result = planner.move_to_pose_with_screw(lift_pose)
    if result == -1:
        if debug:
            print("FAILED with screw motion, trying RRTConnect...")
        result = planner.move_to_pose_with_RRTConnect(lift_pose)

    if debug:
        if result == -1:
            print("FAILED to lift")
        else:
            print("SUCCESS - lifted")

    _hold(planner, 0.5)

    planner.open_gripper(t=10)
    _hold(planner, 0.5)

    # Get final evaluation from environment
    final_info = base_env.evaluate()
    final_info['elapsed_steps'] = torch.tensor([planner.elapsed_steps])

    success = bool(final_info['success'][0].item())

    if debug:
        final_angle = float(base_env.hinge_joint.qpos.cpu().numpy()[0])
        initial_angle = -float(base_env._initial_open_slack)
        angle_change = final_angle - initial_angle

        print(f"\n=== RESULT ===")
        print(f"Initial lid angle: {np.rad2deg(initial_angle):.1f} deg")
        print(f"Final lid angle: {np.rad2deg(final_angle):.1f} deg")
        print(f"Angle change: {np.rad2deg(angle_change):.1f} deg (positive = more open)")
        print(f"Target angle: {np.rad2deg(-base_env._success_open_angle):.1f} deg")
        print(f"Success: {success}")

    planner.close()

    # Return proper format (obs, reward, terminated, truncated, info)
    obs = base_env.get_obs()
    reward = 0.0
    terminated = success
    truncated = False

    return obs, reward, terminated, truncated, final_info
