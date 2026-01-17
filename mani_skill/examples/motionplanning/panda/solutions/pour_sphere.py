import numpy as np
import sapien

from mani_skill.envs.tasks.tabletop.pour_sphere import PourSphereEnv
from transforms3d.quaternions import axangle2quat, qmult

from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)


def solve(env: PourSphereEnv, seed=None, debug=False, vis=False):
    """Grasp cup1 from side, lift, position over cup2, rotate wrist to pour."""
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in ["pd_joint_pos", "pd_joint_pos_vel"]

    robot_base_pose_batched = env.unwrapped.agent.robot.pose
    robot_base_pose = sapien.Pose(
        p=robot_base_pose_batched.p[0].cpu().numpy(),
        q=robot_base_pose_batched.q[0].cpu().numpy()
    )

    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=robot_base_pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    env_sim = env.unwrapped
    cup1_pos = env_sim.cup1.pose.p[0].cpu().numpy()
    cup2_pos = env_sim.cup2.pose.p[0].cpu().numpy()
    print(f"cup1_pos: {cup1_pos}")
    print(f"cup2_pos: {cup2_pos}")

    def move_or_abort(target_pose, label):
        result = planner.move_to_pose_with_RRTConnect(target_pose)
        if not result or result == -1:
            result = planner.move_to_pose_with_screw(target_pose)
        if not result or result == -1:
            print(f"⚠ Plan FAILED at {label}")
            print(f"  Target pose: p={target_pose.p}, q={target_pose.q}")
            planner.close()
            return False
        return True

    planner.gripper_state = planner.OPEN
    robot_pos = robot_base_pose.p

    # Compute approach direction - approach from robot toward cup1
    robot_to_cup = cup1_pos[:2] - robot_pos[:2]
    robot_to_cup = robot_to_cup / (np.linalg.norm(robot_to_cup) + 1e-6)
    approaching = np.array([robot_to_cup[0], robot_to_cup[1], 0], dtype=np.float32)
    # Closing perpendicular to approach
    closing = np.array([-robot_to_cup[1], robot_to_cup[0], 0], dtype=np.float32)

    # Step 1: Pregrasp
    grasp_center = cup1_pos.copy()
    grasp_center[2] += env_sim._cup_height * 0.5
    pregrasp_pos = grasp_center - approaching * 0.08
    pregrasp_pose = env_sim.agent.build_grasp_pose(approaching, closing, pregrasp_pos)
    if not move_or_abort(pregrasp_pose, "pregrasp"):
        return -1

    # Step 2: Grasp
    grasp_pose = env_sim.agent.build_grasp_pose(approaching, closing, grasp_center)
    if not move_or_abort(grasp_pose, "grasp"):
        return -1

    # Step 3: Close gripper
    planner.close_gripper(t=100)
    for _ in range(20):
        env_sim.scene.step()

    # Step 4: Move above cup2 - offset toward robot for better pour
    above_cup2_pos = cup2_pos.copy()
    above_cup2_pos[2] += env_sim._cup_height + 0.05
    # Offset toward robot and in Y direction for better pour
    offset_dir = robot_pos[:2] - cup2_pos[:2]
    offset_dir = offset_dir / (np.linalg.norm(offset_dir) + 1e-6)
    above_cup2_pos[0] += offset_dir[0] * 0.05
    above_cup2_pos[1] += offset_dir[1] * 0.05
    above_cup2_pos[1] -= 0.05  # extra Y offset toward cup1
    above_cup2_pos[0] += 0.06  # 6cm forward (away from robot)
    tcp_q = env_sim.agent.tcp.pose.q[0].cpu().numpy()
    above_cup2_pose = sapien.Pose(p=above_cup2_pos, q=tcp_q)
    if not move_or_abort(above_cup2_pose, "above cup2"):
        return -1
    print("positioned above cup2")

    # Step 6: Tilt forward 120 degrees to pour (rotate around X axis)
    tcp_pose = env_sim.agent.tcp.pose
    tcp_p = tcp_pose.p[0].cpu().numpy()
    tcp_q = tcp_pose.q[0].cpu().numpy()
    tilt_angle = 120 * np.pi / 180
    tilt_quat = qmult(axangle2quat([1, 0, 0], -tilt_angle), tcp_q)  # tilt forward around X
    tilt_pose = sapien.Pose(p=tcp_p, q=tilt_quat)
    if not move_or_abort(tilt_pose, "tilt"):
        return -1

    # Hold to let sphere fall
    res = planner.close_gripper(t=60)

    planner.close()
    return res
