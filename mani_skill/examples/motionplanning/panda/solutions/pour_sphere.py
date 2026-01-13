import numpy as np
import sapien

from mani_skill.envs.tasks.tabletop.pour_sphere import PourSphereEnv
from transforms3d.quaternions import axangle2quat, qmult

from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)


def solve(env: PourSphereEnv, seed=None, debug=False, vis=False):
    """Grasp cup and tilt it over target cup using joint 7 rotation."""
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

    def find_reachable_pose(pos, base_q, yaw_angles):
        for angle in yaw_angles:
            yaw_q = qmult(axangle2quat([0, 0, 1], angle), base_q)
            if np.dot(yaw_q, base_q) < 0:
                yaw_q = -yaw_q
            candidate = sapien.Pose(p=pos, q=yaw_q)
            if planner.move_to_pose_with_RRTConnect(candidate, dry_run=True) != -1:
                return candidate
        return None

    planner.gripper_state = planner.OPEN

    # Step 1: Side approach to cup1 so the opening stays clear.
    approaching = np.array([1, 0, 0], dtype=np.float32)
    closing = np.array([0, -1, 0], dtype=np.float32)
    grasp_center = cup1_pos.copy()
    grasp_center[2] += env_sim._cup_height * 0.6
    pregrasp_pos = grasp_center - approaching * 0.12
    pregrasp_pose = env_sim.agent.build_grasp_pose(
        approaching, closing, pregrasp_pos
    )
    if not move_or_abort(pregrasp_pose, "Step 1 (pregrasp)"):
        return -1

    # Step 2: Move to grasp pose
    grasp_pose = env_sim.agent.build_grasp_pose(approaching, closing, grasp_center)
    if not move_or_abort(grasp_pose, "Step 2 (grasp)"):
        return -1

    # Step 3: Close gripper
    planner.close_gripper(t=120)
    for _ in range(30):
        env_sim.scene.step()

    # Step 4: Lift cup
    current_tcp_pose = env_sim.agent.tcp.pose
    current_tcp_p = current_tcp_pose.p[0].cpu().numpy()
    current_tcp_q = current_tcp_pose.q[0].cpu().numpy()
    lift_pose = sapien.Pose(p=current_tcp_p + np.array([0, 0, 0.22]), q=current_tcp_q)
    if not move_or_abort(lift_pose, "Step 3 (lift)"):
        return -1

    print("cup lifted")

    cup1_pos = env_sim.cup1.pose.p[0].cpu().numpy()
    cup2_pos = env_sim.cup2.pose.p[0].cpu().numpy()
    tcp_pose = env_sim.agent.tcp.pose

    # Step 4 - move above cup2 while keeping orientation.
    above_cup2_pos = cup2_pos.copy()
    above_cup2_pos[1] += 0.08
    above_cup2_pos[2] += env_sim._cup_height + 0.22
    base_q = tcp_pose.q[0].cpu().numpy()
    yaw_angles = [0.0, np.pi / 4, -np.pi / 4]
    above_cup2_pose = find_reachable_pose(above_cup2_pos, base_q, yaw_angles)
    if above_cup2_pose is None:
        above_cup2_pos[2] += 0.10
        above_cup2_pose = find_reachable_pose(above_cup2_pos, base_q, yaw_angles)
    if above_cup2_pose is None:
        print("⚠ Plan FAILED at Step 4 (above cup2): no reachable yaw found")
        planner.close()
        return -1
    if planner.move_to_pose_with_RRTConnect(above_cup2_pose) == -1:
        planner.close()
        return -1

    #Step 4.5 - lower into pouring position
    lower_pose = sapien.Pose(p=above_cup2_pose.p - np.array([0, 0, 0.1]), q=above_cup2_pose.q)
    if not move_or_abort(lower_pose, "Step 4.5 (lower)"):
        return -1
    print("positioned above cup2")
    # Step 5 - tilt to pour.
    tcp_pose = env_sim.agent.tcp.pose
    tcp_p = tcp_pose.p[0].cpu().numpy()
    tcp_q = tcp_pose.q[0].cpu().numpy()
    tilt_quat = qmult(axangle2quat([1, 0, 0], -np.pi * 2 / 3), tcp_q)
    tilt_pose = sapien.Pose(p=tcp_p, q=tilt_quat)
    if not move_or_abort(tilt_pose, "Step 5 (tilt)"):
        return -1

    # Hold pose to let the sphere fall.
    final_res = planner.close_gripper(t=60)

    planner.close()
    return final_res


    
