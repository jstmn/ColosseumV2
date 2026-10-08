"""Successful LiftPegUpright-v2 motion plan.

Off-center pitched grasp, lift, rotate about gripper Y so the peg stands upright,
then lower in one joint-space segment and release.
"""

import numpy as np
import sapien

from mani_skill.envs.mpcm.lift_peg_upright_envs import LiftPegUprightEnv
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

FINGER_LENGTH = 0.025
PITCHED_GRASP_OFFSET = 0.08
PITCHED_GRASP_Y_DEG = -45.0
PITCHED_REACH_DZ = 0.12
PITCHED_LIFT_DZ = 0.30
PITCHED_ROTATE_Y_DEG = 90.0
PITCHED_PLACE_Z_TOL = 0.004


def _rotation_about_y(angle_deg: float) -> sapien.Pose:
    half = np.deg2rad(angle_deg) / 2.0
    return sapien.Pose(q=np.array([np.cos(half), 0.0, np.sin(half), 0.0]))


def _peg_z(env: LiftPegUprightEnv) -> float:
    return float(np.asarray(env.peg.pose.p.detach().cpu().numpy(), dtype=float).reshape(-1)[2])


def _tcp_pose(env: LiftPegUprightEnv) -> sapien.Pose:
    tcp = env.agent.tcp.pose
    p = np.asarray(tcp.p.detach().cpu().numpy(), dtype=float).reshape(-1)
    q = np.asarray(tcp.q.detach().cpu().numpy(), dtype=float).reshape(-1)
    return sapien.Pose(p=p[:3], q=q[:4])


def _pitched_grasp_pose(env: LiftPegUprightEnv) -> sapien.Pose:
    obb = get_actor_obb(env.peg)
    approaching = np.array([0.0, 0.0, -1.0])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    grasp_pose = env.agent.build_grasp_pose(approaching, grasp_info["closing"], grasp_info["center"])
    grasp_pose = grasp_pose * sapien.Pose([PITCHED_GRASP_OFFSET, 0, 0])
    return grasp_pose * _rotation_about_y(PITCHED_GRASP_Y_DEG)


def _straight_down_pose(env: LiftPegUprightEnv) -> sapien.Pose:
    peg_z = _peg_z(env)
    dz = env.peg_half_length - peg_z
    assert dz < -PITCHED_PLACE_Z_TOL, f"lower stage started with the peg already at the place height, peg_z={peg_z}"
    tcp = _tcp_pose(env)
    return sapien.Pose(p=[tcp.p[0], tcp.p[1], tcp.p[2] + dz], q=np.asarray(tcp.q, dtype=float))


def _lower_in_one_motion(planner: PandaArmMotionPlanningSolver, env: LiftPegUprightEnv):
    """One joint-space segment from the current arm pose to the place pose.

    A failed IK or time parameterization fails the demo. There is no waypoint sequence and no screw.
    """
    lower_pose = _straight_down_pose(env)
    mplib_planner = planner.planner
    current_qpos = planner.robot.get_qpos().cpu().numpy()[0].copy()
    index = mplib_planner.move_group_joint_indices
    goal = planner._transform_pose_for_planning(lower_pose)
    goal_pose = np.concatenate([np.asarray(goal.p, dtype=float), np.asarray(goal.q, dtype=float)])
    status, solutions = mplib_planner.IK(goal_pose, current_qpos, n_init_qpos=20)
    if status != "Success":
        print("straight-down IK failed")
        return -1
    solution = min(
        solutions,
        key=lambda candidate: float(np.linalg.norm(np.asarray(candidate)[index] - current_qpos[index])),
    )
    path = np.vstack([current_qpos[index], np.asarray(solution, dtype=float)[index]])
    try:
        _, position, velocity, _, _ = mplib_planner.TOPP(path, planner.base_env.control_timestep)
    except RuntimeError as error:
        print(f"straight-down TOPP failed: {error}")
        return -1
    return planner.follow_path({"position": position, "velocity": velocity})


def solve(
    env: LiftPegUprightEnv,
    seed=None,
    debug=False,
    vis=False,
    slow_down: bool = False,
    add_sinusoidal_noise: bool = False,
):
    env.reset(seed=seed)
    assert env.unwrapped.control_mode == "pd_joint_pos", f"control_mode={env.unwrapped.control_mode!r}"
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        joint_vel_limits=0.75,
        joint_acc_limits=0.75,
        slow_down=slow_down,
        add_sinusoidal_noise=add_sinusoidal_noise,
    )
    env = env.unwrapped

    grasp_pose = _pitched_grasp_pose(env)
    reach_pose = sapien.Pose([0.0, 0.0, PITCHED_REACH_DZ]) * grasp_pose
    res = planner.move_to_pose_with_screw(reach_pose)
    if res == -1:
        return res

    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1:
        return res
    planner.close_gripper(gripper_state=-0.6)

    lift_pose = sapien.Pose([0.0, 0.0, PITCHED_LIFT_DZ]) * grasp_pose
    res = planner.move_to_pose_with_screw(lift_pose)
    if res == -1:
        return res

    upright_pose = lift_pose * _rotation_about_y(PITCHED_ROTATE_Y_DEG)
    res = planner.move_to_pose_with_screw(upright_pose)
    if res == -1:
        return res

    res = _lower_in_one_motion(planner, env)
    if res == -1:
        return res

    res = planner.open_gripper()
    planner.close()
    return res
