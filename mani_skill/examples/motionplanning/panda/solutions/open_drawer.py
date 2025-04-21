import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import OpenCabinetDrawerV2Env
from mani_skill.utils import common
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb
from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply

def solve(env: OpenCabinetDrawerV2Env, seed=None, debug=False, vis=False):
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
    FINGER_LENGTH = 0.025
    env = env.unwrapped
    handle_obj = env.handle_link_goal

    print("handle_obj.pose:", handle_obj.pose)

    print()
    print("======================================================================================")
    print("===========================================")
    print()

    rotation_noise_range = np.deg2rad(0.0001)
    rotation_noise = [np.random.uniform(-rotation_noise_range, rotation_noise_range) for _ in range(3)]
    target_rotation = euler2quat(np.pi / 2 + rotation_noise[0], -np.pi + rotation_noise[1], np.pi / 2 + rotation_noise[2]) # this was working
    
    grasp_pose = handle_obj.pose  # q: [1,  0,  0,  0]
    grasp_pose.q = target_rotation
    grasp_pose.p -= np.array([0.1, 0, 0])
    res = planner.move_to_pose_with_screw(grasp_pose, dry_run=True)
    if res == -1:
        print("YO: Failed to move to grasp pose")
        return


    # handle_target_pose = sapien.Pose(q=target_rotation, p=grasp_pose.p.cpu().numpy() + np.array([-0.15, 0, 0]))
    # reach_pose = sapien.Pose(q=target_rotation, p=grasp_pose.p + np.array([-0.1, 0, 0]))
    # handle_target_pose = grasp_pose * sapien.Pose([0, 0, -0.2])
    # reach_pose = grasp_pose * sapien.Pose([0, 0, -0.15])

    print()
    print("calculated grasp_pose")
    print("======================================================================================")




    
    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.1])


    # TEMP
    for _ in range(10):
        planner.open_gripper()
        planner.move_to_pose_with_screw(reach_pose)
        planner.open_gripper()
        planner.move_to_pose_with_screw(grasp_pose)
        planner.close_gripper()
        # planner.move_to_pose_with_screw(handle_target_pose)
        # planner.open_gripper()


    # 


    # approach_pose = grasp_pose * sapien.Pose(p=[0, 0, -0.01])
    # planner.move_to_pose_with_screw(approach_pose)
    # planner.open_gripper()

    # # TEMP
    # for _ in range(10):
    #     planner.open_gripper()
    #     planner.move_to_pose_with_screw(grasp_pose)
    #     planner.close_gripper()
    #     planner.move_to_pose_with_screw(handle_target_pose)
    #     planner.open_gripper()


    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    # planner.move_to_pose_with_screw(grasp_pose)
    # planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    # lift_pose = sapien.Pose([0, 0, 0.1]) * grasp_pose
    # planner.move_to_pose_with_screw(lift_pose)

    # -------------------------------------------------------------------------- #
    # Stack
    # -------------------------------------------------------------------------- #
    # block_half_size_torch = common.to_tensor(env.block_half_size)
    # goal_pose = env.bin.pose * sapien.Pose([0, 0, (block_half_size_torch[2] * 2).item()])
    # offset = (goal_pose.p - env.obj.pose.p).cpu().numpy()[0] # remember that all data in ManiSkill is batched and a torch tensor
    # align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    # planner.move_to_pose_with_screw(align_pose)

    res = planner.open_gripper()
    planner.close()
    return res
