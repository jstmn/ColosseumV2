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
    handle_pose_0 = handle_obj.pose 

    print("handle_obj.pose:", handle_obj.pose)





    print()
    print("======================================================================================")
    print("===========================================")
    print()

    rotation_noise_range = np.deg2rad(0.00001)
    rotation_noise = [np.random.uniform(-rotation_noise_range, rotation_noise_range) for _ in range(3)]
    target_rotation = euler2quat(np.pi / 2 + rotation_noise[0], -np.pi + rotation_noise[1], np.pi / 2 + rotation_noise[2]) # this was working
    
    grasp_pose = sapien.Pose(q=target_rotation, p=handle_obj.pose.p.cpu().numpy()[0])

    reach_pose = grasp_pose * sapien.Pose([0, 0.0, -0.05])
    # res = planner.move_to_pose_with_screw(grasp_pose, dry_run=False)
    res = planner.move_to_pose_with_screw(reach_pose, dry_run=True)
    if res == -1:
        print("YO: Failed to move to grasp pose")
        exit(1)

    # Configure target pose
    handle_target_pose = sapien.Pose(q=target_rotation, p=handle_pose_0.p.cpu().numpy()[0])
    handle_target_pose.p -= np.array([0.2, 0, 0])
    
    # Execution
    planner.open_gripper()
    planner.move_to_pose_with_screw(reach_pose)
    planner.open_gripper()
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()
    planner.move_to_pose_with_screw(handle_target_pose)
    planner.open_gripper()


    print()
    print("calculated grasp_pose")
    print("======================================================================================")







    # print("======================================================================================")
    # print("=========================================== Trying to plan to reach pose first:")
    # print()

    # target_rotation = euler2quat(np.pi / 2, -np.pi, np.pi / 2) # this was working
    # reach_pose = sapien.Pose(q=target_rotation, p=handle_pose_0.p.cpu().numpy()[0])
    # reach_pose.p -= np.array([0.1, 0, 0])
    # # reach_pose = grasp_pose * sapien.Pose([0, 0.0, -0.1])
    # # res = planner.move_to_pose_with_screw(reach_pose, dry_run=True)
    # res = planner.move_to_pose_with_screw(reach_pose, dry_run=False)
    # if res == -1:
    #     print("YO: Failed to move to reach pose")
    #     exit(1)

    # print()
    # print("calculated grasp, reach_pose")
    # print("======================================================================================")

    # planner.open_gripper()

    # # TEMP
    # for _ in range(10):
    #     planner.move_to_pose_with_screw(reach_pose)
    #     planner.open_gripper()

    #     for x in range(10):
    #         # 
    #         rotation_noise_range = np.deg2rad(2)
    #         rotation_noise = [np.random.uniform(-rotation_noise_range, rotation_noise_range) for _ in range(3)]
    #         target_rotation = euler2quat(np.pi / 2 + rotation_noise[0], -np.pi + rotation_noise[1], np.pi / 2 + rotation_noise[2])
    #         # 
    #         grasp_pose = sapien.Pose(q=target_rotation, p=handle_pose_0.p.cpu().numpy()[0])
    #         if not planner.move_to_pose_with_screw(grasp_pose) == -1:
    #             planner.move_to_pose_with_screw(grasp_pose)
    #             planner.close_gripper()
    #             break
    #         print("  i:", x, "- failed")
    #     else:
    #         print("NO SUCCESSFUL GRASP")
    #         exit(1)
    #     # planner.move_to_pose_with_screw(handle_target_pose)
    #     # planner.open_gripper()





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
