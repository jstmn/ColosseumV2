import gymnasium as gym
import numpy as np
import sapien

from mani_skill.envs.tasks.tabletop.colosseum_v2.book_in_shelf import PlaceBookEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb

def main():
    env: PlaceBookEnv = gym.make(
        "PlaceBookInShelf-v1",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="none",
    )
    for seed in range(100):
        # res = solve(env, seed=seed, debug=True, vis=True)
        res = solve(env, seed=seed, debug=False, vis=True)
        print(res)
    env.close()

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


def solve(env: PlaceBookEnv, seed=None, debug=False, vis=False):

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
        joint_vel_limits=0.75,
        joint_acc_limits=0.75,
    )
    
    env = env.unwrapped
    FINGER_LENGTH = 0.025
    obb = get_actor_obb(env.book_A)
    approaching = np.array([1, 0, 0])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)    
    
    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    approach_pose = grasp_pose * sapien.Pose([0.05, 0.0, -0.1])  # slightly above the book center
    # ^ z: +x world
    # ^ y: -y world
    # ^ x: +z world
    res = move_to_pose(planner, approach_pose, "approach_pose", debug)
    if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    res = move_to_pose(planner, grasp_pose, "grasp_pose", debug)
    if res == -1: 
        return res

    planner.close_gripper(gripper_state=-0.6)
    approach_placement_pose = sapien.Pose(p=[-0.053 + env.shelf.pose.p[0,0]-0.293, -0.160+env.shelf.pose.p[0,1]+0.1, 0.2],q=grasp_pose.q)
    res = move_to_pose(planner, approach_placement_pose, "approach_placement_pose", debug)
    if res == -1: 
        return res

    # -------------------------------------------------------------------------- #
    # Lower
    # -------------------------------------------------------------------------- #
    placement_pose = approach_placement_pose * sapien.Pose([0, 0, 0.2])
    res = move_to_pose(planner, placement_pose, "placement_pose", debug)
    if res == -1: return res

    #
    planner.open_gripper()
    res = move_to_pose(planner, approach_placement_pose, "approach_placement_pose", debug)
    if res == -1: return res

    planner.close()
    
    return res

if __name__ == "__main__":
    main()
