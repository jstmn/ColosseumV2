import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import cast
from termcolor import cprint
import h5py
import os
from copy import deepcopy
import time
import traceback
import argparse
import gymnasium as gym
import json
import numpy as np
from tqdm import tqdm
import os.path as osp
import mani_skill.envs
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.trajectory.merge_trajectory import merge_trajectories
from mani_skill.examples.motionplanning.panda.solutions import solvePushCube, solvePickCube, solveStackCube, solvePegInsertionSide, solvePlugCharger, solvePullCubeTool, solveLiftPegUpright, solvePullCube, solveDrawTriangle, solveDrawSVG, solvePlaceSphere,solveOpenDrawer,solveRaiseCube, solvePlaceBookInShelf, solveHangClothingFrameOnPole, solvePickSodaFromCabinet, solveRotateArrow, solveScoopBanana, solvePickLightbulbPlaceSocket, solvePlaceAppleOnPlate, solveCookItemInPan, solvePickBananaFromOpenDrawer,solvePlaceDishInRack,solvePickDishFromRack,solvePourSphere, solveHammerNail, solveOpenCabinet, solveObjectInCabinet, solvePickCubeFromDrawer, solvePlaceCubeInDrawer
from mani_skill.envs.tasks.tabletop.colosseum_v2.distraction_set import DISTRACTION_SETS
from mani_skill.examples.motionplanning.dual_panda.solutions import solveBimanualLiftPot, solveBimanualLiftTray, solveBimanualPassBottle, solveBimanualPourPot, solveBimanualPassCube, solveBimanualDrawerPlace, solveBimanualPourPot, solveBimanualDrawerOpen, solveBimanualPenCap, solveBimanualPushBox, solveBimanualStack3Cubes, solveBimanualStackCubes, solveBimanualThreading

MP_SOLUTIONS = {
    "DrawTriangle-v1": solveDrawTriangle,
    "PickCube-v1": solvePickCube,
    "PickCubeMP-v1": solvePickCube,
    "StackCube-v1": solveStackCube,
    "PegInsertionSide-v1": solvePegInsertionSide,
    "PlugCharger-v1": solvePlugCharger,
    "PlaceSphere-v1": solvePlaceSphere,
    "PushCube-v1": solvePushCube,
    "PullCubeTool-v1": solvePullCubeTool,
    "LiftPegUpright-v1": solveLiftPegUpright,
    "PullCube-v1": solvePullCube,
    "DrawSVG-v1" : solveDrawSVG,
    # 
    # New tasks:
    "RaiseCube-v1": solveRaiseCube,
    "OpenDrawer-v1": solveOpenDrawer,               # new
    "PushCube-v2": solvePushCube,                   # new
    "StackCube-v2": solveStackCube,                 # new

    "PlaceBookInShelf-v1": solvePlaceBookInShelf,
    "HangClothingFrameOnPole-v1": solveHangClothingFrameOnPole,
    "PickSodaFromCabinet-v1": solvePickSodaFromCabinet,
    "RotateArrow-v1": solveRotateArrow,
    "ScoopBanana-v1": solveScoopBanana,
    "PickBananaFromOpenDrawer-v1": solvePickBananaFromOpenDrawer,    # new
    "PickCubeFromDrawer-v1": solvePickCubeFromDrawer,              # new
    "PickLightbulbPlaceSocket-v1": solvePickLightbulbPlaceSocket, #new
    "PlaceAppleOnPlate-v1": solvePlaceAppleOnPlate, # new
    "CookItemInPan-v1": solveCookItemInPan,
    "PlaceDishInRack-v1": solvePlaceDishInRack, # new
    "PickDishFromRack-v1": solvePickDishFromRack, # new
    "PourSphere-v1": solvePourSphere, # new
    "PegInsertionSideColosseumV2-v1": solvePegInsertionSide, # new
    "PlugChargerColosseumV2-v1": solvePlugCharger, # new
    "HammerNail-v1": solveHammerNail,
    "OpenCabinet-v1": solveOpenCabinet,
    "ObjectInCabinet-v1": solveObjectInCabinet,
    "PlaceCubeInDrawer-v1": solvePlaceCubeInDrawer,
    "StackCubeColosseumV2-v1": solveStackCube,
    "LiftPegUprightColosseumV2-v1": solveLiftPegUpright,
    # Bimanual
    "DualArmPickCube-v1": solveBimanualPassCube,
    "DualArmLiftPot-v1": solveBimanualLiftPot,
    "DualArmLiftTray-v1": solveBimanualLiftTray,
    "DualArmPickBottle-v1": solveBimanualPassBottle,
    "DualArmPourPot-v1": solveBimanualPourPot,
    "DualArmDrawerPlace-v1": solveBimanualDrawerPlace,
    "DualArmDrawerOpen-v1": solveBimanualDrawerOpen,
    "DualArmPenCap-v1": solveBimanualPenCap,
    "DualArmPushBox-v1": solveBimanualPushBox,
    "DualArmStack3Cube-v1": solveBimanualStack3Cubes,
    "DualArmStackCube-v1": solveBimanualStackCubes,
    "DualArmThreading-v1": solveBimanualThreading
}

"""
# Colosseum v2 single-arm tasks
ENV_ID="RaiseCube-v1"
ENV_ID="PickSodaFromCabinet-v1"
ENV_ID="PickDishFromRack-v1"
ENV_ID="StackCube-v1"
ENV_ID="PlaceBookInShelf-v1"
ENV_ID="PlaceDishInRack-v1"
ENV_ID="LiftPegUpright-v1"
ENV_ID="RotateArrow-v1"
ENV_ID="PegInsertionSide-v2"
ENV_ID="PlugChargerColosseumV2-v1"
ENV_ID="HammerNail-v1"
ENV_ID="ScoopBanana-v1"
ENV_ID="OpenDrawer-v1"
ENV_ID="OpenCabinet-v1"
ENV_ID="PlaceCubeInDrawer-v1"
ENV_ID="CookItemInPan-v1"

# Colosseum v2 bimanual tasks
ENV_ID="DualArmPickCube-v1"
ENV_ID="DualArmPickBottle-v1"
ENV_ID="DualArmLiftPot-v1"
ENV_ID="DualArmLiftTray-v1"
ENV_ID="DualArmPushBox-v1"
ENV_ID="DualArmPourPot-v1"
ENV_ID="DualArmThreading-v1"
ENV_ID="DualArmPenCap-v1"
ENV_ID="DualArmDrawerPlace-v1"
ENV_ID="DualArmDrawerOpen-v1"
ENV_ID="DualArmStackCube-v1"
ENV_ID="DualArmStack3Cube-v1"





ENV_ID="RaiseCube-v1"
DISTRACTION_SET=all
# ^ Must be one of: none, all, distractor_object_cfg, MO_color_cfg, MO_texture_cfg, RO_color_cfg, RO_texture_cfg, table_color_cfg, table_texture_cfg, camera_pose_cfg


python mani_skill/examples/motionplanning/panda/run.py \
    --env-id ${ENV_ID} \
    --num-traj 100 \
    --distraction-set ${DISTRACTION_SET} \
    --num-procs 2 \
    --obs-mode "rgb" \
    --reward-mode "sparse" \
    --random-seed \
    --only-count-success \
    --traj-name "trajectory" --vis \
    --save-video      # <- optional
    # --vis           # <- optional

# Convert to ee_delta_pos with:
python mani_skill/trajectory/replay_trajectory.py \
    --traj-path demos/PickCube-v1/motionplanning/trajectory.h5 \
    --obs-mode "rgb" \
    --target_control_mode "pd_ee_delta_pos" \
    --save-traj \
    --save-video      # <- optional
"""


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PickCube-v1", help=f"Environment to run motion planning solver on. Available options are {list(MP_SOLUTIONS.keys())}")
    parser.add_argument("-o", "--obs-mode", type=str, default="none", help="Observation mode to use. Usually this is kept as 'none' as observations are not necesary to be stored, they can be replayed later via the mani_skill.trajectory.replay_trajectory script.")
    parser.add_argument("-n", "--num-traj", type=int, default=10, help="Number of trajectories to generate.")
    parser.add_argument("--only-count-success", action="store_true", help="If true, generates trajectories until num_traj of them are successful and only saves the successful trajectories/videos")
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-b", "--sim-backend", type=str, default="auto", help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
    parser.add_argument("--render-mode", type=str, default="rgb_array", help="can be 'sensors' or 'rgb_array' which only affect what is saved to videos")
    parser.add_argument("--vis", action="store_true", help="whether or not to open a GUI to visualize the solution live")
    parser.add_argument("--save-video", action="store_true", help="whether or not to save videos locally")
    parser.add_argument("--random-seed", action="store_true", help="whether or not to randomize the seed for each process")
    parser.add_argument("--traj-name", type=str, help="The name of the trajectory .h5 file that will be created.")
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("--record-dir", type=str, default="demos", help="where to save the recorded trajectories")
    parser.add_argument("--num-procs", type=int, default=1, help="Number of processes to use to help parallelize the trajectory replay process. This uses CPU multiprocessing and only works with the CPU simulation backend at the moment.")
    parser.add_argument("--distraction-set", type=str, required=True, help=f"Distraction set to use. Available options are {list(DISTRACTION_SETS.keys())}")
    parser.add_argument("--save-images", action="store_true", help="whether or not to save images locally")
    parser.add_argument("--ignore-keys", nargs="*", default=[], help="keys to ignore when saving the trajectory")
    return parser.parse_args()

def _main(args, proc_id: int = 0, start_seed: int = 0) -> str:
    env_id = args.env_id
    distraction_set = DISTRACTION_SETS[args.distraction_set.upper()]
    try:
        env = gym.make(
            env_id,
            obs_mode=args.obs_mode,
            control_mode="pd_joint_pos",
            render_mode=args.render_mode,
            reward_mode="dense" if args.reward_mode is None else args.reward_mode,
            sensor_configs=dict(shader_pack=args.shader),
            human_render_camera_configs=dict(shader_pack=args.shader),
            viewer_camera_configs=dict(shader_pack=args.shader),
            sim_backend=args.sim_backend,
            distraction_set=distraction_set
        )
    except TypeError as e:
        assert "got an unexpected keyword argument 'distraction_set'" in str(e)
        env = gym.make(
            env_id,
            obs_mode=args.obs_mode,
            control_mode="pd_joint_pos",
            render_mode=args.render_mode,
            reward_mode="dense" if args.reward_mode is None else args.reward_mode,
            sensor_configs=dict(shader_pack=args.shader),
            human_render_camera_configs=dict(shader_pack=args.shader),
            viewer_camera_configs=dict(shader_pack=args.shader),
            sim_backend=args.sim_backend,
        )

    if env_id not in MP_SOLUTIONS:
        raise RuntimeError(f"No already written motion planning solutions for {env_id}. Available options are {list(MP_SOLUTIONS.keys())}")

    if not args.traj_name:
        new_traj_name = time.strftime("%Y-%m-%d_%H:%M:%S")
    else:
        new_traj_name = args.traj_name

    if args.num_procs > 1:
        new_traj_name = new_traj_name + "." + str(proc_id)
    env = RecordEpisode(
        cast(BaseEnv, env),
        output_dir=osp.join(args.record_dir, env_id, "motionplanning"),
        trajectory_name=new_traj_name, save_video=args.save_video,
        source_type="motionplanning",
        source_desc="official motion planning solution from ManiSkill contributors",
        video_fps=30,
        record_reward=False,
        save_on_reset=False
    )
    output_h5_path = env._h5_file.filename
    solve = MP_SOLUTIONS[env_id]
    pbar = tqdm(range(args.num_traj), desc=f"proc_id: {proc_id}")
    seed = start_seed
    successes = []
    solution_episode_lengths = []
    failed_motion_plans = 0
    passed = 0
    counter = 0
    while True:
        counter += 1
        env.reset(seed=seed, options={"reconfigure": True}) # reconfigure so distractor variations are resampled
        res = solve(env, seed=seed, debug=False, vis=True if args.vis else False)
        # try:
        # except Exception as e:
        #     print(f"Cannot find valid solution because of an error in motion planning solution: {e}")
        #     print("Traceback:")
        #     print(''.join(traceback.format_tb(e.__traceback__)))
        #     res = -1

        if res == -1:
            success = False
            failed_motion_plans += 1
        else:
            success = res[-1]["success"].item()
            elapsed_steps = res[-1]["elapsed_steps"].item()

        successes.append(success)
        if args.only_count_success and not success:
            seed += 1
            env.flush_trajectory(save=False)
            if args.save_video:
                env.flush_video(save=False)
            continue
        else:
            # Only save episode length if the solution was successful
            solution_episode_lengths.append(elapsed_steps)
            env.flush_trajectory()
            if args.save_video:
                env.flush_video(name=f"{new_traj_name}___n:{len(successes)}")
            pbar.update(1)
            seed += 1
            passed += 1
            if passed == args.num_traj:
                break

        pbar.set_postfix(
            dict(
                success_pct=f"{100 - ((failed_motion_plans / counter) * 100):.2f}%",
                failed_motion_plan_pct=f"{(failed_motion_plans / counter) * 100:.2f}%",
                avg_episode_length=np.mean(solution_episode_lengths),
                max_episode_length=np.max(solution_episode_lengths) if solution_episode_lengths else -1,
                min_episode_length=np.min(solution_episode_lengths) if solution_episode_lengths else -1
            )
        )
    env.close()

    print()
    print(f"Summary ({proc_id=}):")
    print("  success_rate:           ", np.mean(successes),)
    print("  failed_motion_plan_rate:", failed_motion_plans / (seed + 1))
    print("  avg_episode_length:     ", np.mean(solution_episode_lengths))
    print("  std_episode_length:     ", np.std(solution_episode_lengths))
    print("  max_episode_length:     ", np.max(solution_episode_lengths) if solution_episode_lengths else -1)
    print("  min_episode_length:     ", np.min(solution_episode_lengths) if solution_episode_lengths else -1)
    print()

    return output_h5_path


def remove_keys_from_h5(h5_path: str, ignore_keys: list[str]) -> None:
    """ This function removes specific keys from an h5 file and saves the result to a new file. H5 files have the 
    following structure:

        /                        Group
        /traj_0                  Group
        /traj_0/actions          Dataset {79, 8}
        /traj_0/env_states       Group
        /traj_0/env_states/actors Group
        /traj_0/env_states/actors/cube Dataset {80, 13}
        /traj_0/env_states/actors/table Dataset {80, 13}
        /traj_0/env_states/articulations Group
        /traj_0/env_states/articulations/panda Dataset {80, 31}
        /traj_0/obs              Group
        /traj_0/obs/agent        Group
        /traj_0/obs/agent/qpos   Dataset {80, 9}
        /traj_0/obs/agent/qvel   Dataset {80, 9}
        /traj_0/obs/agent/world__T__ee Dataset {80, 4, 4}
        /traj_0/obs/agent/world__T__root Dataset {80, 4, 4}
        /traj_0/obs/extra        Group
        /traj_0/obs/extra/is_grasped Dataset {80}
        /traj_0/obs/extra/tcp_pose Dataset {80, 7}
        /traj_1/
        /traj_1/actions 
        ...

    Ignore keys used the format: '/obs/extra'. Note that the keys are assumed to be relative to a given trajectory in 
    the h5 file.
    """
    if len(ignore_keys) == 0:
        return

    # Normalize ignore keys:
    # - allow either "/obs/extra" or "obs/extra"
    # - treat keys as paths relative to each "traj_*" group
    norm_ignore_keys: list[str] = []
    for k in ignore_keys:
        if k is None:
            continue
        k2 = str(k).strip()
        if not k2:
            continue
        k2 = k2.lstrip("/").rstrip("/")
        if k2:
            norm_ignore_keys.append(k2)

    if not norm_ignore_keys:
        return

    tmp_path = h5_path + ".tmp_pruned"
    if osp.exists(tmp_path):
        os.remove(tmp_path)

    def _copy_attrs(src_obj: h5py.Group | h5py.Dataset, dst_obj: h5py.Group | h5py.Dataset) -> None:
        for k, v in src_obj.attrs.items():
            dst_obj.attrs[k] = v

    def _should_prune(rel_path: str) -> bool:
        # Prune if this node is explicitly ignored.
        # Descendants are naturally pruned by never recursing into pruned groups.
        return rel_path in norm_ignore_keys

    def _copy_group_pruned(src_traj_group: h5py.Group, dst_traj_group: h5py.Group) -> None:
        """Copy a single traj group, excluding any paths matching norm_ignore_keys."""
        _copy_attrs(src_traj_group, dst_traj_group)

        def _recurse(src_group: h5py.Group, dst_group: h5py.Group, rel_prefix: str) -> None:
            for name, obj in src_group.items():
                child_rel = name if rel_prefix == "" else f"{rel_prefix}/{name}"
                if _should_prune(child_rel):
                    continue
                if isinstance(obj, h5py.Dataset):
                    src_group.file.copy(obj, dst_group, name=name)
                elif isinstance(obj, h5py.Group):
                    new_dst = dst_group.create_group(name)
                    _copy_attrs(obj, new_dst)
                    _recurse(obj, new_dst, child_rel)
                else:
                    raise ValueError(f"Unknown HDF5 object type: {type(obj)}")

        _recurse(src_traj_group, dst_traj_group, "")

    with h5py.File(h5_path, "r") as src, h5py.File(tmp_path, "w") as dst:
        _copy_attrs(src, dst)

        for top_name, top_obj in src.items():
            if isinstance(top_obj, h5py.Group) and top_name.startswith("traj_"):
                new_traj = dst.create_group(top_name)
                _copy_group_pruned(top_obj, new_traj)
            else:
                src.copy(top_obj, dst, name=top_name)

    os.replace(tmp_path, h5_path)



def main(args):
    if args.num_procs > 1 and args.num_procs < args.num_traj:
        if args.num_traj < args.num_procs:
            raise ValueError("Number of trajectories should be greater than or equal to number of processes")
        args.num_traj = args.num_traj // args.num_procs
        seeds = [*range(0, args.num_procs * args.num_traj, args.num_traj)]
        proc_args = [(deepcopy(args), i, seeds[i]) for i in range(args.num_procs)]

        # NOTE:
        # multiprocessing.Pool uses *daemon* workers, which cannot spawn child
        # processes. SAPIEN's coacd convex decomposition spawns a child process,
        # so we use ProcessPoolExecutor instead (workers are non-daemonic).
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.num_procs, mp_context=ctx) as ex:
            res = list(ex.map(_main, *(zip(*proc_args))))

        # Merge trajectory files
        output_path = res[0][: -len("0.h5")] + "h5"
        merge_trajectories(output_path, res)
        for h5_path in res:
            tqdm.write(f"Remove {h5_path}")
            os.remove(h5_path)
            json_path = h5_path.replace(".h5", ".json")
            tqdm.write(f"Remove {json_path}")
            os.remove(json_path)
    else:
        if args.random_seed:
            seed = np.random.randint(0, int(2**32-1))
        else:
            seed = 0
        output_path = _main(args, start_seed=seed)

    if args.ignore_keys is not None and len(args.ignore_keys) > 0:
        cprint(f"WARNING: Removing keys: {args.ignore_keys} from {output_path}", "yellow")
        remove_keys_from_h5(output_path, args.ignore_keys)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main(parse_args())
