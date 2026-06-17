ALGO_NAME = 'BC_ACT_rgbd'

import os
from dataclasses import dataclass
import random
from functools import partial
import numpy as np
import torch
from act.evaluate import evaluate
from pandas import read_csv, DataFrame

from act.make_env import make_eval_envs
from diffusers.training_utils import EMAModel
import tyro
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_rgbd import Agent, FlattenRGBDObservationWrapper, Args
from mani_skill.envs.tasks.tabletop.colosseum_v2.perturbation_set import PERTURBATION_SETS
from mani_skill.envs.tasks.tabletop import *


"""
# Run on a single task

"""

ALL_COLOSSEUM_V2_SINGLE_ARM_TASKS = (
    "RaiseCube-v1",
    "PickSodaFromCabinet-v1",
    "PickDishFromRack-v1",
    "StackCube-v1",
    "PlaceBookInShelf-v1",
    "PlaceDishInRack-v1",
    "LiftPegUpright-v1",
    "RotateArrow-v1",
    "PegInsertionSide-v2",
    "PlugCharger-v1",
    "HammerNail-v1",
    "ScoopBanana-v1",
    "OpenDrawer-v1",
    "OpenCabinet-v1",
    "PlaceCubeInDrawer-v1",
    "CookItemInPan-v1",
)

ALL_COLOSSEUM_V2_BIMANUAL_TASKS = (
    "DualArmPickCube-v1",
    "DualArmPickBottle-v1",
    "DualArmLiftPot-v1",
    "DualArmLiftTray-v1",
    "DualArmPushBox-v1",
    "DualArmPourPot-v1",
    "DualArmThreading-v1",
    "DualArmPenCap-v1",
    "DualArmDrawerPlace-v1",
    "DualArmDrawerOpen-v1",
    "DualArmStackCube-v1",
    "DualArmStack3Cube-v1",
)


def update_args_from_results(args: Args):
    assert args.results_path is not None
    expected_columns = [
        "checkpoint_path","perturbation_set","env_id","control_mode","include_depth","num_eval_episodes","max_episode_steps","message","num_sucessful_episodes","success_percent"
    ]
    results = read_csv(args.results_path)
    assert results.columns.tolist() == expected_columns

    if "bimanual" in args.results_path:
        tasks = ALL_COLOSSEUM_V2_BIMANUAL_TASKS
    elif "single_arm" in args.results_path:
        tasks = ALL_COLOSSEUM_V2_SINGLE_ARM_TASKS
    else:
        raise Exception(f"Unclear whether {args.results_path} is for bimanual or single arm tasks")

    for task in tasks:
        for perturbation_set in PERTURBATION_SETS.keys():
            result_found = results[
                (results["env_id"] == task)
                & (results["perturbation_set"].str.lower() == perturbation_set.lower())
            ]
            if len(result_found) > 0:
                print(f"Found existing result for task {task} and perturbation set {perturbation_set}")
                continue
            print(f"Starting evaluation for {task=} and {perturbation_set=}")
            args.env_id = task
            args.perturbation_set = perturbation_set

            row = [
                args.checkpoint_path,
                perturbation_set.lower(),
                task,
                args.control_mode,
                args.include_depth,
                args.num_eval_episodes,
                args.max_episode_steps,
                "placeholder",
                -1,
                -1,
            ]
            results.loc[len(results)] = row
            results.to_csv(args.results_path, index=False)
            return args

    raise Exception("No result found for any task and perturbation set")


if __name__ == "__main__":
    args = tyro.cli(Args)

    assert args.sim_backend in ("physx_cpu", "physx_cuda")
    assert args.checkpoint_path is not None
    assert os.path.exists(args.checkpoint_path)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and "cuda" in args.sim_backend else "cpu")

    if args.results_path is not None:
        args = update_args_from_results(args)

    # env setup
    env_kwargs = dict(
        control_mode=args.control_mode, reward_mode="sparse", obs_mode="rgbd" if args.include_depth else "rgb", render_mode="rgb_array",
        perturbation_set=PERTURBATION_SETS[args.perturbation_set.upper()],
    )
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps
    other_kwargs = None
    wrappers = [partial(FlattenRGBDObservationWrapper, depth=args.include_depth)]
    video_dir = args.checkpoint_path.replace('.pt', '__videos')
    envs = make_eval_envs(args.env_id, args.num_eval_envs, args.sim_backend, env_kwargs, other_kwargs, video_dir=video_dir if args.capture_video else None, wrappers=wrappers)
    obs_mode = "rgb+depth" if args.include_depth else "rgb"

    # agent setup
    agent = Agent(envs, args).to(device)
    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    ema_agent = Agent(envs, args).to(device)

    checkpoint = torch.load(args.checkpoint_path)
    agent.load_state_dict(checkpoint['agent'])
    ema_agent.load_state_dict(checkpoint['ema_agent'])
    stats = checkpoint['norm_stats']

    # Evaluation
    eval_kwargs = dict(
        stats=stats, num_queries=args.num_queries, temporal_agg=args.temporal_agg,
        max_timesteps=args.max_episode_steps, device=device, sim_backend=args.sim_backend
    )

    # ---------------------------------------------------------------------------- #
    # Training begins.
    # ---------------------------------------------------------------------------- #
    agent.eval()
    eval_metrics = evaluate(args.num_eval_episodes, ema_agent, envs, eval_kwargs)
    for metric, value in eval_metrics.items():
        print(f"{metric}: {value}")

    n_episodes = 0
    n_success = 0
    for episode_batch in eval_metrics["success_once"]:
        n_episodes += len(episode_batch)
        n_success += episode_batch.sum()
    success_percentage = 100*(n_success / n_episodes)
    print(f"Success rate: {success_percentage:.2f}% \t ({n_success}/{n_episodes})")
    envs.close()

    if args.results_path is not None:
        results = read_csv(args.results_path)
        new_row = [
            args.checkpoint_path,
            args.perturbation_set.lower(),
            args.env_id,
            args.control_mode,
            args.include_depth,
            args.num_eval_episodes,
            args.max_episode_steps,
            "results",
            n_success,
            f"{success_percentage:.2f}",
        ]
        results.loc[len(results)] = new_row
        results.to_csv(args.results_path, index=False)
        print(f"Saved results to {args.results_path}")