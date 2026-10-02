import csv
import os
from time import time
from termcolor import cprint
from pathlib import Path
import random
from functools import partial
import numpy as np
import torch
from act.evaluate import evaluate
from pandas import read_csv, DataFrame
from datetime import datetime
import socket
from act.make_env import make_eval_envs
from diffusers.training_utils import EMAModel
import tyro
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_rgbd import Agent, FlattenRGBDObservationWrapper, Args
from mani_skill.envs.tasks.tabletop.colosseum_v2.perturbation_set import PERTURBATION_SETS
from mani_skill.envs.tasks.tabletop.colosseum_v2.colosseum_v2_core import PerturbationFactorDisabledError
from mani_skill.envs.tasks.tabletop import *


"""
# Run on a single, single-arm task
python examples/baselines/act_clip/eval_rgbd.py \
    --checkpoint-path checkpoints/hyeonho_simul_results/Multi-task_single_lang/best_eval_success_once.pt \
    --perturbation-set "light_color" \
    --env-id "RaiseCube-v1" \
    --control-mode "pd_ee_delta_pose" \
    --no-include-depth \
    --sim-backend "physx_cuda" \
    --is-multi-task True \
    --target-num-cams 1 \
    --num-eval-episodes 100 \
    --num-eval-envs 50 \
    --max-episode-steps 350 \
    --internal-instruction


# Run on a single task and save video (single arm)
python examples/baselines/act_clip/eval_rgbd.py \
    --checkpoint-path checkpoints/hyeonho_simul_results/Multi-task_single_lang/best_eval_success_once.pt \
    --perturbation-set "none" \
    --env-id "PlaceCubeInDrawer-v1" \
    --control-mode "pd_ee_delta_pose" \
    --no-include-depth \
    --sim-backend "physx_cuda" \
    --is-multi-task True \
    --target-num-cams 1 \
    --num-eval-episodes 12 --num-eval-envs 12 --max-episode-steps 200 \
    --internal-instruction --capture-video
    # --capture-video is gpu intensive, so need to limit the number of environments


# Run on a single task and save video (bimanual)
python examples/baselines/act_clip/eval_rgbd.py \
    --checkpoint-path checkpoints/hyeonho_simul_results/Multi-task_bimanual_lang/best_eval_success_once.pt \
    --perturbation-set "none" \
    --env-id "DualArmDrawerOpen-v1" \
    --control-mode "pd_joint_pos" \
    --no-include-depth \
    --sim-backend "physx_cuda" \
    --is-multi-task True \
    --target-num-cams 1 \
    --internal-instruction \
    --num-eval-episodes 6 --num-eval-envs 6 --max-episode-steps 500 --capture-video 
    # --capture-video is gpu intensive, so need to limit the number of environments



# Run on all tasks x perturbation factors
python examples/baselines/act_clip/eval_rgbd.py \
    --checkpoint-path checkpoints/hyeonho_simul_results/Multi-task_single_lang/best_eval_success_once.pt \
    --control-mode "pd_ee_delta_pose" \
    --no-include-depth \
    --sim-backend "physx_cuda" \
    --is-multi-task True \
    --target-num-cams 1 \
    --num-eval-episodes 100 \
    --num-eval-envs 50 \
    --max-episode-steps 200 \
    --internal-instruction \
    --perturbation-set "BLANK" \
    --results-path logs/results_single_arm.csv
"""

class OutOfTasksError(Exception):
    pass

ALL_COLOSSEUM_V2_SINGLE_ARM_TASKS = (
    "RaiseCube-v1",
    "PickSodaFromCabinet-v1",
    "PickDishFromRack-v1",
    "StackCubeColosseumV2-v1",
    "PlaceBookInShelf-v1",
    "PlaceDishInRack-v1",
    "LiftPegUprightColosseumV2-v1",
    "RotateArrow-v1",
    "PegInsertionSideColosseumV2-v1",
    "PlugChargerColosseumV2-v1",
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



#Language instructions maaping for each task
TASK_TEXT_MAP = {
    #Single_arm
    "PickSodaFromCabinet-v1": "pick up the soda can from the cabinet",
    "PlaceBookInShelf-v1": "place the book into the bookshelf",
    "HammerNail-v1": "use the hammer to hit the nail into the wood",
    "ScoopBanana-v1": "scoop the banana and move it to the target location",
    "OpenDrawer-v1": "grasp the handle and pull the drawer open",
    "OpenCabinet-v1": "grasp the handle and open the cabinet door",
    "PlaceCubeInDrawer-v1": "place the cube inside the open drawer",
    "CookItemInPan-v1": "place the food item in the pan for cooking",
    "LiftPegUpright-v1": "lift the peg and stand it upright on the table",
    "PegInsertionSide-v2": "insert the peg into the hole from the side",
    "PickDishFromRack-v1": "pick up a dish from the drying rack",
    "PlaceDishInRack-v1": "place the dish into the drying rack",
    "PlugCharger-v1": "plug the charger into the wall outlet",
    "RaiseCube-v1": "lift the cube up to a certain height",
    "RotateArrow-v1": "rotate the arrow lever to the target direction",
    "StackCube-v1": "stack red cube on top of green cube",
    "StackCubeColosseumV2-v1": "stack red cube on top of green cube",
    "LiftPegUprightColosseumV2-v1": "lift the peg and stand it upright on the table",
    "PegInsertionSideColosseumV2-v1": "insert the peg into the hole from the side",
    "PlugChargerColosseumV2-v1": "plug the charger into the wall outlet",

    #Bimanual
    "DualArmPickCube-v1": "use both arms to pick up the cube",
    "DualArmPickBottle-v1": "use both arms to pick up the bottle",
    "DualArmLiftPot-v1": "grasp the pot with both hands and lift it up",
    "DualArmLiftTray-v1": "grasp the tray with both hands and lift it up",
    "DualArmPushBox-v1": "use both hands to push the box to the target location",
    "DualArmPourPot-v1": "lift the pot with both hands and pour its contents into the target container",
    "DualArmThreading-v1": "use both hands to thread the object through the target opening",
    "DualArmPenCap-v1": "hold the pen with one hand and remove the cap with the other hand",
    "DualArmDrawerPlace-v1": "open the drawer with one hand and place the object inside with the other hand",
    "DualArmDrawerOpen-v1": "use both hands to grasp and pull the drawer open",
    "DualArmStackCube-v1": "use both hands to stack one cube on top of another",
    "DualArmStack3Cube-v1": "use both hands to stack three cubes into a tower",

}



def get_now_str():
    return datetime.now().strftime("%Y:%m:%d__%H:%M:%S")


VIDEO_OUTCOMES_CSV_COLUMNS = (
    "filepath",
    "env_id",
    "perturbation_set",
    "episode_idx",
    "outcome",
)


def append_video_outcomes_csv(
    video_dir: str,
    video_filename: str,
    env_id: str,
    perturbation_set: str,
    success_once: np.ndarray,
) -> Path:
    """Append one row per recorded episode: video path ↔ success/fail.

    RecordEpisode names CPU videos ``{video_filename}__{episode_idx}.mp4``
    in order, matching the flattened ``success_once`` from evaluate().
    """
    video_dir_path = Path(video_dir)
    video_dir_path.mkdir(parents=True, exist_ok=True)
    csv_path = video_dir_path / "video_outcomes.csv"
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    success_once = np.asarray(success_once).reshape(-1)
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=VIDEO_OUTCOMES_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        for ep_idx, ok in enumerate(success_once):
            stem = f"{video_filename}__{ep_idx}".replace(" ", "_").replace("\n", "_")
            filepath = str((video_dir_path / f"{stem}.mp4").resolve())
            writer.writerow(
                {
                    "filepath": filepath,
                    "env_id": env_id,
                    "perturbation_set": perturbation_set,
                    "episode_idx": ep_idx,
                    "outcome": "success" if bool(ok) else "fail",
                }
            )
    return csv_path


RESULTS_CSV_COLUMNS = [
    "checkpoint_path","pc_hostname","now","t_final","duration_sec","perturbation_set","env_id","control_mode","include_depth","num_eval_episodes","max_episode_steps","message","num_sucessful_episodes","success_percent"
]


def get_tasks_and_perturbations(args: Args) -> tuple[list[str], list[str]]:
    """Return the task list and perturbation list used for sweep scheduling."""
    assert args.results_path is not None
    if "bimanual" in args.results_path:
        tasks = list(ALL_COLOSSEUM_V2_BIMANUAL_TASKS)
    elif ("single_arm" in args.results_path) or ("singlearm" in args.results_path):
        tasks = list(ALL_COLOSSEUM_V2_SINGLE_ARM_TASKS)
    else:
        raise Exception(f"Unclear whether {args.results_path} is for bimanual or single arm tasks")

    if len(args.tasks_subset) > 0:
        tasks = [task for task in tasks if task in args.tasks_subset]
        assert len(tasks) > 0, f"No tasks found in {args.tasks_subset} after filtering by args.tasks_subset: {args.tasks_subset}"

    if len(args.perturbation_factors_subset) > 0:
        perturbation_factors = list(args.perturbation_factors_subset)
    else:
        perturbation_factors = list(PERTURBATION_SETS.keys())
    return tasks, perturbation_factors


def load_results_df(results_path: str) -> DataFrame:
    if not Path(results_path).exists():
        return DataFrame(columns=RESULTS_CSV_COLUMNS)
    results_df = read_csv(results_path)
    assert results_df.columns.tolist() == RESULTS_CSV_COLUMNS
    return results_df


def finalize_results_row(
    args: Args,
    *,
    duration_sec: float,
    message: str,
    n_episodes: int,
    n_success: int | float,
    success_percent,
) -> None:
    """Replace the placeholder for this (env, perturbation) with a final row.

    Falls back to append if no placeholder is found.
    """
    assert args.results_path is not None
    results_df = load_results_df(args.results_path)
    pert = args.perturbation_set.lower()
    env_id = args.env_id
    placeholder_mask = (
        (results_df["env_id"] == env_id)
        & (results_df["perturbation_set"].astype(str).str.lower() == pert)
        & (results_df["message"].astype(str).str.lower() == "placeholder")
    )
    row = {
        "checkpoint_path": args.checkpoint_path,
        "pc_hostname": args.pc_hostname,
        "now": args.now,
        "t_final": get_now_str(),
        "duration_sec": duration_sec,
        "perturbation_set": pert,
        "env_id": env_id,
        "control_mode": args.control_mode,
        "include_depth": args.include_depth,
        "num_eval_episodes": n_episodes,
        "max_episode_steps": args.max_episode_steps,
        "message": message,
        "num_sucessful_episodes": n_success,
        "success_percent": success_percent,
    }
    if placeholder_mask.any():
        idx = results_df.index[placeholder_mask][-1]
        for col, val in row.items():
            results_df.at[idx, col] = val
    else:
        results_df = DataFrame([*results_df.to_dict("records"), row], columns=RESULTS_CSV_COLUMNS)
    results_df.to_csv(args.results_path, index=False)
    print(f"Saved results_df to {args.results_path} (message={message!r})")


def get_remaining_eval_pairs(args: Args, results_df: DataFrame | None = None) -> tuple[list[tuple[str, str]], int]:
    """Return unfinished (task, perturbation) pairs and total planned pairs."""
    assert args.results_path is not None
    tasks, perturbation_factors = get_tasks_and_perturbations(args)
    n_total = len(tasks) * len(perturbation_factors)
    if results_df is None:
        results_df = load_results_df(args.results_path)

    done = set()
    if len(results_df) > 0:
        for env_id, pert in zip(results_df["env_id"], results_df["perturbation_set"]):
            done.add((str(env_id), str(pert).lower()))

    remaining = [
        (task, perturbation_set)
        for task in tasks
        for perturbation_set in perturbation_factors
        if (task, perturbation_set.lower()) not in done
    ]
    return remaining, n_total


def update_args_from_results_csv(args: Args):
    assert args.results_path is not None
    results_df = load_results_df(args.results_path)

    if "bimanual" in args.results_path:
        print("Evaluating bimanual tasks")
        assert args.control_mode == "pd_joint_pos", f"The control_mode should be pd_joint_pos for bimanual tasks"
    elif ("single_arm" in args.results_path) or ("singlearm" in args.results_path):
        print("Evaluating single arm tasks")
    else:
        raise Exception(f"Unclear whether {args.results_path} is for bimanual or single arm tasks")

    args.now = get_now_str()
    args.pc_hostname = socket.gethostname()

    remaining, n_total = get_remaining_eval_pairs(args, results_df=results_df)
    tasks, perturbation_factors = get_tasks_and_perturbations(args)
    print("Considering perturbation factors: ", perturbation_factors)
    print("Considering tasks: ", tasks)
    print(f"Remaining eval pairs: {len(remaining)}/{n_total}")

    if len(remaining) == 0:
        raise OutOfTasksError("No result found for any task and perturbation set")

    task, perturbation_set = remaining[0]
    cprint(f"Starting evaluation for '{task}' with '{perturbation_set}'", "green")
    args.env_id = task
    args.perturbation_set = perturbation_set

    row = {
        "checkpoint_path": args.checkpoint_path,
        "pc_hostname": args.pc_hostname,
        "now": args.now,
        "t_final": "final-time-not-set",
        "duration_sec": -1,
        "perturbation_set": perturbation_set.lower(),
        "env_id": task,
        "control_mode": args.control_mode,
        "include_depth": args.include_depth,
        "num_eval_episodes": args.num_eval_episodes,
        "max_episode_steps": args.max_episode_steps,
        "message": "placeholder",
        "num_sucessful_episodes": -1,
        "success_percent": -1,
    }
    results_df = DataFrame([*results_df.to_dict("records"), row], columns=RESULTS_CSV_COLUMNS)
    results_df.to_csv(args.results_path, index=False)
    return args


if __name__ == "__main__":
    args = tyro.cli(Args)

    assert args.sim_backend in ("physx_cpu", "physx_cuda")
    assert args.checkpoint_path is not None
    assert os.path.exists(args.checkpoint_path), f"Checkpoint not found: {args.checkpoint_path}"
    assert args.is_multi_task is not None, "is_multi_task must be set for evaluation"
    assert args.target_num_cams is not None, "target_num_cams must be set for evaluation"
    t0 = time()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and "cuda" in args.sim_backend else "cpu")

    # Claim pairs until we successfully create an env (skip disabled variations in-process).
    envs = None
    while envs is None:
        if args.results_path is not None:
            try:
                args = update_args_from_results_csv(args)
            except OutOfTasksError as e:
                cprint(f"SUCCESS: {e}. Exiting.", "green")
                exit(0)

        if args.max_episode_steps_from_lookup:
            if args.env_id in MAX_EPISODE_STEPS_BY_TASK:
                args.max_episode_steps = MAX_EPISODE_STEPS_BY_TASK[args.env_id]
            else:
                cprint(
                    f"--max-episode-steps-from-lookup is enabled but environment {args.env_id} not found in "
                    f"MAX_EPISODE_STEPS_BY_TASK. Using user-provided max episode steps: {args.max_episode_steps}",
                    "yellow",
                )

        # env setup
        env_kwargs = dict(
            control_mode=args.control_mode,
            reward_mode="sparse",
            obs_mode="rgbd" if args.include_depth else "rgb",
            render_mode="rgb_array" if args.capture_video else None,
            human_render_shader=args.human_render_shader,
            perturbation_set=PERTURBATION_SETS[args.perturbation_set.upper()].to_dict(),
            _env_id=args.env_id,
        )
        # ^ perturbation_set needs to be pickle-able by ManiSkillVectorEnv, so we convert it to a dictionary
        if args.max_episode_steps is not None:
            env_kwargs["max_episode_steps"] = args.max_episode_steps
        other_kwargs = None
        wrappers = [
            partial(
                FlattenRGBDObservationWrapper,
                is_multi_task=args.is_multi_task,
                target_num_cams=args.target_num_cams,
                depth=args.include_depth,
            )
        ]
        video_dir = args.checkpoint_path.replace(".pt", "__videos")
        video_filename = f"{args.env_id}___ds:{args.perturbation_set}"
        try:
            envs = make_eval_envs(
                args.env_id,
                args.num_eval_envs,
                args.sim_backend,
                env_kwargs,
                other_kwargs,
                video_dir=video_dir if args.capture_video else None,
                wrappers=wrappers,
                video_filename=video_filename,
                info_on_video=args.metrics_on_video,
            )
        except PerturbationFactorDisabledError as e:
            cprint(f"Variation factor disabled error: {e}", "red")
            if args.results_path is None:
                raise
            # Replace placeholder with a permanent skip, then try the next pair.
            finalize_results_row(
                args,
                duration_sec=time() - t0,
                message="variation_factor_disabled",
                n_episodes=0,
                n_success=-1,
                success_percent=-1,
            )
            remaining, n_total = get_remaining_eval_pairs(args)
            cprint(f"Remaining eval pairs after skip: {len(remaining)}/{n_total}", "cyan")
            continue
    obs_mode = "rgb+depth" if args.include_depth else "rgb"

    # agent setup
    agent = Agent(envs, args, is_multi_task=args.is_multi_task).to(device)
    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    ema_agent = Agent(envs, args, is_multi_task=args.is_multi_task).to(device)

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

    # Update language instructions for each environment
    eid = args.env_id
    print(f"Evaluating task: {eid} with language instruction: {TASK_TEXT_MAP[eid]}")
    eval_langs = None
    if args.internal_instruction:
        eval_langs = [TASK_TEXT_MAP[eid]] * args.num_eval_envs
    try:
        eval_langs = envs.unwrapped.update_language_instructions(eval_langs)
        if len(eval_langs) > 10:
            cprint(f"Using updated language instruction: {eval_langs[:10]}... (truncated)", "yellow")
        else:
            cprint(f"Using updated language instruction: {eval_langs}", "yellow")
    except AttributeError as e:
        cprint(f"Environment doesn't support perturbed language instructions: {e}", "yellow")
    except Exception as e:
        raise e

    eval_metrics = evaluate(args.num_eval_episodes, ema_agent, envs, eval_kwargs, lang_instructions=eval_langs)
    for metric, value in eval_metrics.items():
        print(f"{metric}: {value}")

    n_episodes = 0
    n_success = 0
    # With num_envs=1, stacked success_once is shape (n_episodes,) of scalars;
    # with num_envs>1 it is shape (n_chunks, num_envs). Flatten either case.
    success_once = np.asarray(eval_metrics["success_once"]).reshape(-1)
    n_episodes = int(success_once.size)
    n_success = int(success_once.sum())
    success_percentage = 100*(n_success / n_episodes) if n_episodes else 0.0
    print(f"Success rate: {success_percentage:.2f}% \t ({n_success}/{n_episodes})")

    if args.capture_video:
        outcomes_csv = append_video_outcomes_csv(
            video_dir=video_dir,
            video_filename=video_filename,
            env_id=args.env_id,
            perturbation_set=args.perturbation_set,
            success_once=success_once,
        )
        print(f"Video outcomes appended to: {outcomes_csv}")

    envs.close()

    if args.results_path is not None:
        duration_sec = time() - t0
        finalize_results_row(
            args,
            duration_sec=duration_sec,
            message="results_df",
            n_episodes=n_episodes,
            n_success=n_success,
            success_percent=f"{success_percentage:.5f}",
        )

        remaining, n_total = get_remaining_eval_pairs(args)
        eta_hours = len(remaining) * duration_sec / 3600.0
        cprint(
            f"Estimated time remaining: {eta_hours:.2f} hours "
            f"({len(remaining)}/{n_total} pairs left × {duration_sec:.1f}s this run)",
            "cyan",
        )

