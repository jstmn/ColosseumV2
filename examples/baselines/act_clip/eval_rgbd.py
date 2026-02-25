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
from mani_skill.envs.tasks.tabletop.colosseum_v2.distraction_set import DISTRACTION_SETS
from mani_skill.envs.tasks.tabletop import *


"""
# Run on a single task
python examples/baselines/act_clip/eval_rgbd.py \
    --checkpoint-path checkpoints/hyeonho_simul_results/Multi-task_single_lang/best_eval_success_once.pt \
    --distraction-set "distractor_object" \
    --env-id "OpenDrawer-v1" \
    --control-mode "pd_ee_delta_pose" \
    --no-include-depth \
    --sim-backend "physx_cuda" \
    --is-multi-task True \
    --target-num-cams 1 \
    --num-eval-episodes 100 \
    --num-eval-envs 10 \
    --max-episode-steps 200 \
    --hidden-dim 512 --dim-feedforward 1600 --enc-layers 4 --dec-layers 7 \
    --internal-instruction --capture-video # <- gpu intensive, so won't work past a certain number of environments

# Run on all tasks x variation factors
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
    --hidden-dim 512 --dim-feedforward 1600 --enc-layers 4 --dec-layers 7 \
    --internal-instruction \
    --distraction-set "BLANK" \
    --results-path logs/results_single_arm.csv
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


def update_args_from_results(args: Args):
    assert args.results_path is not None
    expected_columns = [
        "checkpoint_path","distraction_set","env_id","control_mode","include_depth","num_eval_episodes","max_episode_steps","message","num_sucessful_episodes","success_percent"
    ]
    results = read_csv(args.results_path)
    assert results.columns.tolist() == expected_columns

    if "bimanual" in args.results_path:
        tasks = ALL_COLOSSEUM_V2_BIMANUAL_TASKS
        print("Evaluating bimanual tasks")
        assert args.control_mode == "pd_joint_pos", f"The control_mode should be pd_joint_pos for bimanual tasks"
    elif "single_arm" in args.results_path:
        tasks = ALL_COLOSSEUM_V2_SINGLE_ARM_TASKS
        print("Evaluating single arm tasks")
    else:
        raise Exception(f"Unclear whether {args.results_path} is for bimanual or single arm tasks")

    for task in tasks:
        for distraction_set in DISTRACTION_SETS.keys():
            result_found = results[
                (results["env_id"] == task)
                & (results["distraction_set"].str.lower() == distraction_set.lower())
            ]
            if len(result_found) > 0:
                print(f"Found existing result for task {task} and distraction set {distraction_set}")
                continue
            print(f"Starting evaluation for {task=} and {distraction_set=}")
            args.env_id = task
            args.distraction_set = distraction_set

            row = [
                args.checkpoint_path,
                distraction_set.lower(),
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

    raise Exception("No result found for any task and distraction set")


if __name__ == "__main__":
    args = tyro.cli(Args)

    assert args.sim_backend in ("physx_cpu", "physx_cuda")
    assert args.checkpoint_path is not None
    assert os.path.exists(args.checkpoint_path), f"Checkpoint not found: {args.checkpoint_path}"
    assert args.is_multi_task is not None, "is_multi_task must be set for evaluation"
    assert args.target_num_cams is not None, "target_num_cams must be set for evaluation"

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
        distraction_set=DISTRACTION_SETS[args.distraction_set.upper()].to_dict(),
    )
    # ^ distraction_set needs to be pickle-able by ManiSkillVectorEnv, so we convert it to a dictionary
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps
    other_kwargs = None
    wrappers = [partial(FlattenRGBDObservationWrapper, is_multi_task=args.is_multi_task, target_num_cams=args.target_num_cams, depth=args.include_depth)]
    video_dir = args.checkpoint_path.replace('.pt', '__videos')
    envs = make_eval_envs(args.env_id, args.num_eval_envs, args.sim_backend, env_kwargs, other_kwargs, video_dir=video_dir if args.capture_video else None, wrappers=wrappers)
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

    #TODO: should be give lanaugage instructions for each task
    eid = args.env_id
    print(f"Evaluating task: {eid}")
    print(f"with language instruction: {TASK_TEXT_MAP[eid]}")
    eval_lang = TASK_TEXT_MAP[eid] if args.internal_instruction else None

    eval_metrics = evaluate(args.num_eval_episodes, ema_agent, envs, eval_kwargs, lang_instruction=eval_lang)
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
            args.distraction_set.lower(),
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