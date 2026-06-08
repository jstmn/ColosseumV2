""" This script meaures the runtime of the ACT model for a given batch size. Note that a checkpoint isn't loaded.

# Example usage:

python examples/baselines/act_clip/measure_runtime.py \
    --env-id "RaiseCube-v1" \
    --num_eval_envs 1 \
    --control-mode "pd_joint_pos" \
    --no-include-depth \
    --sim-backend "physx_cuda" \
    --target-num-cams 3 \
    --perturbation-set "none"
"""

from time import time
import os
import torch
from act.make_env import make_eval_envs
import tyro
import sys
from functools import partial
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_rgbd import Agent, Args, FlattenRGBDObservationWrapper
from eval_rgbd import MAX_EPISODE_STEPS_BY_TASK
from mani_skill.envs.tasks.tabletop.colosseum_v2.perturbation_set import PERTURBATION_SETS
from mani_skill.envs.tasks.tabletop import *




if __name__ == "__main__":
    args = tyro.cli(Args)

    assert args.sim_backend in ("physx_cpu", "physx_cuda")

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env_kwargs = dict(
        control_mode=args.control_mode, reward_mode="sparse", obs_mode="rgbd" if args.include_depth else "rgb", render_mode="rgb_array",
        perturbation_set=PERTURBATION_SETS[args.perturbation_set.upper()],
        _env_id=args.env_id,
    )
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps
    other_kwargs = None

    wrappers = [partial(FlattenRGBDObservationWrapper, is_multi_task=args.is_multi_task, target_num_cams=args.target_num_cams, depth=args.include_depth)]
    envs = make_eval_envs(args.env_id, args.num_eval_envs, args.sim_backend, env_kwargs, other_kwargs, wrappers=wrappers)

    # agent setup
    agent = Agent(envs, args, is_multi_task=args.is_multi_task).to(device)

    obs, info = envs.reset()

    k = 50
    t0 = time()
    for i in range(k):
        action = agent.get_action(obs)
    tf = time()
    time_per_action_inference = (tf - t0) / k
    print(f"Average per action inference: {time_per_action_inference} seconds")

    NUM_VARIATIONS = 16
    n_timesteps_total = 0
    for task, max_n_steps in MAX_EPISODE_STEPS_BY_TASK.items():
        n_timesteps_total += max_n_steps * NUM_VARIATIONS

    n_seconds_total = n_timesteps_total * time_per_action_inference
    print(f"Estimated total inference time for all tasks (num-parallel-envs: {args.num_eval_envs}):")
    print(f"  - {round(n_seconds_total, 3)} seconds")
    print(f"  - {round(n_seconds_total / 60, 5)} minutes")
    print(f"  - {round(n_seconds_total / 3600, 5)} hours")

    # Colosseumv1: 20 tasks x 14 axes of environmental perturbations.
