ALGO_NAME = 'BC_ACT_rgbd'

import os
from dataclasses import dataclass
import random
from functools import partial
import numpy as np
import torch
from act.evaluate import evaluate


from act.make_env import make_eval_envs
from diffusers.training_utils import EMAModel
import tyro
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_rgbd import Agent, FlattenRGBDObservationWrapper, Args
from mani_skill.envs.tasks.tabletop.colosseum_v2.distraction_set import DISTRACTION_SETS
from mani_skill.envs.tasks.tabletop import *


"""

python examples/baselines/act/eval_rgbd.py \
    --checkpoint-path checkpoints/best_eval_success_once__BIMANUAL_JAN30.pt \
    --distraction-set "none" \
    --env-id "DualArmPickCube-v1" \
    --control-mode "pd_joint_pos" \
    --no-include-depth \
    --sim-backend "physx_cuda" \
    --capture-video \
    --num-eval-episodes 100 \
    --num-eval-envs 50 \
    --max-episode-steps 200
"""


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

    # env setup
    env_kwargs = dict(
        control_mode=args.control_mode, reward_mode="sparse", obs_mode="rgbd" if args.include_depth else "rgb", render_mode="rgb_array",
        distraction_set=DISTRACTION_SETS[args.distraction_set.upper()],
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
    print(f"Success rate: {100*(n_success / n_episodes):.2f}% \t ({n_success}/{n_episodes})")
    envs.close()

