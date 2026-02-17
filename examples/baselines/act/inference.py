import torch
import numpy as np
import os
from act.make_env import make_eval_envs
from act.evaluate import evaluate
from functools import partial
import tyro
from dataclasses import dataclass
import train_rgbd 
from train_rgbd import Agent, Args, FlattenRGBDObservationWrapper

@dataclass
class EvalArgs(Args):
    checkpoint_path: str = "runs/default/checkpoints/best_eval_success_at_end.pt"
    output_dir: str = "./inference_videos"
    num_eval_episodes: int = 10
    num_eval_envs: int = 1
    distraction_set: str = "none"

def run_inference():
    global train_rgbd 
    
    eval_args = tyro.cli(EvalArgs)
    
    train_rgbd.args = eval_args 
    
    device = torch.device("cuda" if torch.cuda.is_available() and eval_args.cuda else "cpu")


    if not os.path.exists(eval_args.checkpoint_path):
        raise FileNotFoundError(f"Not Found checkpoint: {eval_args.checkpoint_path}")

    print(f"[*] Loading checkpoint: {eval_args.checkpoint_path}")
    checkpoint = torch.load(eval_args.checkpoint_path, map_location=device)
    norm_stats = checkpoint.get('norm_stats', None)
    

    print(f"[*] Creating environment: {eval_args.env_id}")
    from mani_skill.envs.distraction_set import DISTRACTION_SETS
    
    env_kwargs = dict(
        control_mode=eval_args.control_mode, 
        reward_mode="sparse", 
        obs_mode="rgbd" if eval_args.include_depth else "rgb",
        render_mode="rgb_array",
        distraction_set=DISTRACTION_SETS[eval_args.distraction_set.upper()],
    )
    
    if eval_args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = eval_args.max_episode_steps

    other_kwargs = None
    wrappers = [partial(FlattenRGBDObservationWrapper, depth=eval_args.include_depth)]
    
    eval_envs = make_eval_envs(
        eval_args.env_id, 
        eval_args.num_eval_envs, 
        eval_args.sim_backend, 
        env_kwargs, 
        other_kwargs,
        video_dir=eval_args.output_dir, 
        wrappers=wrappers
    )


    train_rgbd.envs = eval_envs 
    agent = Agent(eval_envs, eval_args).to(device)
    
    if 'ema_agent' in checkpoint:
        print("[+] Using EMA weights.")
        agent.load_state_dict(checkpoint['ema_agent'])
    else:
        print("[!] Using standard agent weights.")
        agent.load_state_dict(checkpoint['agent'])
    
    agent.eval()

    if norm_stats is not None:
        eval_stats = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in norm_stats.items()}
    else:
        eval_stats = None
        
    save_name = f"eval_{eval_args.env_id}_{eval_args.num_eval_episodes}eps"
        
    eval_kwargs = dict(
        stats=eval_stats,
        num_queries=eval_args.num_queries,
        temporal_agg=eval_args.temporal_agg,
        max_timesteps=eval_args.max_episode_steps if eval_args.max_episode_steps else 200,
        device=device,
        sim_backend=eval_args.sim_backend,
    )


    print(f"[*] Starting evaluation...")


    metrics = evaluate(
        eval_args.num_eval_episodes, # n
        agent,                       # agent
        eval_envs,                   # eval_envs
        eval_kwargs,                 # eval_kwargs
    )

    print("\n" + "="*40)
    print(f"📊 Results for {eval_args.env_id}")
    for k, v in metrics.items():
        if isinstance(v, (list, np.ndarray)):
            print(f"{k:25s}: {np.mean(v):.4f}")
    print("="*40)

    eval_envs.close()

if __name__ == "__main__":
    run_inference()