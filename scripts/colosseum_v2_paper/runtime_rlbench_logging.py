import argparse
import os
import pandas as pd
from pathlib import Path


"""
This script logs t_elapsed_sec, n_processes to a csv file

Example usage:
python scripts/colosseum_v2_paper/runtime_rlbench_logging.py \
    --t_elapsed_sec 100 \
    --n_processes 10 \
    --results_filepath logs/fps/rlbench_fps.csv
"""



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t_elapsed_sec", type=float, required=True)
    parser.add_argument("--n_processes", type=int, required=True)
    parser.add_argument("--results_filepath", type=str, required=True)
    parser.add_argument("--timesteps_per_task", type=int, required=True)
    parser.add_argument("--num_eval_episodes", type=int, required=False)
    args = parser.parse_args()

    Path(args.results_filepath).parent.mkdir(parents=True, exist_ok=True)

    DF_COLS = ["batch_size", "t_elapsed_sec", "frames_per_second", "seconds_per_frame", "timesteps_per_task", "rlbench_estimated_total_sim_time_seconds", "rlbench_estimated_total_sim_time_minutes", "rlbench_estimated_total_sim_time_hours"]
    if os.path.exists(args.results_filepath):
        df = pd.read_csv(args.results_filepath)
        assert list(df.columns) == DF_COLS, f"CSV columns must be exactly {DF_COLS} (in order), got {list(df.columns)}"
    else:
        df = pd.DataFrame(columns=pd.Index(DF_COLS))


    if args.num_eval_episodes is not None:
        # total number of frames = num_eval_episodes * timesteps_per_task
        # total number of seconds = t_elapsed_sec
        # frames_per_second = total number of frames / total number of seconds
        frames_per_second = (args.num_eval_episodes * args.timesteps_per_task) / args.t_elapsed_sec
    else:
        frames_per_second = args.timesteps_per_task * args.n_processes / args.t_elapsed_sec
    seconds_per_frame = 1 / frames_per_second


    time_per_task_sec = args.t_elapsed_sec / args.n_processes
    NUM_VARIATIONS=17  # includes none, all
    N_TASKS=20
    total_sec = NUM_VARIATIONS * N_TASKS * time_per_task_sec
    print(f'Estimated total sim time: {total_sec:.0f}s  |  {total_sec/60:.1f} min  |  {total_sec/3600:.2f} hours')

    df.loc[len(df)] = {
        "batch_size": args.n_processes,
        "t_elapsed_sec": round(args.t_elapsed_sec, 3),
        "frames_per_second": round(frames_per_second, 3),
        "seconds_per_frame": round(seconds_per_frame, 3),
        "timesteps_per_task": args.timesteps_per_task,
        "rlbench_estimated_total_sim_time_seconds": round(total_sec, 3),
        "rlbench_estimated_total_sim_time_minutes": round(total_sec / 60, 3),
        "rlbench_estimated_total_sim_time_hours": round(total_sec / 3600, 3),
    }
    df.to_csv(args.results_filepath, index=False)
