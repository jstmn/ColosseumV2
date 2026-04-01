import argparse
import psutil
from time import time, sleep
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import rlbench
import gymnasium as gym
import multiprocessing as mp
from typing import Any
from rlbench.action_modes.action_mode import JointPositionActionMode
import numpy as np

import sys
sys.path.append("examples/baselines/act_clip")
from eval_rgbd import MAX_EPISODE_STEPS_BY_TASK


"""
This script measures maniskill's FPS for difference batch sizes. Note that you need to run each environment in a 
separate process.


###### How to install rlbench:
mkdir thirdparty/
# Download, install Coppelia Sim. These instructions assume you are using Ubuntu 20. There are __no__ instructions for Ubuntu 22 or 24 on the PyRep github (https://github.com/stepjam/PyRep). I recommend opening an issue there if you need to run this but are on 22/24.
curl -L -o thirdparty/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz https://www.coppeliarobotics.com/files/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
tar -C thirdparty/ -xf thirdparty/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
echo "" >> ~/.bashrc
echo "export COPPELIASIM_ROOT=$(realpath thirdparty/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04/)" >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT' >> ~/.bashrc
echo 'export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT' >> ~/.bashrc
source ~/.bashrc
pip install git+https://github.com/stepjam/RLBench.git

# Example usage:
python scripts/colosseum_v2_paper/runtime_rlbench.py \
    --results_filepath "logs/fps/rlbench_fps.csv" --batch_size 10

"""


def _measure_fps_subprocess(raw_results, process_id: int, env_id: str, n_steps: int):
    print(f"Starting process {process_id}")
    env = gym.make(
        env_id,
        action_mode=JointPositionActionMode(),
    )
    env.reset()

    raw_results[process_id] = {
        "started": True,
        "timing_data": None
    }
    timing_data = {i: 0.0 for i in range(n_steps)}

    for i in range(n_steps):
        env.step(env.action_space.sample())
        # This dict setting is counted as part of the computation time which is slightly unfair, but in practice it's
        # runs in microseconds so it's negligible compared to the 10+ seconds from RLBench.
        timing_data[i] = time()

    raw_results[process_id] = {"started": True, "timing_data": timing_data}
    env.close()
    print(f"process {process_id} finished")



def measure_runtime_fps(
    results_filepath: str,
    batch_size: int,
    n_steps: int,
    env_id: str = "rlbench/slide_block_to_target-vision-v0",
):

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    raw_results = manager.dict()

    procs: list[tuple[int, Any]] = []
    for process_id in range(batch_size):
        p = ctx.Process(
            target=_measure_fps_subprocess,
            args=(raw_results, process_id, env_id, n_steps),
            daemon=False,
        )
        p.start()
        procs.append((process_id, p))

    # measure cpu usage
    while True:
        for process_id, result in raw_results.items():
            if "started" in result and not result["started"]:
                sleep(0.01)
                print("waiting for process ", process_id, " to start")
                continue
        print("all processes started")
        break
    usage = psutil.cpu_percent(percpu=True, interval=1.0)
    print("cpu usage: ", usage)

    #
    for process_id, p in procs:
        p.join()


    # We need to do something tricky here. We only want to measure the fps when all environments are running at the same
    # time. To do this, we need to find the minimum t0 and the maximum t1 where between these two timestamps, all
    # environments are running. After doing this, we need to calculate how many steps were taken in each environment.
    min_t0 = float('inf')
    max_t1 = float('-inf')
    for process_id, result in raw_results.items():
        assert result["started"], f"process {process_id} did not start"
        assert result["timing_data"] is not None, f"process {process_id} did not return timing data"
        timing_data = result["timing_data"]
        min_t0 = min(min_t0, min(timing_data.values()))
        max_t1 = max(max_t1, max(timing_data.values()))
    print("min_t0: ", min_t0, "max_t1: ", max_t1, "time_delta: ", max_t1 - min_t0)
    assert min_t0 < max_t1

    n_steps_taken = n_steps * batch_size
    fps = n_steps_taken / (max_t1 - min_t0)
    seconds_per_frame = 1 / fps

    # Colosseumv1: 20 tasks x 17 axes of environmental perturbations.
    NUM_VARIATIONS = 17  # includes none, all
    N_TASKS = 20
    TIMESTEPS_PER_TASK = 200
    estimated_total_sim_time_seconds = N_TASKS * NUM_VARIATIONS * TIMESTEPS_PER_TASK * seconds_per_frame

    print()
    print("New data:")
    print(f"  - batch_size:         {batch_size}")
    print(f"  - frames_per_second:  {fps}")
    print(f"  - seconds_per_frame:  {seconds_per_frame}")
    print(f"  - per_core_cpu_usage: {usage}")
    print(f"  - average_cpu_usage:  {np.mean(usage)}")
    print(f"  - estimated_total_sim_time_sec: {estimated_total_sim_time_seconds}")
    print(f"  - estimated_total_sim_time_min: {estimated_total_sim_time_seconds / 60}")
    print(f"  - estimated_total_sim_time_hours: {estimated_total_sim_time_seconds / 3600}")
    print()

    DF_COLS = ["batch_size", "frames_per_second", "seconds_per_frame", "per_core_cpu_usage", "average_cpu_usage", "estimated_total_sim_time_seconds", "estimated_total_sim_time_minutes", "estimated_total_sim_time_hours"]
    if os.path.exists(results_filepath):
        df = pd.read_csv(results_filepath)
        assert list(df.columns) == DF_COLS, f"CSV columns must be exactly {DF_COLS} (in order), got {list(df.columns)}"
    else:
        df = pd.DataFrame(columns=pd.Index(DF_COLS))


    df.loc[len(df)] = {
        "batch_size": batch_size,
        "frames_per_second": fps,
        "seconds_per_frame": seconds_per_frame,
        "per_core_cpu_usage": usage,
        "average_cpu_usage": np.mean(usage),
        "estimated_total_sim_time_seconds": estimated_total_sim_time_seconds,
        "estimated_total_sim_time_minutes": estimated_total_sim_time_seconds / 60,
        "estimated_total_sim_time_hours": estimated_total_sim_time_seconds / 3600,
    }
    df.to_csv(results_filepath, index=False)
    print(df)
    print(f"Saved to {results_filepath}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_filepath", type=str, required=True)
    parser.add_argument("--env_id", type=str, default="rlbench/slide_block_to_target-vision-v0")
    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, required=True)
    args = parser.parse_args()
    Path(args.results_filepath).parent.mkdir(parents=True, exist_ok=True)

    measure_runtime_fps(args.results_filepath, args.batch_size, env_id=args.env_id, n_steps=args.n_steps)