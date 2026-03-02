import argparse
from time import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import gymnasium as gym
from mani_skill.envs.tasks.tabletop.colosseum_v2.raise_cube import RaiseCubeEnv

"""
This script measures maniskill's FPS for difference batch sizes.

# Example usage:
python scripts/generate_colosseum_v2_runtime_results.py \
    --results_filepath "logs/maniskill_fps.csv" --batch_sizes 1 2 4 8 16 32 64 128 256 512 1024 2048 4096
"""



def measure_runtime_fps(results_filepath: str, batch_sizes: list[int], env_id: str = "RaiseCube-v1"):

    rows: list[dict] = []

    for batch_size in batch_sizes:
        env = gym.make(
            env_id,
            obs_mode="rgbd",
            control_mode="pd_joint_pos",
            render_mode="none",
            reward_mode="none",
            sim_backend="physx_cuda",
            distraction_set=None,
            num_envs=batch_size,
            _env_id=env_id
        )
        obs, _ = env.reset()
        print(obs["sensor_data"]["base_camera"].keys())
        assert obs
        t0 = time()
        n_steps = 100
        for _ in range(n_steps):
            env.step(env.action_space.sample())
        t1 = time()
        fps = (batch_size * n_steps) / (t1 - t0)
        row = {"batch_size": batch_size, "frames_per_second": fps, "seconds_per_frame": 1/fps}
        rows.append(row)
        print()
        print(f"Batch size: {batch_size}, Frames per second: {fps}, Seconds per frame: {1/fps}")
        print(row)

    df = pd.DataFrame(rows, columns=["batch_size", "frames_per_second", "seconds_per_frame"])
    df.to_csv(results_filepath, index=False)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_filepath", type=str, required=True)
    parser.add_argument("--batch_sizes", type=int, nargs="+", required=True)
    args = parser.parse_args()

    measure_runtime_fps(args.results_filepath, args.batch_sizes)