import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
from pathlib import Path
from typing import Any, Final, cast
import sys
sys.path.append("examples/baselines/act_clip")
from eval_rgbd import MAX_EPISODE_STEPS_BY_TASK

"""This script generates the figures for the Colosseum-V2 paper.

The provided csv should have the following format:

batch_size,t_elapsed_sec,frames_per_second,seconds_per_frame,timesteps_per_task,rlbench_estimated_total_sim_time_seconds,rlbench_estimated_total_sim_time_minutes,rlbench_estimated_total_sim_time_hours
1,46.517,21.498,0.047,200,15815.78,263.596,4.393

# Example usage:
python scripts/colosseum_v2_paper/figure_runtime.py \
    --timing-csvs logs/fps/rlbench_runtime.csv logs/fps/act_runtime.csv \
    --model-names "Colosseum" "Colosseum-V2" \
    --output-filepath logs/fps/runtime_figure.png \
    --colosseum-v2-filepath logs/fps/act_runtime.csv --rlbench-filepath logs/fps/rlbench_runtime.csv
"""

LEFT_PLOT_BS: Final[list[int]] = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    20,
    30,
    40,
    50,
]

RIGHT_PLOT_BS: Final[list[int]] = [
    b
    for b in sorted(
        set(
            LEFT_PLOT_BS
            + [
                100,
                150,
                200,
                250,
                300,
                350,
                400,
                450,
                500
            ]
        )
    )
    if b >= 50
]


def generate_runtime_figure(timing_csvs: list[str], model_names: list[str], output_filepath: str):
    """
    This figure has two subplots. The left subplot is in the range [1, 50]. The right subplot is in the range [1, 2000].
    The x axes aren't uniform in terms of their actual values; they should be spaced evenly instead (categorical axis).
    """
    assert len(timing_csvs) == len(model_names), f"Expected {len(model_names)} timing CSVs, got {len(timing_csvs)}"

    DF_COLS = ["batch_size", "t_elapsed_sec", "frames_per_second", "seconds_per_frame", "timesteps_per_task", "rlbench_estimated_total_sim_time_seconds", "rlbench_estimated_total_sim_time_minutes", "rlbench_estimated_total_sim_time_hours"]

    def load_df(csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        missing = [c for c in DF_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"CSV {csv_path} missing columns {missing}. Got columns: {list(df.columns)}"
            )

        df = df[DF_COLS].copy()
        # Pandas typing stubs can be loose here; cast for type-checkers.
        df["batch_size"] = cast(pd.Series, pd.to_numeric(df["batch_size"], errors="coerce"))
        df["frames_per_second"] = cast(
            pd.Series, pd.to_numeric(df["frames_per_second"], errors="coerce")
        )
        df = cast(Any, df).dropna(subset=["batch_size", "frames_per_second"])
        df["batch_size"] = cast(pd.Series, df["batch_size"]).astype(int)

        # If the same batch size was measured multiple times, average it.
        df = (
            df.groupby("batch_size", as_index=False)
            .mean(numeric_only=True)
            .pipe(lambda d: cast(Any, d).sort_values(by=["batch_size"]))
            .reset_index(drop=True)
        )
        return df

    timing_dfs = [load_df(p) for p in timing_csvs]

    fontsize = 14

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(10, 4),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.4]},
    )

    n_models = len(model_names)
    bar_width = 0.8 / n_models if n_models > 0 else 0.8

    def plot_grouped_bars(ax: Axes, bs_list: list[int]):
        xs = np.arange(len(bs_list), dtype=float)
        for i, (model_name, df) in enumerate(zip(model_names, timing_dfs)):
            bs_to_fps = dict(zip(df["batch_size"].tolist(), df["frames_per_second"].tolist()))
            ys = np.array([bs_to_fps.get(bs, np.nan) for bs in bs_list], dtype=float)
            x_offsets = xs + (i - n_models / 2) * bar_width + bar_width / 2
            ax.bar(x_offsets, ys, width=bar_width, label=model_name, zorder=3)

    plot_grouped_bars(ax_left, LEFT_PLOT_BS)
    plot_grouped_bars(ax_right, RIGHT_PLOT_BS)

    tick_fontsize = fontsize-2

    ax_left.set_xlabel("Number of Environments",  fontsize=fontsize)
    ax_right.set_xlabel("Number of Environments",  fontsize=fontsize)
    ax_left.set_ylabel("ACT + Sim. Steps per Second [FPS]", fontsize=fontsize)
    for ax, bs_list in ((ax_left, LEFT_PLOT_BS), (ax_right, RIGHT_PLOT_BS)):
        ax.set_axisbelow(True)  # ensure grids are behind plotted artists
        ax.set_xticks(np.arange(len(bs_list), dtype=float))
        ax.set_xticklabels([str(b) for b in bs_list], fontsize=tick_fontsize)
        ax.grid(True, axis="y", alpha=0.25, zorder=0)
        # Enabling minor grid lines in matplotlib requires the minor ticks to be active.
        ax.minorticks_on()
        ax.grid(True, axis="y", which="minor", alpha=0.25, linestyle="--", zorder=0)
        if ax is ax_right:
            ax.set_ylim(bottom=0)
        ax.tick_params(axis="y", labelsize=tick_fontsize)

    # Left subplot: log scale on FPS (y-axis). Need strictly positive limits.
    left_vals: list[float] = []
    for df in timing_dfs:
        bs_to_fps = dict(zip(df["batch_size"].tolist(), df["frames_per_second"].tolist()))
        for bs in LEFT_PLOT_BS:
            v = bs_to_fps.get(bs, np.nan)
            if isinstance(v, (int, float)) and np.isfinite(v) and v > 0:
                left_vals.append(float(v))
    ax_left.set_yscale("log")
    if left_vals:
        vmin = min(left_vals)
        vmax = max(left_vals)
        ax_left.set_ylim(bottom=max(vmin * 0.8, 1e-3), top=vmax * 1.25)

    # Keep tick labels close to the axis; add spacing via xlabel labelpad instead.
    ax_left.tick_params(axis="x", labelsize=tick_fontsize)
    ax_right.tick_params(axis="x", labelsize=tick_fontsize)

    ax_left.legend(frameon=True, fontsize=fontsize-1)

    out_path = Path(output_filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250)
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing-csvs", type=str, required=True, nargs="+")
    parser.add_argument("--model-names", type=str, required=True, nargs="+")
    parser.add_argument("--output-filepath", type=str, required=True)
    parser.add_argument("--colosseum-v2-filepath", type=str, required=False)
    parser.add_argument("--rlbench-filepath", type=str, required=False)
    args = parser.parse_args()
    generate_runtime_figure(args.timing_csvs, args.model_names, args.output_filepath)

    if args.colosseum_v2_filepath:
        df = pd.read_csv(args.colosseum_v2_filepath)
        _200_bs_fps = df[df["batch_size"] == 200]["frames_per_second"].values[0]
        n_secs_total = 0
        n_episode_evals = 200 # 200 is the number of episode evaluations for Colosseum-V2
        n_perturbations = 17 # 17 is the number of perturbations for Colosseum-V2 (including none, all)
        print(f"With a batch_size of 200, FPS={_200_bs_fps}")

        # examples/baselines/act_clip/eval_rgbd.py has MAX_EPISODE_STEPS_BY_TASK
        for task, max_n_steps in MAX_EPISODE_STEPS_BY_TASK.items():
            n_secs_total += max_n_steps * n_episode_evals * n_perturbations / _200_bs_fps 

        print(f"Total estimated runtime for Colosseum-V2:")
        print(f"  - {round(n_secs_total, 3)} seconds")
        print(f"  - {round(n_secs_total / 60, 3)} minutes")
        print(f"  - {round(n_secs_total / 3600, 3)} hours")

    if args.rlbench_filepath:
        df = pd.read_csv(args.rlbench_filepath)
        _20_fps = df[df["batch_size"] == 20]["frames_per_second"].values[0]
        n_secs_total = 0
        n_episode_evals = 20 # 20 is the number of episode evaluations for RLBench/Colosseum-V1
        n_perturbations = 15 # 17 is the number of perturbations for Colosseum-V2 (including none, all)
        n_tasks = 20
        n_timesteps_per_task = 200
        print(f"With a batch_size of 20, FPS={_20_fps}")
        n_secs_total = n_episode_evals * n_perturbations * n_tasks * n_timesteps_per_task / _20_fps

        print(f"Total estimated runtime for RLBench/Colosseum-V1:")
        print(f"  - {round(n_secs_total, 3)} seconds")
        print(f"  - {round(n_secs_total / 60, 3)} minutes")
        print(f"  - {round(n_secs_total / 3600, 3)} hours")