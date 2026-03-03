import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
from pathlib import Path
from typing import Any, Final, cast

"""This script generates the figures for the Colosseum-V2 paper.

The provided csv should have the following format:

batch_size,frames_per_second,seconds_per_frame
1,71.64126893897986,0.01395843505859375
2,132.72559500679085,0.007534341812133789


# Example usage:
python scripts/generate_colosseum_v2_runtime_figure.py \
    --timing-csvs logs/fps/rlbench_fps.csv logs/fps/maniskill_fps.csv \
    --model-names "Colosseum" "Colosseum-V2" \
    --output-filepath logs/fps/runtime_figure.png
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
                250,
                500,
                750,
                1000,
                1250,
                1500,
                1750,
                2000,
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
    if len(timing_csvs) != len(model_names):
        raise ValueError(
            f"`timing_csvs` and `model_names` must have same length, got "
            f"{len(timing_csvs)} and {len(model_names)}"
        )

    DF_COLS = ["batch_size", "frames_per_second", "seconds_per_frame"]

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

    fontsize = 15

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

    tick_fontsize = fontsize-3

    ax_left.set_xlabel("Number of Environments",  fontsize=fontsize)
    ax_right.set_xlabel("Number of Environments",  fontsize=fontsize)
    ax_left.set_ylabel("Frames per second [FPS]", fontsize=fontsize)
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

    ax_left.legend(frameon=True, fontsize=fontsize)

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
    args = parser.parse_args()
    generate_runtime_figure(args.timing_csvs, args.model_names, args.output_filepath)
