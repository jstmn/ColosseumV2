import pandas as pd
import numpy as np
import argparse
import matplotlib
import matplotlib.lines
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from numpy import isnan



""" Compares sim performance vs hardware performance

# Example usage:
python scripts/colosseum_v2_paper/figure_hardware.py \
    --sim-csv-filepath logs/parsed_ACT/single_arm.formatted.csv \
    --out-dir logs/
"""


HARDWARE_ROWS = [
    {
        "Task": "RaiseCube",
        "none": 55,
        "mo_size": 42.5,
        "light_color": 10,
        "distractor_object": 35,
        "background_color": 50,
        "mo_color": 0,
    },
    {
        "Task": "RotateArrow",
        "none": 65,
        "light_color": 25,
        "mo_size": 57.5,
        "distractor_object": 50,
        "background_color": 50,
        "mo_color": 50,
    },
    {
        "Task": "LiftPegUpright",
        "none": 60,
        "light_color": 25,
        "distractor_object": 50,
        "background_color": 30,
        "mo_size": 15,
        "mo_color": 45,
    },
    # {
    #     "Task": "OpenDrawer",
    #     "none": 80,
    #     "light_color": 5,
    #     "distractor_object": 70,
    #     "background_color": 65,
    #     "mo_color": 25,
    # },
]

DISTRACTION_SET_DISPLAY_NAMES = {
    "none".lower(): "None",
    "all".lower(): "All",
    "MO_color".lower(): "MO Color",
    "RO_color".lower(): "RO Color",
    "MO_texture".lower(): "MO Texture",
    "RO_texture".lower(): "RO Texture",
    "MO_size".lower(): "MO Size",
    "RO_size".lower(): "RO Size",
    "table_color".lower(): "Table Color",
    "light_color".lower(): "Light Color",
    "table_texture".lower(): "Table Texture",
    "distractor_object".lower(): "Distractor Object",
    "background_texture".lower(): "Background Texture",
    "background_color".lower(): "Background Color",
    "camera_pose".lower(): "Camera Pose",
    "MO_mass".lower(): "MO Mass",
    "language".lower(): "Language",
}


def plot_by_task(sim_csv_filepath: str, out_dir: str):

    sim_df = pd.read_csv(sim_csv_filepath)
    if "Task" not in sim_df.columns:
        raise ValueError(f"Expected a 'Task' column in {sim_csv_filepath}, got columns: {list(sim_df.columns)}")

    # Index both tables by Task so we can use .at[] lookups safely.
    # If sim has duplicate tasks, average them.
    sim_df = sim_df.set_index("Task")
    hardware_df = pd.DataFrame(HARDWARE_ROWS).set_index("Task")

    task_names = (
        "RaiseCube",
        "RotateArrow",
        "LiftPegUpright",
        # "OpenDrawer",
    )
    variation_names = (
        "none",
        "mo_size",
        "light_color",
        "distractor_object",
        "background_color",
        "mo_color",
    )
    print("Sim:")
    print(sim_df)
    print("\nHardware:")
    print(hardware_df)

    LABEL_FONTSIZE = 15
    LEGEND_FONTSIZE = 12
    TICK_FONTSIZE = 12

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig.subplots_adjust(right=0.55)

    task_colors = {
        "RaiseCube": "#FF8C00",      # vivid orange
        "RotateArrow": "#0066FF",    # strong blue
        "LiftPegUpright": "#00A651", # bright green-teal
        "OpenDrawer": "#CC00CC",     # vivid magenta
    }
    variation_markers = {
        "none": "o",                # circle
        "mo_size": "X",             # x (filled)
        "light_color": "s",         # square
        "distractor_object": "^",   # triangle_up
        "mo_color": "D",            # diamond
        "background_color": "P",    # plus (filled)
    }
    all_x1s = []
    all_x2s = []
    deltas_by_variation = {var: [] for var in variation_names}
    deltas_by_task = {task: [] for task in task_names}

    print()
    print("-------------")
    for task_name in task_names:

        task_x1s = []
        task_x2s = []
        none_scatter = None

        for variation_name in variation_names:

            hw_val = hardware_df.at[task_name, variation_name]
            sim_val = sim_df.at[task_name, variation_name]
            delta = hw_val - sim_val
            if isnan(hw_val) or isnan(sim_val):
                continue

            deltas_by_variation[variation_name].append(delta)
            deltas_by_task[task_name].append(delta)
            task_x1s.append(hw_val)
            task_x2s.append(sim_val)
            all_x1s.append(hw_val)
            all_x2s.append(sim_val)

            color = task_colors[task_name]
            if variation_name == "none":
                none_scatter = ax.scatter(
                    hw_val,
                    sim_val,
                    label=f"{task_name}",
                    color=color,
                    marker=variation_markers[variation_name],
                    s=125,
                )
            else:
                ax.scatter(
                    hw_val,
                    sim_val,
                    label=None,
                    color=color,
                    marker=variation_markers[variation_name],
                    s=125,
                )
        print(f"{task_name}: {task_x1s} {task_x2s}")
        assert none_scatter is not None
        best_fit_line = np.polyfit(task_x1s, task_x2s, 1)
        x1_range = np.arange(min(task_x1s), max(task_x1s))
        task_x2s_arr = np.array(task_x2s)
        y_pred = np.polyval(best_fit_line, task_x1s)
        R_squared = 1 - (np.sum((task_x2s_arr - y_pred) ** 2) / np.sum((task_x2s_arr - np.mean(task_x2s_arr)) ** 2))
        print(f"R-squared: {R_squared}")
        ax.plot(x1_range, np.polyval(best_fit_line, x1_range), color=color, linestyle="--")
        none_scatter.set_label(f"{task_name} (R²={R_squared:.3f})")

    print()
    print("Deltas by variation:")
    for var in variation_names:
        print(f"    {var}: {deltas_by_variation[var]}")

    print()
    print("Deltas by task:")
    for task in task_names:
        print(f"    {task}: {deltas_by_task[task]}")


    ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper left")
    ax.grid(True,alpha=0.5)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    # ax.legend(fontsize=LEGEND_FONTSIZE, ncol=3, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xlabel("Hardware Success Rate", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Sim Success Rate", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    plt.tight_layout()
    fig.savefig(Path(out_dir) / "hardware_vs_sim.png", bbox_inches="tight")
    plt.close(fig)
    

def plot_by_variation(sim_csv_filepath: str, out_dir: str):
    """This function lumps the success rates for each task by variation
    """

    sim_df = pd.read_csv(sim_csv_filepath)
    if "Task" not in sim_df.columns:
        raise ValueError(f"Expected a 'Task' column in {sim_csv_filepath}, got columns: {list(sim_df.columns)}")

    sim_df = sim_df.set_index("Task")
    hardware_df = pd.DataFrame(HARDWARE_ROWS).set_index("Task")

    task_names = (
        "RaiseCube",
        "RotateArrow",
        "LiftPegUpright",
        # "OpenDrawer",
    )
    variation_names = (
        "none",
        "mo_size",
        "light_color",
        "distractor_object",
        "background_color",
        "mo_color",
    )

    LABEL_FONTSIZE = 15
    LEGEND_FONTSIZE = 12
    TICK_FONTSIZE = 12

    variation_colors = {
        "none":               "#333333",
        "mo_size":            "#E6194B",
        "light_color":        "#F58231",
        "distractor_object":  "#3CB44B",
        "background_color":   "#4363D8",
        "mo_color":           "#911EB4",
    }
    task_markers = {
        "RaiseCube":     "o",
        "RotateArrow":   "X",
        "LiftPegUpright":"s",
        "OpenDrawer":    "^",
    }

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_xlim(0, 125)
    ax.set_ylim(0, 103)

    all_x1s = []
    all_x2s = []

    for variation_name in variation_names:
        var_hw = []
        var_sim = []
        first_scatter = None

        display_name = DISTRACTION_SET_DISPLAY_NAMES.get(variation_name.lower(), variation_name)
        color = variation_colors[variation_name]

        for task_name in task_names:
            hw_val = hardware_df.at[task_name, variation_name]
            sim_val = sim_df.at[task_name, variation_name]

            if isnan(hw_val) or isnan(sim_val):
                continue

            var_hw.append(hw_val)
            var_sim.append(sim_val)
            all_x1s.append(hw_val)
            all_x2s.append(sim_val)

            sc = ax.scatter(
                hw_val,
                sim_val,
                label=display_name if first_scatter is None else None,
                color=color,
                marker=task_markers[task_name],
                s=125,
                zorder=3,
            )
            if first_scatter is None:
                first_scatter = sc

        if len(var_hw) >= 2:
            best_fit_line = np.polyfit(var_hw, var_sim, 1)
            x_range = np.linspace(min(var_hw), max(var_hw), 100)
            y_pred = np.polyval(best_fit_line, var_hw)
            var_sim_arr = np.array(var_sim)
            ss_res = np.sum((var_sim_arr - y_pred) ** 2)
            ss_tot = np.sum((var_sim_arr - np.mean(var_sim_arr)) ** 2)
            R_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            ax.plot(x_range, np.polyval(best_fit_line, x_range), color=color, linestyle="--")
            if first_scatter is not None:
                first_scatter.set_label(f"{display_name} (R²={R_squared:.2f})")

    all_x1s_arr = np.array(all_x1s)
    all_x2s_arr = np.array(all_x2s)
    best_fit_line = np.polyfit(all_x1s_arr, all_x2s_arr, 1)
    x1_range = np.arange(min(all_x1s), max(all_x1s))
    y_pred = np.polyval(best_fit_line, all_x1s_arr)
    R_squared = 1 - (np.sum((all_x2s_arr - y_pred) ** 2) / np.sum((all_x2s_arr - np.mean(all_x2s_arr)) ** 2))
    ax.plot(x1_range, np.polyval(best_fit_line, x1_range), color="grey", linestyle="--", label=f"All (R²={R_squared:.2f})")
    # ax.text(0.05, 0.95, f"R²={R_squared:.2f}", fontsize=LABEL_FONTSIZE, transform=ax.transAxes, verticalalignment="top", horizontalalignment="left")

    # Marker legend for tasks
    task_legend_handles = [
        matplotlib.lines.Line2D(
            [], [],
            marker=task_markers[t],
            color="gray",
            linestyle="None",
            markersize=9,
            label=t,
        )
        for t in task_names
    ]
    variation_legend = ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper right", title="Variation")
    ax.add_artist(variation_legend)
    ax.legend(handles=task_legend_handles, fontsize=LEGEND_FONTSIZE, loc="lower right", title="Task")

    ax.grid(True, alpha=0.5)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_xlabel("Hardware Success Rate", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Sim Success Rate", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    plt.tight_layout()
    fig.savefig(Path(out_dir) / "hardware_vs_sim_by_variation.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-csv-filepath", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="plots/colosseum_v2_hardware")
    parser.add_argument("--no-plots", action="store_true", help="Skip writing plot images (prints summaries only).")
    args = parser.parse_args()
    plot_by_task(args.sim_csv_filepath, args.out_dir)
    plot_by_variation(args.sim_csv_filepath, args.out_dir)
