import pandas as pd
import numpy as np
import argparse
import matplotlib
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
        "background_color": None,
        "mo_size": 15,
        "mo_color": 45,
    },
    {
        "Task": "OpenDrawer",
        "none": 80,
        "light_color": 5,
        "distractor_object": 70,
        "background_color": 65,
        "mo_color": 25,
    },
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




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-csv-filepath", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="plots/colosseum_v2_hardware")
    parser.add_argument("--no-plots", action="store_true", help="Skip writing plot images (prints summaries only).")
    args = parser.parse_args()

    sim_df = pd.read_csv(args.sim_csv_filepath)
    if "Task" not in sim_df.columns:
        raise ValueError(f"Expected a 'Task' column in {args.sim_csv_filepath}, got columns: {list(sim_df.columns)}")

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


    # TITLE_FONTSIZE = 20
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
        "none": "o",
        "mo_size": "X",
        "light_color": "s",
        "distractor_object": "^",
        "mo_color": "D",
        "background_color": "P",
    }


    all_x1s = []
    all_x2s = []


    deltas_by_variation = {
        var: [] for var in variation_names
    }
    deltas_by_task = {
        task: [] for task in task_names
    }

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
            # print(f"{task_name}\t{variation_name}\t | hw, sim: ({hw_val}, {sim_val}) \t | \t delta:\t{delta}")

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
        none_scatter.set_label(f"{task_name} (R^2={R_squared:.3f})")

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
    # ax.set_title("Colosseum-V2 Hardware vs Sim", fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    plt.tight_layout()
    fig.savefig(Path(args.out_dir) / "hardware_vs_sim.png", bbox_inches="tight")
    
    # 
    # ax.legend(fontsize=LEGEND_FONTSIZE, ncol=1, bbox_to_anchor=(1.02, 1), loc="upper left")
    # plt.tight_layout()
    # fig.savefig(Path(args.out_dir) / "hardware_vs_sim__with_legend.png", bbox_inches="tight")    
    plt.close(fig)
    