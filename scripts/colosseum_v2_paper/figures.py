import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, cast
from matplotlib.projections.polar import PolarAxes

"""This script generates the figures for the Colosseum-V2 paper.

The provided csv should have the following format:

Task,none,all,distractor_object,mo_color,mo_texture,mo_size,mo_mass,ro_color,ro_texture,ro_size,table_color,table_texture,camera_pose,light_color,background_texture,background_color,language
DualArmLiftPot,97.5,0.0,78.0,61.5,27.0,42.0,27.5,,,,34.5,,49.0,32.0,86.5,97.0,94.0
DualArmPushBox,82.0,1.5,66.0,42.0,21.5,75.5,52.5,,,,54.0,,72.5,73.0,80.5,81.5,76.5
DualArmLiftTray,70.5,0.0,55.5,23.5,17.0,8.0,1.0,,,,,,41.5,56.5,61.5,52.0,66.5
DualArmPourPot,68.0,2.5,60.0,71.0,69.0,36.5,8.5,66.5,56.0,70.5,,,72.5,65.0,60.0,55.0,63.5
DualArmStackCube,59.5,0.0,9.5,0.0,1.0,39.5,38.5,23.5,24.0,56.0,4.0,,5.0,19.5,50.5,56.0,56.0
DualArmPenCap,18.0,2.0,6.0,7.0,11.0,,2.5,5.0,8.5,,,,8.0,8.5,15.0,11.5,15.5
DualArmThreading,9.0,0.0,0.0,3.5,3.5,,4.5,2.5,6.0,,,,0.5,0.0,0.5,1.0,8.5
DualArmDrawerOpen,6.0,5.5,9.0,,,,12.0,,,,,,5.5,7.0,8.0,7.5,10.0
DualArmStackTwoCubes,3.5,0.0,0.0,0.0,0.0,2.5,3.0,7.0,6.5,5.0,0.0,0.0,0.0,0.0,1.0,4.0,5.0
DualArmBottleHandover,1.0,4.5,1.5,2.5,0.5,0.5,1.5,,,,1.5,,1.0,2.0,1.0,2.0,0.0
DualArmCubeHandover,0.0,0.0,0.5,0.0,0.0,0.0,0.0,,,,0.0,,0.5,0.0,1.0,0.0,0.0
DualArmDrawerPlace,0.0,0.0,0.0,0.0,0.0,0.0,0.0,,,,,,0.0,0.0,0.0,0.0,0.0


# Example usage:
python scripts/colosseum_v2_paper/figures.py \
    --act-single-arm-csv logs/parsed_ACT/single_arm.formatted.csv \
    --act-bimanual-csv logs/parsed_ACT/bimanual.formatted.csv \
    --pi0-single-arm-csv logs/parsed_pi0/single_arm_5epochs.formatted.csv \
    --pi0-bimanual-csv logs/parsed_pi0/bimanual_5epochs.formatted.csv \
    --output-dir logs/
"""

COLOR_MAP = {
    "ACT - Single-Arm": "#ffd700", # yellow
    "ACT - Bimanual": "#fe6e5a", # orange
    "Pi0.5 - Single-Arm": "#0070ff", # blue
    "Pi0.5 - Bimanual": "#cd34b5", # purple
    # "ACT - Bimanual": "#ffd700",
    # "Pi0.5 - Bimanual": "#fe6e5a",
    # "ACT - Single-Arm": "#cd34b5",
    # "Pi0.5 - Single-Arm": "#0070ff", # this one is pretty good - more muted than 0000ff so doesn't stand out as much
}

LINE_STYLE_MAP = {
    "ACT - Bimanual":  "--",
    "Pi0.5 - Bimanual":"--",
    "ACT - Single-Arm": None,
    "Pi0.5 - Single-Arm": None,
}

LINE_STYLE_MAP = {
    "ACT - Bimanual":  "--",
    "Pi0.5 - Bimanual":"--",
    "ACT - Single-Arm": None,
    "Pi0.5 - Single-Arm": None,
}

PERTURBATION_SETS=(
    "none",
    "all",
    "MO_color",
    "RO_color",
    "MO_texture",
    "RO_texture",
    "MO_size",
    "RO_size",
    "table_color",
    "light_color",
    "table_texture",
    "distractor_object",
    "background_texture",
    "background_color",
    "camera_pose",
    "pose_randomization",
    "language"
)

PERTURBATION_SET_DISPLAY_NAMES = {
    "none": "None",
    "all": "All",
    "MO_color": "MO Color",
    "RO_color": "RO Color",
    "MO_texture": "MO Texture",
    "RO_texture": "RO Texture",
    "MO_size": "MO Size",
    "RO_size": "RO Size",
    "table_color": "Table Color",
    "light_color": "Light Color",
    "table_texture": "Table Texture",
    "distractor_object": "Distractor Object",
    "background_texture": "Background Texture",
    "background_color": "Background Color",
    "camera_pose": "Camera Pose",
    "pose_randomization": "Pose Randomization",
    "language": "Language",
}

VISION_PERTURBATION_SETS = [
    "MO_color",
    "RO_color",
    "MO_texture",
    "RO_texture",
    "table_color",
    "light_color",
    "table_texture",
    "distractor_object",
    "background_texture",
    "background_color",
    "camera_pose",
]
LANGUAGE_PERTURBATION_SETS = [
    "language"
]
ACTION_PERTURBATION_SETS = [
    "MO_size",
    "RO_size",
    "pose_randomization",
]
assert len(VISION_PERTURBATION_SETS) + len(ACTION_PERTURBATION_SETS) + len(LANGUAGE_PERTURBATION_SETS) == len(PERTURBATION_SETS) - 2
# ^ 2 not included are 'none' and 'all'

# Don't count tasks with none success rate below this threshold
LOWEST_NONE_SUCCESS_RATE_FOR_COUNTING = 10


def calculate_mean_changes_from_none(result_csvs: list[str], model_names: list[str]):
    mean_changes_from_none = {
        name: {} for name in model_names
    }
    mean_absolute_sr = {
        name: {} for name in model_names
    }
    for model_name, result_csv in zip(model_names, result_csvs):
        df = pd.read_csv(result_csv)
        df["Task"] = df["Task"].astype(str).str.strip()
        df = df.set_index("Task", drop=True)
        
        # Allow distraction-set columns to be either exact-case (e.g. MO_color) or lower-case (e.g. mo_color).
        col_by_lower = {str(c).strip().lower(): c for c in df.columns}
        
        for ds in PERTURBATION_SETS:
            ds_col = col_by_lower.get(str(ds).lower(), ds)
            mean_absolute_sr[model_name][ds] = df[ds_col].mean()


    for model_name, result_csv in zip(model_names, result_csvs):
        df = pd.read_csv(result_csv)

        # Index by task name for easy lookup: df.loc[task, perturbation_set]
        df["Task"] = df["Task"].astype(str).str.strip()
        df = df.set_index("Task", drop=True)
        df = df[df["none"] >= LOWEST_NONE_SUCCESS_RATE_FOR_COUNTING]

        # Allow perturbation-set columns to be either exact-case (e.g. MO_color) or lower-case (e.g. mo_color).
        col_by_lower = {str(c).strip().lower(): c for c in df.columns}

        success_rates = {}
        for task in df.index:
            success_rates[task] = {}
            for perturbation_set in PERTURBATION_SETS:
                ds_col = perturbation_set
                if ds_col not in df.columns:
                    ds_col = col_by_lower.get(str(perturbation_set).lower(), perturbation_set)
                if ds_col not in df.columns:
                    raise KeyError(
                        f"CSV {result_csv} missing perturbation-set column '{perturbation_set}' "
                        f"(also tried '{str(perturbation_set).lower()}'). Got columns: {list(df.columns)}"
                    )
                v = df.loc[task, ds_col]
                # Cells may be empty -> NaN; treat as 0.0 success.
                # TODO: ignore tasks with none success rate
                if pd.isna(v):
                    continue
                success_rates[task][perturbation_set] = float(v)

        for ds in PERTURBATION_SETS:
            all_changes = []
            for task in success_rates.keys():
                base = success_rates[task]["none"]
                if base <= 0:
                    print(f"Task {task} has none success rate <= 0")
                    continue
                # Mean change: (success[perturbation_set] - success[none])
                if ds not in success_rates[task]:
                    print(f"Task {task} has no success rate for distraction set {ds}")
                    continue
                change = success_rates[task][ds] - base
                all_changes.append(change)
            mean_changes_from_none[model_name][ds] = float(np.mean(all_changes)) if all_changes else 0.0
    return mean_changes_from_none, mean_absolute_sr


def generate_clumped_change_figure(mean_changes_from_none: Dict[str, Dict[str, float]], model_names: list[str], output_dir: str):
    n_models = len(model_names)

    # Compute per-model means clumped by modality.
    mean_clumped_changes: Dict[str, Dict[str, float]] = {
        model: {"vision": 0.0, "action": 0.0, "language": 0.0} for model in model_names
    }
    for model in model_names:
        mean_changes = mean_changes_from_none.get(model, {})
        
        vision_vals = []
        action_vals = []
        language_vals = []
        for ds in VISION_PERTURBATION_SETS:
            vision_vals.append(float(mean_changes[ds]))
        for ds in ACTION_PERTURBATION_SETS:
            action_vals.append(float(mean_changes[ds]))
        for ds in LANGUAGE_PERTURBATION_SETS:
            language_vals.append(float(mean_changes[ds]))

        mean_clumped_changes[model]["vision"] = float(np.mean(vision_vals)) if vision_vals else 0.0
        mean_clumped_changes[model]["action"] = float(np.mean(action_vals)) if action_vals else 0.0
        mean_clumped_changes[model]["language"] = float(np.mean(language_vals)) if language_vals else 0.0

    # Plot grouped barchart (matches style of generate_mean_change_figure).
    categories = ["vision", "language", "action"]
    display = {"vision": "Vision", "action": "Action", "language": "Language"}
    values = [[mean_clumped_changes[model][c] for c in categories] for model in model_names]

    x = range(len(categories))
    bar_width = 0.8 / n_models if n_models > 0 else 0.8

    plt.figure(figsize=(10, 3))
    for i, model in enumerate(model_names):
        plt.bar(
            [xx + (i - n_models / 2) * bar_width + bar_width / 2 for xx in x],
            values[i],
            width=bar_width,
            label=model,
            color=COLOR_MAP[model]
        )

    fontsize = 13
    plt.xticks(
        list(x),
        [display[c] for c in categories],
        rotation=0,
        ha="center",
        fontsize=fontsize,
    )
    plt.ylabel("Mean Change\nin Success Rate", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "mean_change_clumped_barchart.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved clumped mean change barchart to {save_path}")


def generate_clumped_change_figure_radial(mean_changes_from_none: Dict[str, Dict[str, float]], model_names: list[str], output_dir: str):

    # Compute per-model means clumped by modality.
    mean_clumped_changes: Dict[str, Dict[str, float]] = {
        model: {"vision": 0.0, "action": 0.0, "language": 0.0} for model in model_names
    }
    for model in model_names:
        mean_changes = mean_changes_from_none.get(model)
        assert isinstance(mean_changes, dict)
        vision_vals = []
        action_vals = []
        language_vals = []
        for ds in VISION_PERTURBATION_SETS:
            vision_vals.append(float(mean_changes[ds]))
        for ds in ACTION_PERTURBATION_SETS:
            action_vals.append(float(mean_changes[ds]))
        for ds in LANGUAGE_PERTURBATION_SETS:
            language_vals.append(float(mean_changes[ds]))

        mean_clumped_changes[model]["vision"] = -float(np.mean(vision_vals))
        mean_clumped_changes[model]["action"] = -float(np.mean(action_vals))
        mean_clumped_changes[model]["language"] = -float(np.mean(language_vals))

    # Plot radar / radial chart.
    categories = ["vision", "language", "action"]
    display = {"vision": "Vision", "action": "Action", "language": "Language"}
    values = {
        model: [mean_clumped_changes[model][cat] for cat in categories] for model in model_names
    }

    # Angles for each axis (close the loop by repeating the first).
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles.append(angles[0])

    fig, ax0 = plt.subplots(figsize=(6.2, 5.2), subplot_kw={"polar": True})
    ax = cast(PolarAxes, ax0)
    ax.set_theta_offset(np.pi / 2)     # start at the top
    ax.set_theta_direction(-1)         # clockwise

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([display[c] for c in categories], fontsize=14)
    label_pads = {"vision": 4, "action": 15, "language": 25}
    for tick, cat in zip(ax.xaxis.get_major_ticks(), categories):
        tick.set_pad(label_pads[cat])
    ax.tick_params(axis="y", labelsize=12)

    ax.set_rlabel_position(180)  # move radial labels to the left
    ax.grid(True, alpha=0.35)

    # Plot each model.
    for model_name in model_names:
        vals = values[model_name]
        vals.append(vals[0])
        ax.plot(angles, vals, linewidth=2, label=model_name, color=COLOR_MAP[model_name], linestyle=LINE_STYLE_MAP.get(model_name))
        ax.fill(angles, vals, color=COLOR_MAP[model_name], alpha=0.10)

    # ax.set_title("Mean Change in Success Rate (Clumped)", fontsize=13, pad=18)
    legend = ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=12, frameon=True)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#cccccc")
    legend.get_frame().set_linewidth(0.8)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "radial_relative_to_none.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved clumped mean change barchart to {save_path}")


def generate_radial_absolute(mean_absolute_sr: Dict[str, Dict[str, float]], model_names: list[str], output_dir: str):

    # Compute per-model means clumped by modality.
    mean_clumped_absolute_sr: Dict[str, Dict[str, float]] = {
        model: {"vision": 0.0, "action": 0.0, "language": 0.0} for model in model_names
    }
    for model in model_names:
        mean_srs = mean_absolute_sr.get(model)
        assert isinstance(mean_srs, dict)
        vision_vals = []
        action_vals = []
        language_vals = []
        for ds in VISION_PERTURBATION_SETS:
            vision_vals.append(float(mean_srs[ds]))
        for ds in ACTION_PERTURBATION_SETS:
            action_vals.append(float(mean_srs[ds]))
        for ds in LANGUAGE_PERTURBATION_SETS:
            language_vals.append(float(mean_srs[ds]))

        mean_clumped_absolute_sr[model]["vision"] = float(np.mean(vision_vals))
        mean_clumped_absolute_sr[model]["action"] = float(np.mean(action_vals))
        mean_clumped_absolute_sr[model]["language"] = float(np.mean(language_vals))

    # Plot radar / radial chart.
    categories = ["vision", "language", "action"]
    display = {"vision": "Vision", "action": "Action", "language": "Language"}
    values = {
        model: [mean_clumped_absolute_sr[model][cat] for cat in categories] for model in model_names
    }

    # Angles for each axis. Need to close the loop by repeating the first.
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles.append(angles[0])

    fig, ax0 = plt.subplots(figsize=(6.2, 5.2), subplot_kw={"polar": True})
    ax = cast(PolarAxes, ax0)
    ax.set_theta_offset(np.pi / 2)     # start at the top
    ax.set_theta_direction(-1)         # clockwise

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([display[c] for c in categories], fontsize=14)
    label_pads = {"vision": 4, "action": 15, "language": 25}
    for tick, cat in zip(ax.xaxis.get_major_ticks(), categories):
        tick.set_pad(label_pads[cat])
    ax.tick_params(axis="y", labelsize=12)

    ax.set_rlabel_position(180)  # move radial labels to the left
    ax.grid(True, alpha=0.35)

    # Plot each model.
    for model_name in model_names:
        vals = values[model_name]
        vals.append(vals[0])
        ax.plot(angles, vals, linewidth=2, label=model_name, color=COLOR_MAP[model_name], linestyle=LINE_STYLE_MAP.get(model_name))
        ax.fill(angles, vals, color=COLOR_MAP[model_name], alpha=0.10)

    # ax.set_title("Mean Change in Success Rate (Clumped)", fontsize=13, pad=18)
    legend = ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=12, frameon=True)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#cccccc")
    legend.get_frame().set_linewidth(0.8)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "radial_absolute_sr.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved clumped mean change barchart to {save_path}")


def generate_radial_two_plots(mean_absolute_sr: Dict[str, Dict[str, float]], mean_changes_from_none: Dict[str, Dict[str, float]], model_names: list[str], output_dir: str):
    # Compute per-model means clumped by modality.
    mean_clumped_changes: Dict[str, Dict[str, float]] = {
        model: {"vision": 0.0, "action": 0.0, "language": 0.0} for model in model_names
    }
    mean_clumped_absolute_sr: Dict[str, Dict[str, float]] = {
        model: {"vision": 0.0, "action": 0.0, "language": 0.0} for model in model_names
    }
    # SR
    for model in model_names:
        mean_srs = mean_absolute_sr.get(model)
        assert isinstance(mean_srs, dict)
        vision_vals = []
        action_vals = []
        language_vals = []
        for ds in VISION_PERTURBATION_SETS:
            vision_vals.append(float(mean_srs[ds]))
        for ds in ACTION_PERTURBATION_SETS:
            action_vals.append(float(mean_srs[ds]))
        for ds in LANGUAGE_PERTURBATION_SETS:
            language_vals.append(float(mean_srs[ds]))

        mean_clumped_absolute_sr[model]["vision"] = float(np.mean(vision_vals))
        mean_clumped_absolute_sr[model]["action"] = float(np.mean(action_vals))
        mean_clumped_absolute_sr[model]["language"] = float(np.mean(language_vals))

    # Change
    for model in model_names:
        mean_changes = mean_changes_from_none.get(model)
        assert isinstance(mean_changes, dict)
        vision_vals = []
        action_vals = []
        language_vals = []
        for ds in VISION_PERTURBATION_SETS:
            vision_vals.append(float(mean_changes[ds]))
        for ds in ACTION_PERTURBATION_SETS:
            action_vals.append(float(mean_changes[ds]))
        for ds in LANGUAGE_PERTURBATION_SETS:
            language_vals.append(float(mean_changes[ds]))

        mean_clumped_changes[model]["vision"] = -float(np.mean(vision_vals))
        mean_clumped_changes[model]["action"] = -float(np.mean(action_vals))
        mean_clumped_changes[model]["language"] = -float(np.mean(language_vals))

    # Plot radar / radial chart.
    categories = ["language", "vision", "action"]
    display = {"vision": "Vision", "action": "Action", "language": "Language"}
    values_delta = {
        model: {
            "delta": [mean_clumped_changes[model][cat] for cat in categories],
            "abs": [mean_clumped_absolute_sr[model][cat] for cat in categories]
        } for model in model_names
    } 
    for model in model_names:
        print(f"Model {model}:")
        print(f"  - values_delta[model]['delta']:")
        for val, cat in zip(values_delta[model]['delta'], categories):
            print(f"      - {cat}: {val:.4f}")
        print(f"  - values_delta[model]['abs']:")
        for val, cat in zip(values_delta[model]['abs'], categories):
            print(f"      - {cat}: {val:.4f}")

    # Angles for each axis (close the loop by repeating the first).
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles.append(angles[0])
    # 
    ticks_fontsize = 12

    fig, axs = plt.subplots(figsize=(10, 4.4), ncols=2, subplot_kw={"polar": True})
    for ax in axs:
        ax.set_theta_offset(np.pi / 2)     # start at the top
        ax.set_theta_direction(-1)         # clockwise
        ax.set_theta_offset(np.pi / 2)     # start at the top
        ax.set_theta_direction(-1)         # clockwise
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([display[c] for c in categories], fontsize=ticks_fontsize)
        label_pads = {"vision": 10, "action": 8, "language": 0}
        for tick, cat in zip(ax.xaxis.get_major_ticks(), categories):
            tick.set_pad(label_pads[cat])
        ax.tick_params(axis="y", labelsize=12)
        ax.set_rlabel_position(180)  # move radial labels to the left
        ax.grid(True, alpha=0.35)

    # Plot each model.
    for i, model in enumerate(model_names):
        vals_delta = values_delta[model]["delta"]
        vals_delta.append(vals_delta[0])
        vals_abs = values_delta[model]["abs"]
        vals_abs.append(vals_abs[0])
        axs[0].plot(angles, vals_delta, linewidth=2, label=model, color=COLOR_MAP[model], linestyle=LINE_STYLE_MAP.get(model))
        axs[0].fill(angles, vals_delta, color=COLOR_MAP[model], alpha=0.10)
        axs[1].plot(angles, vals_abs, linewidth=2, label=model, color=COLOR_MAP[model], linestyle=LINE_STYLE_MAP.get(model))
        axs[1].fill(angles, vals_abs, color=COLOR_MAP[model], alpha=0.10)

    legend = axs[1].legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=12, frameon=True)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(1.0)
    legend.get_frame().set_edgecolor("#cccccc")
    legend.get_frame().set_linewidth(1.0)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.0)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "radial_two_plots.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved radial two plots to {save_path}")



def generate_radial_two_plots_v2(mean_absolute_sr: Dict[str, Dict[str, float]], mean_changes_from_none: Dict[str, Dict[str, float]], model_names: list[str], output_dir: str):
    # Compute per-model means clumped by modality.
    mean_clumped_changes: Dict[str, Dict[str, float]] = {
        model: {"vision": 0.0, "action": 0.0, "language": 0.0} for model in model_names
    }
    mean_clumped_absolute_sr: Dict[str, Dict[str, float]] = {
        model: {"vision": 0.0, "action": 0.0, "language": 0.0} for model in model_names
    }
    # SR
    for model in model_names:
        mean_srs = mean_absolute_sr.get(model)
        assert isinstance(mean_srs, dict)
        vision_vals = []
        action_vals = []
        language_vals = []
        for ds in VISION_DISTRACTION_SETS:
            vision_vals.append(float(mean_srs[ds]))
        for ds in ACTION_DISTRACTION_SETS:
            action_vals.append(float(mean_srs[ds]))
        for ds in LANGUAGE_DISTRACTION_SETS:
            language_vals.append(float(mean_srs[ds]))

        mean_clumped_absolute_sr[model]["vision"] = float(np.mean(vision_vals))
        mean_clumped_absolute_sr[model]["action"] = float(np.mean(action_vals))
        mean_clumped_absolute_sr[model]["language"] = float(np.mean(language_vals))

    # Change
    for model in model_names:
        mean_changes = mean_changes_from_none.get(model)
        assert isinstance(mean_changes, dict)
        vision_vals = []
        action_vals = []
        language_vals = []
        for ds in VISION_DISTRACTION_SETS:
            vision_vals.append(float(mean_changes[ds]))
        for ds in ACTION_DISTRACTION_SETS:
            action_vals.append(float(mean_changes[ds]))
        for ds in LANGUAGE_DISTRACTION_SETS:
            language_vals.append(float(mean_changes[ds]))

        mean_clumped_changes[model]["vision"] = float(np.mean(vision_vals))
        mean_clumped_changes[model]["action"] = float(np.mean(action_vals))
        mean_clumped_changes[model]["language"] = float(np.mean(language_vals))

    # Plot: left = grouped bar chart (delta), right = radial chart (abs SR).
    categories = ["language", "vision", "action"]
    display = {"vision": "Vision", "action": "Action", "language": "Language"}
    values_delta = {
        model: {
            "delta": [mean_clumped_changes[model][cat] for cat in categories],
            "abs": [mean_clumped_absolute_sr[model][cat] for cat in categories]
        } for model in model_names
    }
    for model in model_names:
        print(f"Model {model}:")
        print(f"  - values_delta[model]['delta']:")
        for val, cat in zip(values_delta[model]['delta'], categories):
            print(f"      - {cat}: {val:.4f}")
        print(f"  - values_delta[model]['abs']:")
        for val, cat in zip(values_delta[model]['abs'], categories):
            print(f"      - {cat}: {val:.4f}")

    ticks_fontsize = 12

    fig = plt.figure(figsize=(11, 4.4))
    ax_radar = fig.add_subplot(1, 2, 1, polar=True)
    ax_bar = fig.add_subplot(1, 2, 2)

    # --- Left: grouped bar chart of robustness drop (delta) ---
    n_models = len(model_names)
    n_cats = len(categories)
    bar_width = 0.8 / n_models
    x = np.arange(n_cats)
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * bar_width

    for i, model in enumerate(model_names):
        vals = values_delta[model]["delta"]
        ax_bar.bar(
            x + offsets[i],
            vals,
            width=bar_width,
            label=model,
            color=COLOR_MAP[model],
            edgecolor="white",
            linewidth=0.5,
        )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([display[c] for c in categories], fontsize=ticks_fontsize)
    ax_bar.tick_params(axis="y", labelsize=ticks_fontsize)
    ax_bar.set_ylabel("Mean Decrease vs. No Variation [%]", fontsize=ticks_fontsize)
    ax_bar.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_bar.grid(axis="y", alpha=0.35)
    ax_bar.set_axisbelow(True)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # --- Right: radial chart of absolute SR ---
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles.append(angles[0])

    ax_radar.set_theta_offset(np.pi / 2)
    ax_radar.set_theta_direction(-1)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels([display[c] for c in categories], fontsize=ticks_fontsize)
    label_pads = {"vision": 10, "action": 8, "language": 0}
    for tick, cat in zip(ax_radar.xaxis.get_major_ticks(), categories):
        tick.set_pad(label_pads[cat])
    ax_radar.tick_params(axis="y", labelsize=ticks_fontsize)
    ax_radar.set_rlabel_position(180)
    ax_radar.grid(True, alpha=0.35)
    ax_radar.set_xlabel("Mean Absolute Success Rate [%]", fontsize=ticks_fontsize)

    for model in model_names:
        vals_abs = values_delta[model]["abs"] + [values_delta[model]["abs"][0]]
        ax_radar.plot(angles, vals_abs, linewidth=2, label=model, color=COLOR_MAP[model], linestyle=LINE_STYLE_MAP.get(model))
        ax_radar.fill(angles, vals_abs, color=COLOR_MAP[model], alpha=0.10)

    # legend = ax_radar.legend(loc="upper right", bbox_to_anchor=(1.42, 1.15), fontsize=12, frameon=True)
    legend = ax_radar.legend(loc="upper left", bbox_to_anchor=(-0.35, 1.1), fontsize=12, frameon=True)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(1.0)
    legend.get_frame().set_edgecolor("#cccccc")
    legend.get_frame().set_linewidth(1.0)
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.0)
    plt.subplots_adjust(wspace=0.15)
    # plt.subplots_adjust(wspace=0.6)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "radial_two_plots_v2.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved radial two plots to {save_path}")


# def generate_radial_two_plots_v2(mean_absolute_sr: Dict[str, Dict[str, float]], mean_changes_from_none: Dict[str, Dict[str, float]], model_names: list[str], output_dir: str):
#     # Compute per-model means clumped by modality.
#     mean_clumped_changes: Dict[str, Dict[str, float]] = {
#         model: {"vision": 0.0, "action": 0.0, "language": 0.0} for model in model_names
#     }
#     mean_clumped_absolute_sr: Dict[str, Dict[str, float]] = {
#         model: {"vision": 0.0, "action": 0.0, "language": 0.0} for model in model_names
#     }
#     # SR
#     for model in model_names:
#         mean_srs = mean_absolute_sr.get(model)
#         assert isinstance(mean_srs, dict)
#         vision_vals = []
#         action_vals = []
#         language_vals = []
#         for ds in VISION_DISTRACTION_SETS:
#             vision_vals.append(float(mean_srs[ds]))
#         for ds in ACTION_DISTRACTION_SETS:
#             action_vals.append(float(mean_srs[ds]))
#         for ds in LANGUAGE_DISTRACTION_SETS:
#             language_vals.append(float(mean_srs[ds]))

#         mean_clumped_absolute_sr[model]["vision"] = float(np.mean(vision_vals))
#         mean_clumped_absolute_sr[model]["action"] = float(np.mean(action_vals))
#         mean_clumped_absolute_sr[model]["language"] = float(np.mean(language_vals))

#     # Change
#     for model in model_names:
#         mean_changes = mean_changes_from_none.get(model)
#         assert isinstance(mean_changes, dict)
#         vision_vals = []
#         action_vals = []
#         language_vals = []
#         for ds in VISION_DISTRACTION_SETS:
#             vision_vals.append(float(mean_changes[ds]))
#         for ds in ACTION_DISTRACTION_SETS:
#             action_vals.append(float(mean_changes[ds]))
#         for ds in LANGUAGE_DISTRACTION_SETS:
#             language_vals.append(float(mean_changes[ds]))

#         mean_clumped_changes[model]["vision"] = -float(np.mean(vision_vals))
#         mean_clumped_changes[model]["action"] = -float(np.mean(action_vals))
#         mean_clumped_changes[model]["language"] = -float(np.mean(language_vals))

#     # Plot: left = grouped bar chart (delta), right = radial chart (abs SR).
#     categories = ["language", "vision", "action"]
#     display = {"vision": "Vision", "action": "Action", "language": "Language"}
#     values_delta = {
#         model: {
#             "delta": [mean_clumped_changes[model][cat] for cat in categories],
#             "abs": [mean_clumped_absolute_sr[model][cat] for cat in categories]
#         } for model in model_names
#     }
#     for model in model_names:
#         print(f"Model {model}:")
#         print(f"  - values_delta[model]['delta']:")
#         for val, cat in zip(values_delta[model]['delta'], categories):
#             print(f"      - {cat}: {val:.4f}")
#         print(f"  - values_delta[model]['abs']:")
#         for val, cat in zip(values_delta[model]['abs'], categories):
#             print(f"      - {cat}: {val:.4f}")

#     ticks_fontsize = 12

#     fig = plt.figure(figsize=(10, 4.4))
#     ax_bar = fig.add_subplot(1, 2, 1)
#     ax_radar = fig.add_subplot(1, 2, 2, polar=True)

#     # --- Left: grouped bar chart of robustness drop (delta) ---
#     n_models = len(model_names)
#     n_cats = len(categories)
#     bar_width = 0.8 / n_models
#     x = np.arange(n_cats)
#     offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * bar_width

#     for i, model in enumerate(model_names):
#         vals = values_delta[model]["delta"]
#         ax_bar.bar(
#             x + offsets[i],
#             vals,
#             width=bar_width,
#             label=model,
#             color=COLOR_MAP[model],
#             edgecolor="white",
#             linewidth=0.5,
#         )

#     ax_bar.set_xticks(x)
#     ax_bar.set_xticklabels([display[c] for c in categories], fontsize=ticks_fontsize)
#     ax_bar.tick_params(axis="y", labelsize=ticks_fontsize)
#     ax_bar.set_ylabel("Mean Decrease vs. No Variation [%]", fontsize=ticks_fontsize)
#     ax_bar.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
#     ax_bar.grid(axis="y", alpha=0.35)
#     ax_bar.set_axisbelow(True)
#     ax_bar.spines["top"].set_visible(False)
#     ax_bar.spines["right"].set_visible(False)

#     # --- Right: radial chart of absolute SR ---
#     angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
#     angles.append(angles[0])

#     ax_radar.set_theta_offset(np.pi / 2)
#     ax_radar.set_theta_direction(-1)
#     ax_radar.set_xticks(angles[:-1])
#     ax_radar.set_xticklabels([display[c] for c in categories], fontsize=ticks_fontsize)
#     label_pads = {"vision": 10, "action": 8, "language": 0}
#     for tick, cat in zip(ax_radar.xaxis.get_major_ticks(), categories):
#         tick.set_pad(label_pads[cat])
#     ax_radar.tick_params(axis="y", labelsize=ticks_fontsize)
#     ax_radar.set_rlabel_position(180)
#     ax_radar.grid(True, alpha=0.35)
#     ax_radar.set_xlabel("Mean Absolute Success Rate [%]", fontsize=ticks_fontsize)

#     for model in model_names:
#         vals_abs = values_delta[model]["abs"] + [values_delta[model]["abs"][0]]
#         ax_radar.plot(angles, vals_abs, linewidth=2, label=model, color=COLOR_MAP[model], linestyle=LINE_STYLE_MAP.get(model))
#         ax_radar.fill(angles, vals_abs, color=COLOR_MAP[model], alpha=0.10)

#     # legend = ax_radar.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=12, frameon=True)
#     legend = ax_radar.legend(loc="upper left", bbox_to_anchor=(-0.4, 1.15), fontsize=12, frameon=True)
#     legend.get_frame().set_facecolor("white")
#     legend.get_frame().set_alpha(1.0)
#     legend.get_frame().set_edgecolor("#cccccc")
#     legend.get_frame().set_linewidth(1.0)
#     plt.tight_layout()
#     plt.subplots_adjust(wspace=0.0)

#     os.makedirs(output_dir, exist_ok=True)
#     save_path = os.path.join(output_dir, "radial_two_plots_v2.png")
#     plt.savefig(save_path, bbox_inches="tight")
#     plt.close()
#     print(f"Saved radial two plots to {save_path}")


def generate_waterfall_plot(single_arm_csvs: list[str], single_arm_model_names: list[str], bimanual_csvs: list[str], bimanual_model_names: list[str], output_dir: str):
    """This function generates a waterfall plot of the success rate on the 'none' environment perturbation for each task. 
    For each model, the y-axis is the success rate on the 'none' environment perturbation. The x-axis is the task index. 
    The plot is a line plot of the x1,x1 point to the x1,xn point.

    Args:
        result_csvs (list[str]): A list of paths to the result csvs.
        model_names (list[str]): A list of model names.
        output_dir (str): The directory to save the plot to.
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    def _plot(ax, model_names, result_csvs):
        
        for model_name, result_csv in zip(model_names, result_csvs):
            df = pd.read_csv(result_csv)
            if "Task" not in df.columns:
                raise ValueError(f"CSV {result_csv} missing required 'Task' column. Got columns: {list(df.columns)}")

            df["Task"] = df["Task"].astype(str).str.strip()
            none_col = "none"
            if none_col not in df.columns:
                raise KeyError(f"CSV {result_csv} missing 'none' column. Got columns: {list(df.columns)}")

            task_success_rates = []
            for _, row in df.iterrows():
                task = row["Task"]
                success_rate = row[none_col]
                if pd.notna(success_rate):
                    task_success_rates.append((task, float(success_rate)))

            task_success_rates.sort(key=lambda x: x[1], reverse=True)
            task_indices = list(range(len(task_success_rates)))
            success_rates = [sr for _, sr in task_success_rates]

            ax.plot(
                task_indices,
                success_rates,
                marker='o',
                linewidth=2,
                markersize=6,
                label=model_name,
                color=COLOR_MAP[model_name]
            )
            ax.scatter(
                task_indices,
                success_rates,
                color=COLOR_MAP[model_name]
            )
    _plot(axs[0], single_arm_model_names, single_arm_csvs)
    _plot(axs[1], bimanual_model_names, bimanual_csvs)

    fontsize = 13
    axs[0].set_xlabel("Task Index", fontsize=fontsize)
    axs[0].set_ylabel("Success Rate [%]", fontsize=fontsize)
    axs[0].legend(fontsize=fontsize)
    axs[0].grid(True, alpha=0.4)
    axs[0].grid(True, which="minor", linestyle="--", alpha=0.2)
    axs[0].minorticks_on()
    axs[1].set_xlabel("Task Index", fontsize=fontsize)
    axs[1].legend(fontsize=fontsize)
    axs[1].grid(True, alpha=0.4)
    axs[1].grid(True, which="minor", linestyle="--", alpha=0.2)
    axs[1].minorticks_on()
    fig.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "waterfall_plot.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved waterfall plot to {save_path}")


def generate_mean_change_figure(mean_changes_from_none: Dict[str, Dict[str, float]], model_names: list[str], output_dir: str):

    # Plotting barchart
    perturbation_sets = list(PERTURBATION_SETS)
    n_ds = len(perturbation_sets)
    n_models = len(model_names)

    # Retrieve mean changes in matrix form for easier plotting
    values = []
    for model in model_names:
        values.append([mean_changes_from_none[model][ds] for ds in perturbation_sets])  # absolute change

    x = range(n_ds)
    bar_width = 0.8 / n_models

    plt.figure(figsize=(16, 4))
    for i, model in enumerate(model_names):
        plt.bar(
            [xx + (i - n_models/2) * bar_width + bar_width/2 for xx in x],
            values[i],
            width=bar_width,
            label=model,
            color=COLOR_MAP[model]
        )

    fontsize = 11
    plt.xticks(
        x,
        [PERTURBATION_SET_DISPLAY_NAMES[ds] for ds in perturbation_sets],
        rotation=45,
        ha='right',
        fontsize=fontsize
    )
    plt.ylabel("Mean Absolute Change\nin Success Rate", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "mean_change_barchart.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved mean change barchart to {save_path}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--act-single-arm-csv", type=str, required=True)
    parser.add_argument("--act-bimanual-csv", type=str, required=True)
    parser.add_argument("--pi0-single-arm-csv", type=str, required=True)
    parser.add_argument("--pi0-bimanual-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    result_csvs = [args.act_single_arm_csv, args.act_bimanual_csv, args.pi0_single_arm_csv, args.pi0_bimanual_csv]
    model_names = ["ACT - Single-Arm", "ACT - Bimanual", "Pi0.5 - Single-Arm", "Pi0.5 - Bimanual"]
    mean_changes_from_none, mean_absolute_sr = calculate_mean_changes_from_none(result_csvs, model_names)

    # First, print out some stats.
    print()
    act_bimanual_results = mean_changes_from_none["ACT - Bimanual"]
    act_single_arm_results = mean_changes_from_none["ACT - Single-Arm"]
    pi0_bimanual_results = mean_changes_from_none["Pi0.5 - Bimanual"]
    pi0_single_arm_results = mean_changes_from_none["Pi0.5 - Single-Arm"]
    act_deltas = []
    pi0_deltas = []
    for ds in DISTRACTION_SETS:
        if 'none'.lower() in ds.lower():
            continue
        act_bimanual_result = act_bimanual_results[ds]
        act_single_arm_result = act_single_arm_results[ds]
        pi0_bimanual_result = pi0_bimanual_results[ds]
        pi0_single_arm_result = pi0_single_arm_results[ds]
        act_delta = abs(act_single_arm_result - act_bimanual_result)
        pi0_delta = abs(pi0_single_arm_result - pi0_bimanual_result)
        print(f"{ds}:\tACT - Single/Bimanual: {act_delta:.4f}\tPi0.5 - Single/Bimanual: {pi0_delta:.4f}")
        act_deltas.append(act_delta)
        pi0_deltas.append(pi0_delta)
    print(f"ACT - Single-Arm, Bimanual: {np.mean(act_deltas):.4f}")
    print(f"Pi0.5 - Bimanual, Single-Arm: {np.mean(pi0_deltas):.4f}")
    print()


    # Generate plots
    generate_mean_change_figure(mean_changes_from_none, model_names, args.output_dir)
    generate_clumped_change_figure(mean_changes_from_none, model_names, args.output_dir)
    generate_clumped_change_figure_radial(mean_changes_from_none, model_names, args.output_dir)
    generate_radial_absolute(mean_absolute_sr, model_names, args.output_dir)
    generate_radial_two_plots(mean_absolute_sr, mean_changes_from_none, model_names, args.output_dir)
    generate_radial_two_plots_v2(mean_absolute_sr, mean_changes_from_none, model_names, args.output_dir)

    # Waterfall plots
    generate_waterfall_plot(
        single_arm_csvs=[args.act_single_arm_csv, args.pi0_single_arm_csv],
        single_arm_model_names=["ACT - Single-Arm", "Pi0.5 - Single-Arm"],
        bimanual_csvs=[args.act_bimanual_csv, args.pi0_bimanual_csv],
        bimanual_model_names=["ACT - Bimanual", "Pi0.5 - Bimanual"],
        output_dir=args.output_dir,
    )