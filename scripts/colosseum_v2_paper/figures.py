import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any

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
    --result-csvs logs/parsed_ACT/bimanual.formatted.csv logs/parsed_pi0/bimanual.formatted.csv \
                  logs/parsed_ACT/single_arm.formatted.csv logs/parsed_pi0/single_arm.formatted.csv \
    --model-names "ACT - Bimanual" "Pi0.5 - Bimanual" "ACT - Single Arm" "Pi0.5 - Single Arm" \
    --output-dir logs/
"""


DISTRACTION_SETS=(
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

DISTRACTION_SET_DISPLAY_NAMES = {
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

VISION_DISTRACTION_SETS = [
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
LANGUAGE_DISTRACTION_SETS = [
    "language"
]
ACTION_DISTRACTION_SETS = [
    "MO_size",
    "RO_size",
    "pose_randomization",
]
assert len(VISION_DISTRACTION_SETS) + len(ACTION_DISTRACTION_SETS) + len(LANGUAGE_DISTRACTION_SETS) == len(DISTRACTION_SETS) - 2
# ^ 2 not included are 'none' and 'all'



# Don't count tasks with none success rate below this threshold
LOWEST_NONE_SUCCESS_RATE_FOR_COUNTING = 10


def calculate_mean_changes_from_none(result_csvs: List[str], model_names: List[str]):
    mean_changes_from_none = {
        name: {} for name in model_names
    }

    for model_name, result_csv in zip(model_names, result_csvs):
        df = pd.read_csv(result_csv)
        if "Task" not in df.columns:
            raise ValueError(f"CSV {result_csv} missing required 'Task' column. Got columns: {list(df.columns)}")

        # Index by task name for easy lookup: df.loc[task, distraction_set]
        df["Task"] = df["Task"].astype(str).str.strip()
        df = df.set_index("Task", drop=True)
        df = df[df["none"] >= LOWEST_NONE_SUCCESS_RATE_FOR_COUNTING]

        # Allow distraction-set columns to be either exact-case (e.g. MO_color) or lower-case (e.g. mo_color).
        col_by_lower = {str(c).strip().lower(): c for c in df.columns}

        success_rates: Dict[str, Dict[str, float]] = {}
        for task in df.index:
            success_rates[task] = {}
            for distraction_set in DISTRACTION_SETS:
                ds_col = distraction_set
                if ds_col not in df.columns:
                    ds_col = col_by_lower.get(str(distraction_set).lower(), distraction_set)
                if ds_col not in df.columns:
                    raise KeyError(
                        f"CSV {result_csv} missing distraction-set column '{distraction_set}' "
                        f"(also tried '{str(distraction_set).lower()}'). Got columns: {list(df.columns)}"
                    )
                v = df.loc[task, ds_col]
                # Cells may be empty -> NaN; treat as 0.0 success.
                # TODO: ignore tasks with none success rate
                if pd.isna(v):
                    continue
                success_rates[task][distraction_set] = float(v)

        for ds in DISTRACTION_SETS:
            all_changes = []
            for task in success_rates.keys():
                base = success_rates[task]["none"]
                if base <= 0:
                    print(f"Task {task} has none success rate <= 0")
                    continue
                # Mean change: (success[distraction_set] - success[none])
                if ds not in success_rates[task]:
                    print(f"Task {task} has no success rate for distraction set {ds}")
                    continue
                change = success_rates[task][ds] - base
                all_changes.append(change)
            mean_changes_from_none[model_name][ds] = float(np.mean(all_changes)) if all_changes else 0.0
    return mean_changes_from_none


def generate_clumped_change_figure(mean_changes_from_none: Dict[str, Dict[str, float]], model_names: List[str], output_dir: str):
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
        for ds in VISION_DISTRACTION_SETS:
            vision_vals.append(float(mean_changes[ds]))
        for ds in ACTION_DISTRACTION_SETS:
            action_vals.append(float(mean_changes[ds]))
        for ds in LANGUAGE_DISTRACTION_SETS:
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


def generate_mean_change_figure(mean_changes_from_none: Dict[str, Dict[str, float]], model_names: List[str], output_dir: str):

    # Plotting barchart
    distraction_sets = list(DISTRACTION_SETS)
    n_ds = len(distraction_sets)
    n_models = len(model_names)

    # Retrieve mean changes in matrix form for easier plotting
    values = []
    for model in model_names:
        values.append([mean_changes_from_none[model][ds] for ds in distraction_sets])  # absolute change

    x = range(n_ds)
    bar_width = 0.8 / n_models

    plt.figure(figsize=(16, 4))
    for i, model in enumerate(model_names):
        plt.bar(
            [xx + (i - n_models/2) * bar_width + bar_width/2 for xx in x],
            values[i],
            width=bar_width,
            label=model,
        )

    fontsize = 11
    plt.xticks(
        x,
        [DISTRACTION_SET_DISPLAY_NAMES[ds] for ds in distraction_sets],
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
    parser.add_argument("--result-csvs", type=str, required=True, nargs="+")
    parser.add_argument("--model-names", type=str, required=True, nargs="+")
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    mean_changes_from_none = calculate_mean_changes_from_none(args.result_csvs, args.model_names)
    generate_mean_change_figure(mean_changes_from_none, args.model_names, args.output_dir)
    generate_clumped_change_figure(mean_changes_from_none, args.model_names, args.output_dir)