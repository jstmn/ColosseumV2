"""
Parse a Colosseum-V2 evaluation CSV (written by `examples/baselines/act/eval_rgbd.py`)
and emit a LaTeX table of success rates.

The CSV is expected to have the following columns (same as `eval_rgbd.py`):

checkpoint_path,distraction_set,env_id,control_mode,include_depth,num_eval_episodes,max_episode_steps,message,num_sucessful_episodes,success_percent

Key behavior:
- **Tasks are derived from the CSV** (first-seen order after filtering), no comparison
  against any pre-saved task list.
- If there are multiple entries for the same (task, distraction_set), the success rate
  is **recomputed from totals**: (100 * sum_success / sum_episodes).
- Rows with `num_sucessful_episodes < 0` are ignored (placeholders / incomplete runs).

# Example usage:

python scripts/parse_colosseum_v2_logs.py --results-path logs/results_single_arm.csv
python scripts/parse_colosseum_v2_logs.py --results-path logs/results_bimanual.csv
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

from mani_skill.envs.tasks.tabletop.colosseum_v2.distraction_set import DISTRACTION_SETS


EXPECTED_COLUMNS: tuple[str, ...] = (
    "checkpoint_path",
    "distraction_set",
    "env_id",
    "control_mode",
    "include_depth",
    "num_eval_episodes",
    "max_episode_steps",
    "message",
    # yes, the writer has a typo: "sucessful" (kept for compatibility)
    "num_sucessful_episodes",
    "success_percent",
)

def _escape_latex(s: str) -> str:
    # Minimal escaping for typical strings; env IDs don't include many special chars,
    # but checkpoint paths sometimes do.
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("^", r"\^{}")
        .replace("~", r"\~{}")
    )


def _iter_distraction_sets_in_order() -> list[str]:
    # Dict insertion order is the canonical ordering here.
    return [k.lower() for k in DISTRACTION_SETS.keys()]


def _format_percent(x: object, *, decimals: int) -> str:
    if x is None:
        return "-"
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "-"
    return f"{v:.{decimals}f}"

def _safe_int(x: object) -> int | None:
    if x is None:
        return None
    try:
        return int(float(str(x).strip()))
    except (TypeError, ValueError):
        return None

def build_success_matrix(
    rows: list[dict[str, str]],
    *,
    distraction_sets: Iterable[str] | None,
    checkpoint_path: str | None,
    sort_by_none_sr: bool = True,
) -> tuple[list[str], list[str], dict[tuple[str, str], float | None]]:
    ds_list_raw = list(distraction_sets) if distraction_sets is not None else _iter_distraction_sets_in_order()
    # CSV distraction_set values are normalized to lowercase for aggregation keys.
    ds_list = [str(ds).lower() for ds in ds_list_raw]

    totals: dict[tuple[str, str], tuple[int, int]] = {}
    task_order: list[str] = []
    seen_tasks: set[str] = set()

    matched_any = False
    valid_any = False

    for row in rows:
        for k in ("checkpoint_path", "distraction_set", "env_id", "num_eval_episodes", "num_sucessful_episodes"):
            if k not in row:
                raise ValueError(
                    f"CSV schema mismatch: missing column {k!r}. Got columns: {sorted(row.keys())}"
                )

        if checkpoint_path is not None and row["checkpoint_path"] != checkpoint_path:
            continue
        matched_any = True

        task = row["env_id"]
        ds = str(row["distraction_set"]).lower()

        n_success = _safe_int(row["num_sucessful_episodes"])
        n_episodes = _safe_int(row["num_eval_episodes"])
        if n_success is None or n_episodes is None:
            continue
        if n_success < 0:
            continue
        if n_episodes <= 0:
            continue

        valid_any = True
        if task not in seen_tasks:
            seen_tasks.add(task)
            task_order.append(task)

        prev_succ, prev_eps = totals.get((task, ds), (0, 0))
        totals[(task, ds)] = (prev_succ + n_success, prev_eps + n_episodes)

    if checkpoint_path is not None and not matched_any:
        raise ValueError("No rows matched (after checkpoint filter).")
    if not valid_any:
        raise ValueError("No valid rows (after filtering out negative successful runs).")

    if sort_by_none_sr:
        # Sort tasks by the success rate under the "none" distraction set (descending),
        # placing tasks with missing "none" entries at the end. Keep stability for ties.
        none_sr: dict[str, float | None] = {}
        for t in task_order:
            pair = totals.get((t, "none"))
            if pair is None:
                none_sr[t] = None
                continue
            succ, eps = pair
            none_sr[t] = (100.0 * succ / eps) if eps > 0 else None

        task_order = sorted(
            task_order,
            key=lambda t: (none_sr[t] is None, -(none_sr[t] or 0.0)),
        )

    matrix: dict[tuple[str, str], float | None] = {}
    for t in task_order:
        for ds in ds_list:
            pair = totals.get((t, ds))
            if pair is None:
                matrix[(t, ds)] = None
                continue
            succ, eps = pair
            matrix[(t, ds)] = (100.0 * succ / eps) if eps > 0 else None

    return task_order, ds_list, matrix


def render_latex_table(
    tasks: list[str],
    distraction_sets: list[str],
    matrix: dict[tuple[str, str], float | None],
    *,
    caption: str | None,
    label: str | None,
    decimals: int,
) -> str:
    header_cells = ["Task"] + [c.upper() for c in distraction_sets]

    def get_header_cell(x: str) -> str:
        def to_bold(x: str) -> str:
            return r"\textbf{" + x + "}"

        if x.upper() == "TASK":
            return to_bold("TASK")
        escaped = _escape_latex(x)
        escaped = escaped.replace("_", " ").lower().capitalize()
        return r"\rotatebox{90}{ " + to_bold(escaped) + " }"

    # tabular spec: 1 left column + N right columns
    tabular_spec = "l" + ("r" * len(distraction_sets))
    lines: list[str] = []
    lines.append(r"\centering")
    lines.append(rf"\begin{{tabular}}{{{tabular_spec}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(  get_header_cell(x) for x in header_cells) + r" \\")
    lines.append(r"\midrule")
    # & \rotatebox{90}{NONE}  < we want to rotate the columns 90 deg

    for task in tasks:
        cells = [_escape_latex(str(task))]
        for ds in distraction_sets:
            cells.append(_format_percent(matrix.get((task, ds)), decimals=decimals))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if caption:
        lines.append(rf"\caption{{{_escape_latex(caption)}}}")
    if label:
        lines.append(rf"\label{{{_escape_latex(label)}}}")
    lines.append("")  # trailing newline
    return "\n".join(lines)


def save_to_csv(tasks: list[str], distraction_sets: list[str], matrix: dict[tuple[str, str], float | None], path: str):
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Task"] + distraction_sets)
        for task in tasks:
            writer.writerow([task] + [matrix.get((task, ds)) for ds in distraction_sets])
    print(f"Wrote CSV table to {path}")

def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Parse Colosseum-V2 eval results CSV and emit a LaTeX table.")
    p.add_argument("--results-path", required=True, type=str, help="Path to results CSV (from eval_rgbd.py).")
    p.add_argument(
        "--checkpoint-path",
        default=None,
        type=str,
        help="If provided, filter rows to this exact checkpoint_path.",
    )
    p.add_argument("--decimals", default=1, type=int, help="Decimal places for success_percent.")
    p.add_argument("--caption", default=None, type=str, help="Optional LaTeX caption.")
    p.add_argument("--label", default=None, type=str, help="Optional LaTeX label, e.g. tab:colosseum_results.")

    args = p.parse_args(argv)
    with open(args.results_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    tasks, distraction_sets, matrix = build_success_matrix(
        rows,
        distraction_sets=_iter_distraction_sets_in_order(),
        checkpoint_path=args.checkpoint_path,
    )

    latex = render_latex_table(
        tasks,
        distraction_sets,
        matrix,
        caption=args.caption,
        label=args.label,
        decimals=args.decimals,
    )

    out_tex = args.results_path.replace(".csv", ".tex")
    out_csv = args.results_path.replace(".csv", ".results.csv")
    with open(out_tex, "w") as f:
        f.write(latex)
    print(f"Wrote LaTeX table to {out_tex}")
    save_to_csv(tasks, distraction_sets, matrix, out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


