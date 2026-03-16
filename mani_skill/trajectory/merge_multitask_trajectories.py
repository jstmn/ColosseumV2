import argparse
from pathlib import Path
import h5py
import numpy as np
from mani_skill.utils.logging_utils import logger
from mani_skill.utils.io_utils import dump_json, load_json


def merge_trajectories(output_path: str, traj_paths: list, recompute_id: bool = True):
    logger.info(f"Merging {output_path}")

    merged_h5_file = h5py.File(output_path, "w")
    merged_json_path = output_path.replace(".h5", ".json")
    merged_json_data = {"episodes": []}
    cnt = 0
    episode_lengths_by_env = {}

    all_env_ids = set()
    merge_summary = []

    for traj_path in traj_paths:
        traj_path = str(traj_path)
        file_episode_cnt = 0
        logger.info(f"Merging {traj_path}")

        try:
            h5_file = h5py.File(traj_path, "r")
        except OSError as e:
            logger.error(f"Failed to open {traj_path}: {e}")
            raise e
        json_data = load_json(traj_path.replace(".h5", ".json"))

        assert "env_info" in json_data and "env_id" in json_data["env_info"], f"'env_info.env_id' not found in {traj_path}"
        cur_env_id = json_data["env_info"]["env_id"]
        all_env_ids.add(cur_env_id)

        for key, value in json_data.items():
            if key == "episodes":
                continue
            if key not in merged_json_data:
                merged_json_data[key] = value
            else:
                if merged_json_data[key] != value:
                    logger.warning(
                        f"Conflict detected for key {key} in {traj_path}: "
                        f"{merged_json_data[key]} != {value}"
                    )

        for ep in json_data["episodes"]:
            ep_length = ep.get("elapsed_steps", 0)
        
            if cur_env_id not in episode_lengths_by_env:
                episode_lengths_by_env[cur_env_id] = []
            episode_lengths_by_env[cur_env_id].append(ep_length)

            old_episode_id = ep["episode_id"]
            old_traj_id = f"traj_{old_episode_id}"

            new_traj_id = f"traj_{cnt}" if recompute_id else old_traj_id
            assert new_traj_id not in merged_h5_file, new_traj_id
            h5_file.copy(old_traj_id, merged_h5_file, new_traj_id)

            new_ep = dict(ep)
            if recompute_id:
                new_ep["episode_id"] = cnt

            new_ep["env_id"] = cur_env_id
            new_ep["source_traj_path"] = traj_path
            new_ep["source_episode_id"] = old_episode_id

            merged_json_data["episodes"].append(new_ep)
            cnt += 1
            file_episode_cnt += 1
        h5_file.close()
        
        merge_summary.append((cur_env_id, file_episode_cnt))

    print("\n" + "="*80)
    print(f"{'Env ID':<35} | {'Count':<10} | {'Mean Episode Length':<10} | {'Std Episode Length':<10} | {'Max Episode Length':<10}")
    print("-" * 80)
    for task_name, count in merge_summary:
        lens = episode_lengths_by_env[task_name]
        print(f"{task_name:<35} | {count:<10} | {round(float(np.mean(lens)), 2):<10} | {round(float(np.std(lens)), 2):<10} | {round(float(np.max(lens)), 2):<10}")
    print("-" * 80)
    print(f"{'Total Merged Episodes':<35} | {cnt:<10}")
    print("="*80 + "\n")

    merged_json_data["multi_env"] = True
    merged_json_data["env_ids"] = sorted(list(all_env_ids))
    merged_json_data["episode_lengths_by_env"] = episode_lengths_by_env

    merged_h5_file.close()
    dump_json(merged_json_path, merged_json_data, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dirs", nargs="+", required=True)
    parser.add_argument("-o", "--output-path", type=str, required=True)
    parser.add_argument("-p", "--pattern", type=str, default="trajectory.h5")
    args = parser.parse_args()

    traj_paths = []
    for input_dir in args.input_dirs:
        input_dir = Path(input_dir)
        assert input_dir.exists(), f"Directory not found: {input_dir}"
        traj_paths.extend(sorted(input_dir.rglob(args.pattern)))
    assert len(traj_paths) > 0, "No trajectory files found with the given pattern."
    for path in traj_paths:
        assert path.exists(), f"File not found: {path}"

    output_dir = Path(args.output_path).parent
    output_dir.mkdir(exist_ok=True, parents=True)

    merge_trajectories(args.output_path, traj_paths)


if __name__ == "__main__":
    main()