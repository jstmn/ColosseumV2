import os
import h5py
import argparse
from tqdm import tqdm

def is_traj_file(filename):
    return filename.startswith("traj_") and filename.endswith(".h5")

def collect_traj_files(root):
    traj_files = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if is_traj_file(fname):
                traj_files.append(os.path.join(dirpath, fname))
    traj_files.sort()
    return traj_files

def merge_h5_files(traj_files, output_file):
    """
    Merge a list of trajectory .h5 files into a single output file.
    Each file becomes a group: /traj_0000, /traj_0001, etc.

    If you need concatenation instead of grouping, I can adapt this.
    """
    if len(traj_files) == 0:
        print("No traj_*.h5 files found.")
        return

    print(f"Merging {len(traj_files)} trajectories into: {output_file}")

    with h5py.File(output_file, "w") as fout:
        for i, fpath in enumerate(tqdm(traj_files, desc="Merging")):
            group_name = f"traj_{i}"
            with h5py.File(fpath, "r") as fin:
                fout.copy(fin, group_name)

    print("Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Folder containing experiment folders")
    parser.add_argument("--output", type=str, default="traj.h5", help="Name of merged output file")
    args = parser.parse_args()

    traj_files = collect_traj_files(args.root)
    print(f"Found {len(traj_files)} trajectory files.")

    merge_h5_files(traj_files, args.output)


if __name__ == "__main__":
    main()