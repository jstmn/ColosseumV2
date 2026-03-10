""" This is a convenience script to remove a specific key from a h5 file.

Example usage:
python scripts/data_generation/remove_key_from_h5.py \
   --h5-file demos/LiftPegUprightColosseumV2-v1/motionplanning/trajectory__pd_joint_pos__100.rgb.pd_ee_delta_pose.physx_cpu.h5 \
   --key traj_91 \
   --dry-run
"""

import argparse
import h5py
import os


def remove_key_from_h5(h5_file: str, key: str, dry_run: bool = False):
    with h5py.File(h5_file, "r+") as f:
        if key in f:
            if dry_run:
                print(f"Would remove key {key} from {h5_file}")
            else:
                del f[key]
                print(f"Removed key {key} from {h5_file}")
        else:
            print(f"Key {key} not found in {h5_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-file", type=str, required=True)
    parser.add_argument("--key", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    assert os.path.exists(args.h5_file), f"File not found: {args.h5_file}"
    remove_key_from_h5(args.h5_file, args.key, args.dry_run)

if __name__ == "__main__":
    main()