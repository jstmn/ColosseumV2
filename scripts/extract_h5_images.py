import argparse
import h5py
import os
import cv2
from pathlib import Path
from mani_skill.utils.visualization.misc import images_to_video


""" This script accepts a path to a h5 file. It extracts all the images and saves them to a directory of the same name.
The output format is:

<h5_file_name>/
    camera_name_1/
        traj_1__000.png
        traj_1__001.png
    camera_name_2/
        traj_1__000.png
        traj_1__001.png
    ...

The h5 file format is:
    $ h5ls -r demos/RaiseCube-v1/motionplanning/trajectory.h5
    /                        Group
    /traj_0                  Group
    /traj_0/actions          Dataset {76, 8}
    /traj_0/env_states       Group
    /traj_0/env_states/actors Group
    /traj_0/env_states/actors/cube Dataset {77, 13}
    /traj_0/env_states/actors/table Dataset {77, 13}
    /traj_0/env_states/articulations Group
    /traj_0/env_states/articulations/panda Dataset {77, 31}
    /traj_0/obs              Group
    /traj_0/obs/agent        Group
    /traj_0/obs/agent/qpos   Dataset {77, 9}
    /traj_0/obs/agent/qvel   Dataset {77, 9}
    /traj_0/obs/agent/world__T__ee Dataset {77, 4, 4}
    /traj_0/obs/agent/world__T__root Dataset {77, 4, 4}
    /traj_0/obs/extra        Group
    /traj_0/obs/extra/is_grasped Dataset {77}
    /traj_0/obs/extra/tcp_pose Dataset {77, 7}
    /traj_0/obs/sensor_data  Group
    /traj_0/obs/sensor_data/camera_center Group
    /traj_0/obs/sensor_data/camera_center/rgb Dataset {77, 256, 256, 3}
    /traj_0/obs/sensor_data/camera_left Group
    /traj_0/obs/sensor_data/camera_left/rgb Dataset {77, 256, 256, 3}
    /traj_0/obs/sensor_data/camera_right Group
    /traj_0/obs/sensor_data/camera_right/rgb Dataset {77, 256, 256, 3}
    /traj_0/obs/sensor_param Group
    /traj_0/obs/sensor_param/camera_center Group
    /traj_0/obs/sensor_param/camera_center/cam2world_gl Dataset {77, 4, 4}
    /traj_0/obs/sensor_param/camera_center/extrinsic_cv Dataset {77, 3, 4}
    /traj_0/obs/sensor_param/camera_center/intrinsic_cv Dataset {77, 3, 3}
    /traj_0/obs/sensor_param/camera_left Group
    /traj_0/obs/sensor_param/camera_left/cam2world_gl Dataset {77, 4, 4}
    /traj_0/obs/sensor_param/camera_left/extrinsic_cv Dataset {77, 3, 4}
    /traj_0/obs/sensor_param/camera_left/intrinsic_cv Dataset {77, 3, 3}
    /traj_0/obs/sensor_param/camera_right Group
    /traj_0/obs/sensor_param/camera_right/cam2world_gl Dataset {77, 4, 4}
    /traj_0/obs/sensor_param/camera_right/extrinsic_cv Dataset {77, 3, 4}
    /traj_0/obs/sensor_param/camera_right/intrinsic_cv Dataset {77, 3, 3}
    /traj_0/success          Dataset {76}
    /traj_0/terminated       Dataset {76}
    /traj_0/truncated        Dataset {76}

# Example usage:
python scripts/extract_h5_images.py \
    --h5-file demos/RaiseCube-v1/motionplanning/trajectory_500x500.h5 \
    --first-n-trajectories 3 \
    --first-n-timesteps 65 \
    --save-video
"""

def save_h5_images(h5_file_path: str, first_n_timesteps: int | None = None, first_n_trajectories: int | None = None, prefix: str = "", save_video: bool = False):
    with h5py.File(h5_file_path, 'r') as f:

        # First, create the save directories
        save_dirs = {}
        camera_names = f['traj_0']['obs']['sensor_data'].keys()
        for camera_name in camera_names:
            save_dirs[camera_name] = Path(h5_file_path.replace(".h5", "")).parent / camera_name
            save_dirs[camera_name].mkdir(parents=True, exist_ok=True)

        # Then, save the images
        for traj_key in f.keys():
            assert traj_key.startswith('traj_')

            traj_idx = int(traj_key.split('_')[-1])
            if first_n_trajectories is not None and traj_idx > first_n_trajectories:
                continue

            if save_video:
                video_images = {
                    camera_name: [] for camera_name in camera_names
                }

            traj_group = f[traj_key]
            assert 'obs' in traj_group
            assert 'sensor_data' in traj_group['obs']
            sensor_data = traj_group['obs']['sensor_data']
            for camera_name in sensor_data.keys():
                rgbs = sensor_data[camera_name]['rgb']
                for i, rgb in enumerate(rgbs):
                    image_path = save_dirs[camera_name] / f"{prefix}{traj_key}___{i:03d}.png"
                    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(image_path), rgb_bgr)
                    print(f"Saved image to {image_path}")

                    if save_video:
                        video_images[camera_name].append(rgb)

                    if first_n_timesteps is not None and i > first_n_timesteps:
                        break

            if save_video:
                for camera_name in video_images.keys():
                    images_to_video(
                        images=video_images[camera_name],
                        output_dir=str(save_dirs[camera_name]),
                        video_name=f"{prefix}{traj_key}__video",
                        fps=30,
                        quality=5,
                        verbose=True,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-file", type=str, required=True)
    parser.add_argument("--first-n-timesteps", type=int, required=False, default=None)
    parser.add_argument("--first-n-trajectories", type=int, required=False, default=None)
    parser.add_argument("--save-video", action="store_true", required=False, default=False)
    parser.add_argument("--prefix", type=str, required=False, default="")
    args = parser.parse_args()
    assert os.path.exists(args.h5_file), f"H5 file {args.h5_file} does not exist"
    save_h5_images(
        h5_file_path=args.h5_file,
        first_n_timesteps=args.first_n_timesteps,
        first_n_trajectories=args.first_n_trajectories,
        prefix=args.prefix,
        save_video=args.save_video,
    )