"""Extract the first N frames from one or more mp4 videos as PNGs.

Example:
python scripts/extract_video_frames.py \
    --video demos/RaiseCube-v1/motionplanning/trajectory_human_rt__none___n:1.mp4 \
    --output-dir demos/RaiseCube-v1/motionplanning/render_camera_rt/none \
    --prefix traj_1___ \
    --first-n-frames 65
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def extract_video_frames(
    video_path: str | Path,
    output_dir: str | Path,
    prefix: str = "",
    first_n_frames: int | None = None,
) -> list[Path]:
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    assert video_path.is_file(), f"Video not found: {video_path}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    saved: list[Path] = []
    frame_idx = 0
    try:
        while True:
            if first_n_frames is not None and frame_idx >= first_n_frames:
                break
            ok, frame_bgr = cap.read()
            if not ok:
                break
            out_path = output_dir / f"{prefix}{frame_idx:03d}.png"
            cv2.imwrite(str(out_path), frame_bgr)
            saved.append(out_path)
            print(f"Saved image to {out_path}")
            frame_idx += 1
    finally:
        cap.release()

    if not saved:
        raise RuntimeError(f"No frames extracted from {video_path}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=str, required=True, help="Path to an .mp4 file")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for PNG frames")
    parser.add_argument("--prefix", type=str, default="", help="Filename prefix, e.g. traj_1___")
    parser.add_argument(
        "--first-n-frames",
        type=int,
        default=None,
        help="Only save the first N frames (default: all)",
    )
    args = parser.parse_args()
    extract_video_frames(
        video_path=args.video,
        output_dir=args.output_dir,
        prefix=args.prefix,
        first_n_frames=args.first_n_frames,
    )


if __name__ == "__main__":
    main()
