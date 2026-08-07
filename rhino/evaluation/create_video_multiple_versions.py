"""Multi-version comparison video: one GT row plus one row per training version."""

from functools import partial
from pathlib import Path
from typing import List, Optional

import cv2
import imageio.v3 as iio
import numpy as np
import tyro
from tqdm import trange

from rhino.evaluation.video_utils import draw_label, encode_video

draw_big_label = partial(
    draw_label,
    org=(30, 200),
    font=cv2.FONT_HERSHEY_DUPLEX,
    scale=7,
    thickness=12,
    outline=24,
)


def create_comparison(
    seq_name: str,
    version_dirs: list,
    labels: list,
    gt_dir: Path,
    data_dir: Path,
    output_dir: Path,
    cam_ids: list,
):
    """Create comparison jpgs + mp4: top row GT, bottom rows one per version.

    `cam_ids` must contain only stage cameras (iPhone is filtered at the CLI
    entrypoint because its GT is per-version-coupled and cannot be shared
    across runs).
    """
    print(f"Creating comparison for {seq_name}")
    print(f"Versions: {labels}")
    print(f"Cameras: {cam_ids}")

    output_dir.mkdir(exist_ok=True, parents=True)
    for stale in output_dir.glob("*.jpg"):
        stale.unlink()

    first_pred_dir = version_dirs[0] / "render_pred"
    first_cam = cam_ids[0]
    frame_ids = sorted(
        int(p.stem.rsplit("_", 1)[1])
        for p in first_pred_dir.glob(f"{first_cam}_*.jpg")
    )
    num_frames = len(frame_ids)
    print(f"Number of frames: {num_frames}")
    if num_frames == 0:
        raise FileNotFoundError(
            f"No frames found in {first_pred_dir} matching {first_cam}_*.jpg"
        )

    ref_img = iio.imread(first_pred_dir / f"{first_cam}_{frame_ids[0]:04d}.jpg")
    cam_height, cam_width = ref_img.shape[:2]
    print(f"Camera dimensions: {cam_width}x{cam_height}")

    img_dir = data_dir / "image"

    composite = None
    for i in trange(num_frames):
        idx = frame_ids[i]
        rows = []

        input_img = iio.imread(img_dir / f"{idx:04d}.png")
        input_h, input_w = input_img.shape[:2]
        scale = cam_height / input_h
        input_resized = cv2.resize(input_img, (int(input_w * scale), cam_height))

        gt_images = [draw_big_label(input_resized.copy(), "GT")]
        for cam_id in cam_ids:
            gt_path = gt_dir / f"{cam_id}_{idx:04d}.jpg"
            if gt_path.exists():
                gt_img = iio.imread(gt_path)
                if gt_img.shape[:2] != (cam_height, cam_width):
                    gt_img = cv2.resize(gt_img, (cam_width, cam_height))
            else:
                gt_img = np.zeros((cam_height, cam_width, 3), dtype=np.uint8)
            cam_label = "iPhone" if cam_id == "iphone" else f"Cam {cam_id}"
            gt_images.append(draw_big_label(gt_img, cam_label))

        rows.append(np.concatenate(gt_images, axis=1))

        for version_dir, label in zip(version_dirs, labels):
            render_pred_dir = version_dir / "render_pred"
            cam_images = [draw_big_label(input_resized.copy(), label)]

            for cam_id in cam_ids:
                cam_path = render_pred_dir / f"{cam_id}_{idx:04d}.jpg"
                if cam_path.exists():
                    cam_img = iio.imread(cam_path)
                    if cam_img.shape[:2] != (cam_height, cam_width):
                        cam_img = cv2.resize(cam_img, (cam_width, cam_height))
                else:
                    cam_img = np.zeros((cam_height, cam_width, 3), dtype=np.uint8)
                cam_images.append(cam_img)

            rows.append(np.concatenate(cam_images, axis=1))

        composite = np.concatenate(rows, axis=0)
        composite = cv2.resize(composite, (composite.shape[1] // 2, composite.shape[0] // 2))

        iio.imwrite(output_dir / f"{i:04d}.jpg", composite)

    if composite is None:
        raise FileNotFoundError("No composite frames were produced.")

    encode_video(
        output_dir,
        output_dir / f"{seq_name}-comparison.mp4",
        width=composite.shape[1],
        height=composite.shape[0],
    )
    print(f"Images saved to: {output_dir}")


def main(
    seq_name: str,
    version_names: List[str],
    subdirs: List[str],
    cam_ids: List[str],
    output_dir: Path,
    data_dir: Path,
    version_labels: Optional[List[str]] = None,
):
    """Multi-version composite mp4 with one GT row + one row per version.

    iPhone is filtered from `cam_ids` because its GT render is anchored to
    each version's `init_trans.npy` (per-ICP), so no single iPhone GT is
    version-independent. For iPhone-view comparison, use `create_video.py`
    per version.
    """
    output_dir = Path(output_dir)
    data_dir = Path(data_dir)
    if version_labels is None:
        version_labels = list(version_names)

    if not (len(version_names) == len(subdirs) == len(version_labels)):
        raise ValueError(
            "version_names, subdirs, and version_labels must have the same length"
        )

    if "iphone" in cam_ids:
        print(
            "[multi-version] Skipping iPhone from comparison: iPhone GT is "
            "per-version-coupled (depends on each version's ICP init_trans.npy). "
            "Use create_video.py per version to inspect iPhone-view GT/pred."
        )
        cam_ids = [c for c in cam_ids if c != "iphone"]
    if not cam_ids:
        raise ValueError(
            "Multi-version comparison requires at least one stage camera; "
            "iPhone alone is version-coupled and cannot be shared across runs."
        )

    version_dirs = [
        output_dir / seq_name / sd / vn for vn, sd in zip(version_names, subdirs)
    ]
    gt_dir = output_dir / seq_name / "render_gt"
    seq_data_dir = data_dir / seq_name / "processed"
    comp_out_dir = (
        output_dir / seq_name / f"comparison_{'-'.join(version_labels)}"
    )

    create_comparison(
        seq_name=seq_name,
        version_dirs=version_dirs,
        labels=version_labels,
        gt_dir=gt_dir,
        data_dir=seq_data_dir,
        output_dir=comp_out_dir,
        cam_ids=cam_ids,
    )


if __name__ == "__main__":
    tyro.cli(main)
