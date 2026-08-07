"""Single-version composite video: input | per-camera GT row over prediction row."""

from pathlib import Path
from typing import List

import cv2
import imageio.v3 as iio
import numpy as np
import tyro
from tqdm import trange

from rhino.evaluation.video_utils import draw_label, encode_video


def create_video_multiview(
    seq_name: str,
    version_name: str,
    data_dir: Path,
    render_dir: Path,
    cam_ids: list,
):
    img_dir = data_dir / "image"
    pred_dir = render_dir / "render_pred"
    stage_gt_dir = render_dir.parents[1] / "render_gt"
    iphone_gt_dir = render_dir / "render_gt"
    output_dir = render_dir / "render_all"
    output_dir.mkdir(exist_ok=True)
    for stale in output_dir.glob("*.jpg"):
        stale.unlink()
    if not cam_ids:
        raise ValueError("cam_ids must not be empty")
    print(f"Compositing {seq_name}/{version_name} from {pred_dir}")
    frame_ids = sorted(
        int(p.stem.rsplit("_", 1)[1])
        for p in pred_dir.glob(f"{cam_ids[0]}_*.jpg")
    )
    num_frames = len(frame_ids)
    composite = None
    for i in trange(num_frames):
        idx = frame_ids[i]
        img = iio.imread(img_dir / f"{idx:04d}.png")
        h, w, _ = img.shape
        gt_lists = []
        pred_lists = []
        draw_label(img, f"Input {idx:04d}")
        for cam_id in cam_ids:
            cam_gt_dir = iphone_gt_dir if cam_id == "iphone" else stage_gt_dir
            gt = iio.imread(cam_gt_dir / f"{cam_id}_{idx:04d}.jpg")
            pred = iio.imread(pred_dir / f"{cam_id}_{idx:04d}.jpg")
            gt = cv2.resize(gt, (int(gt.shape[1] * h / gt.shape[0]), h))
            pred = cv2.resize(pred, (int(pred.shape[1] * h / pred.shape[0]), h))
            cam_str = "iPhone" if cam_id == "iphone" else f"Stage Cam {cam_id}"
            draw_label(gt, f"GT {cam_str}")
            draw_label(pred, f"RHINO {cam_str}")
            gt_lists.append(gt)
            pred_lists.append(pred)
        gt = np.concatenate([img] + gt_lists, axis=1)
        pred = np.concatenate([img] + pred_lists, axis=1)
        composite = np.concatenate([gt, pred], axis=0)
        iio.imwrite(output_dir / f"{i:04d}.jpg", composite)

    if composite is None:
        raise FileNotFoundError(
            f"No frames rendered for composite video: {pred_dir} has no {cam_ids[0]}_*.jpg"
        )

    encode_video(
        output_dir,
        render_dir / f"{seq_name}-{version_name}.mp4",
        width=composite.shape[1],
        height=composite.shape[0],
    )


def main(
    seq_name: str,
    version_name: str,
    cam_ids: List[str],
    output_dir: Path,
    data_dir: Path,
    subdir: str = "hoi",
):
    """Single-version composite mp4 from render_pred/ + shared render_gt/."""
    output_dir = Path(output_dir)
    data_dir = Path(data_dir)
    render_dir = output_dir / seq_name / subdir / version_name
    seq_data_dir = data_dir / seq_name / "processed"
    create_video_multiview(
        seq_name=seq_name,
        version_name=version_name,
        data_dir=seq_data_dir,
        render_dir=render_dir,
        cam_ids=cam_ids,
    )


if __name__ == "__main__":
    tyro.cli(main)
