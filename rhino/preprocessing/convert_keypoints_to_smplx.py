"""Convert Sapiens coco_wholebody keypoints to SMPL-X format.

This script runs under the AiOS Python environment (which has detrsmpl).
It is invoked as a subprocess by estimate_keypoints.py.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure rhino package is importable when invoked with external Python interpreter
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# detrsmpl lives inside the AiOS submodule
_AIOS_ROOT = str(_REPO_ROOT / "third_party" / "AiOS")
if _AIOS_ROOT not in sys.path:
    sys.path.insert(0, _AIOS_ROOT)

import numpy as np
from tqdm import tqdm

from detrsmpl.core.conventions.keypoints_mapping import convert_kps


@dataclass
class ConvertKeypointsArgs:
    sapiens_dir: Path
    """Path to Sapiens JSON output directory."""
    output_dir: Path
    """Path to write SMPL-X keypoints."""


def main(args: ConvertKeypointsArgs) -> None:
    sapiens_dir = args.sapiens_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(sapiens_dir.glob("*.json"))
    if not json_files:
        raise RuntimeError(f"No JSON files in {sapiens_dir}")

    mask_saved = False
    for i, jf in enumerate(tqdm(json_files, desc="Converting to SMPL-X")):
        with open(jf) as f:
            data = json.load(f)
        instances = data.get("instance_info", [])
        if instances:
            kpts_2d = np.array(instances[0]["keypoints"], dtype=np.float32)
            kpts_conf = np.array(instances[0]["keypoint_scores"], dtype=np.float32)[:, None]
            kpts_w_conf = np.hstack([kpts_2d, kpts_conf])[None]
        else:
            kpts_w_conf = np.zeros((1, 133, 3), dtype=np.float32)

        converted, mask = convert_kps(kpts_w_conf, src="coco_wholebody", dst="smplx")
        if not mask_saved:
            np.save(output_dir / "mask.npy", np.array(mask))
            mask_saved = True
        np.save(output_dir / f"{i:04d}.npy", converted[0])

    print(f"Converted {len(json_files)} files → {output_dir}")


if __name__ == "__main__":
    import tyro
    main(tyro.cli(ConvertKeypointsArgs))
