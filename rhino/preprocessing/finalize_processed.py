"""Finalize step: Validate processed/ directory against dataset loader contract.

Checks all files required by the dataset loader, validates frame count consistency,
and writes manifest.json on success.
"""

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from rhino.preprocessing.path_utils import resolve_sequence_dir


REQUIRED_DIRS = ["image", "depth", "normal", "all_mask", "human_mask", "obj_mask", "hand_mask"]
REQUIRED_FILES_BASE = ["intrinsic.npy", "c2ws.npy", "object_poses_centered.npy",
                       "contact_labels.npy", "contact_vertices_refined.npy"]


@dataclass
class FinalizeProcessedArgs:
    data_dir: Optional[Path] = None
    """Path to the sequence data directory. Required unless --data-root + --subject are given."""
    data_root: Optional[Path] = None
    """Root directory containing stage/ and/or inthewild/. Resolves to --data-dir via --subject."""
    subject: Optional[str] = None
    """Sequence subject name under stage/ or inthewild/ inside --data-root."""
    quantization_mode: str = "4bit"
    """InteractVLM quantization mode used in step 12; recorded in manifest.json."""


def count_files(directory: Path, pattern: str) -> int:
    return len(sorted(directory.glob(pattern)))


def main(args: FinalizeProcessedArgs) -> None:
    data_dir = resolve_sequence_dir(args.data_dir, args.data_root, args.subject).resolve()
    processed_dir = data_dir / "processed"
    print(f"Validating processed/ directory at {processed_dir}...")

    errors = []

    # Check processed/ exists
    if not processed_dir.is_dir():
        raise RuntimeError(f"processed/ directory not found at {processed_dir}")

    # Check required directories and count frames
    frame_counts = {}
    for d in REQUIRED_DIRS:
        p = processed_dir / d
        if not p.is_dir():
            errors.append(f"Missing directory: {d}/")
            continue
        if d in ("image", "all_mask", "human_mask", "obj_mask", "hand_mask"):
            count = count_files(p, "*.png")
        elif d == "depth" or d == "normal":
            count = count_files(p, "*.npy")
        else:
            count = 0
        frame_counts[d] = count
        if count == 0:
            errors.append(f"Empty directory: {d}/")

    # Check required base files
    for f in REQUIRED_FILES_BASE:
        if not (processed_dir / f).exists():
            errors.append(f"Missing file: {f}")

    # Check aligned_smplx.pkl (canonical or variant)
    if not (processed_dir / "aligned_smplx.pkl").exists():
        errors.append("Missing file: aligned_smplx.pkl (or symlink from variant)")

    # Validate frame count consistency
    if frame_counts:
        num_images = frame_counts.get("image", 0)
        for name, count in frame_counts.items():
            if count != num_images and count > 0:
                errors.append(f"Frame count mismatch: {name}/ has {count} files, image/ has {num_images}")

        # Check per-frame arrays
        for arr_file in ["c2ws.npy", "contact_labels.npy"]:
            p = processed_dir / arr_file
            if p.exists():
                arr = np.load(p)
                if arr.shape[0] != num_images:
                    errors.append(f"{arr_file}: {arr.shape[0]} frames, expected {num_images}")

        # Check object poses
        for pose_file in ["object_poses_centered.npy", "object_poses.npy"]:
            p = processed_dir / pose_file
            if p.exists():
                arr = np.load(p)
                if arr.shape[0] != num_images:
                    errors.append(f"{pose_file}: {arr.shape[0]} frames, expected {num_images}")

        # Check contact vertices
        cv_path = processed_dir / "contact_vertices_refined.npy"
        if cv_path.exists():
            arr = np.load(cv_path)
            if arr.shape[0] != num_images:
                errors.append(f"contact_vertices_refined.npy: {arr.shape[0]} frames, expected {num_images}")

    if errors:
        print("\nValidation FAILED:")
        for e in errors:
            print(f"  - {e}")
        raise RuntimeError(f"Finalize failed: {len(errors)} validation error(s)")

    # Get submodule SHAs for manifest
    tool_versions = {}
    repo_root = Path(__file__).resolve().parents[2]
    for submod in ["InteractVLM", "AiOS", "sapiens", "Metric3D", "mast3r", "Grounded-SAM-2", "Hierarchical-Localization"]:
        submod_dir = repo_root / "third_party" / submod
        if submod_dir.is_dir():
            try:
                sha = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=str(submod_dir), stderr=subprocess.DEVNULL
                ).decode().strip()
                tool_versions[submod] = sha[:12]
            except Exception:
                tool_versions[submod] = "unknown"

    manifest = {
        "status": "complete",
        "timestamp": datetime.now().isoformat(),
        "smplx_estimator": "AiOS",
        "quantization_mode": args.quantization_mode,
        "num_frames": num_images,
        "tool_versions": tool_versions,
        "files_validated": REQUIRED_DIRS + REQUIRED_FILES_BASE + ["aligned_smplx.pkl"],
    }

    manifest_path = processed_dir / "manifest.json"
    if manifest_path.exists():
        print(f"Warning: overwriting existing manifest at {manifest_path}")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nValidation PASSED. {num_images} frames validated.")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    import tyro
    main(tyro.cli(FinalizeProcessedArgs))
