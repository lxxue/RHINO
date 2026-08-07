"""Shared path-resolution helpers for the preprocessing pipeline."""

import sys
from pathlib import Path
from typing import Optional


def resolve_camera_pose_dir(
    data_dir: Path,
    sfm_tool: str,
    feature_type: str,
    matching_method: str,
    matcher_type: str,
    camera_use_mast3r: bool = False,
    camera_use_glomap: bool = False,
    camera_pair_window_size: int = 5,
    record3d_num_ba_iterations: int = 3,
) -> Path:
    """Return the sparse-model directory produced by step 2."""
    if camera_use_mast3r:
        mapper = "glomap" if camera_use_glomap else "colmap"
        return (
            data_dir
            / "camera_poses"
            / "mast3r"
            / f"{mapper}_sparse_{camera_pair_window_size}"
            / "0"
        )
    if sfm_tool == "hloc":
        return (
            data_dir
            / "localization"
            / f"hloc-{feature_type}-{matching_method}-{matcher_type}"
            / "sparse"
            / "0"
        )
    if sfm_tool == "colmap":
        return (
            data_dir
            / "localization"
            / f"colmap-sift-{matching_method}-NN-ratio"
            / "sparse"
            / "0"
        )
    if sfm_tool == "record3d":
        if record3d_num_ba_iterations < 0:
            raise ValueError("record3d_num_ba_iterations must be non-negative")
        sparse_dir = (
            data_dir
            / "localization"
            / f"record3d-{feature_type}-{matching_method}-{matcher_type}"
            / "sparse"
        )
        if record3d_num_ba_iterations == 0:
            return sparse_dir / "0"
        return sparse_dir / f"{record3d_num_ba_iterations - 1}_ba"
    raise ValueError(
        f"Unsupported scene SfM tool {sfm_tool!r}; expected hloc, colmap, or record3d"
    )


def resolve_sequence_dir(
    data_dir: Optional[Path], data_root: Optional[Path], subject: Optional[str]
) -> Path:
    """Resolve the sequence directory from --data-dir or --data-root + --subject.

    If both are supplied, --data-dir wins (with a warning on mismatch).
    --data-root + --subject searches stage/<subject> then inthewild/<subject>,
    requiring exactly one match.
    """
    if data_root is not None and subject is not None:
        candidates = [
            data_root / "stage" / subject,
            data_root / "inthewild" / subject,
        ]
        matches = [p for p in candidates if p.is_dir()]
        if not matches:
            print(
                f"Error: --subject {subject!r} not found under "
                f"{data_root}/stage/ or {data_root}/inthewild/",
                file=sys.stderr,
            )
            sys.exit(1)
        if len(matches) > 1:
            print(
                f"Error: --subject {subject!r} is ambiguous — exists in "
                f"both stage/ and inthewild/ under {data_root}. Pass --data-dir explicitly.",
                file=sys.stderr,
            )
            sys.exit(1)
        resolved = matches[0]
        if data_dir is not None and data_dir.resolve() != resolved.resolve():
            print(
                f"Warning: --data-dir {data_dir} != resolved --data-root/--subject "
                f"{resolved}; using --data-dir.",
                file=sys.stderr,
            )
            return data_dir
        return resolved
    if data_dir is not None:
        return data_dir
    print(
        "Error: must provide --data-dir, or --data-root together with --subject.",
        file=sys.stderr,
    )
    sys.exit(1)
