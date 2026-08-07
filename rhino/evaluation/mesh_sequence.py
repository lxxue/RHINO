"""Filesystem contracts for RHINO prediction and ground-truth mesh sequences."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


RHINO_GT_FRAME_COUNTS = {449, 450}
_GT_PATTERN = re.compile(r"mesh-f(\d{5})\.obj")
_PRED_PATTERN = re.compile(r"(\d{4})\.ply")
_EPOCH_PATTERN = re.compile(r"epoch_(\d+)")


@dataclass(frozen=True)
class MeshSequence:
    gt_files: dict[int, Path]
    human_files: dict[int, Path]
    object_files: dict[int, Path]
    frame_pairs: list[tuple[int, int]]


def check_gt_frame_count(
    num_frames_gt: int, expected_gt_frames: Optional[int], where
) -> None:
    if expected_gt_frames is not None:
        if num_frames_gt != expected_gt_frames:
            raise ValueError(
                f"num_frames_gt: {num_frames_gt} != --expected-gt-frames "
                f"{expected_gt_frames} ({where})"
            )
        return
    if num_frames_gt not in RHINO_GT_FRAME_COUNTS:
        raise ValueError(
            f"num_frames_gt: {num_frames_gt} not in {sorted(RHINO_GT_FRAME_COUNTS)} "
            f"({where}). These are the RHINO capture lengths; pass "
            "--expected-gt-frames for other GT."
        )


def _parse_indexed_files(directory: Path, suffix: str, pattern, label: str) -> dict[int, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"{label} directory not found: {directory}")
    paths = sorted(directory.glob(f"*{suffix}"))
    if not paths:
        raise FileNotFoundError(f"No {suffix} files found in {label} directory: {directory}")

    indexed = {}
    invalid_names = []
    for path in paths:
        match = pattern.fullmatch(path.name)
        if match is None:
            invalid_names.append(path.name)
            continue
        frame_id = int(match.group(1))
        if frame_id in indexed:
            raise ValueError(f"Duplicate {label} frame ID {frame_id} in {directory}")
        indexed[frame_id] = path
    if invalid_names:
        raise ValueError(
            f"Unexpected {label} filenames in {directory}: {invalid_names[:5]}"
        )
    return indexed


def _require_contiguous(frame_ids: set[int], label: str, start: int | None = None) -> None:
    first = min(frame_ids) if start is None else start
    expected = set(range(first, max(frame_ids) + 1))
    if frame_ids != expected:
        missing = sorted(expected - frame_ids)
        extra = sorted(frame_ids - expected)
        raise ValueError(
            f"{label} frame IDs are not contiguous from {first}; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )


def find_latest_epoch(version_dir: Path) -> Path:
    if not version_dir.is_dir():
        raise FileNotFoundError(f"Predicted version directory not found: {version_dir}")
    candidates = []
    for path in version_dir.glob("epoch_*"):
        if not path.is_dir():
            continue
        match = _EPOCH_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Invalid epoch directory name: {path}")
        candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No epoch_<number> subdirectory under {version_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def parse_gt_meshes(
    gt_frames_dir: Path, expected_gt_frames: Optional[int] = None
) -> dict[int, Path]:
    gt_files = _parse_indexed_files(gt_frames_dir, ".obj", _GT_PATTERN, "GT mesh")
    check_gt_frame_count(len(gt_files), expected_gt_frames, f"gt_dir={gt_frames_dir}")
    _require_contiguous(set(gt_files), "GT mesh")
    return gt_files


def parse_predicted_meshes(directory: Path, label="predicted mesh") -> dict[int, Path]:
    files = _parse_indexed_files(directory, ".ply", _PRED_PATTERN, label)
    _require_contiguous(set(files), label, start=0)
    return files


def build_frame_pairs(
    predicted_ids: set[int], gt_ids: set[int], offset: int
) -> list[tuple[int, int]]:
    if offset not in gt_ids:
        raise ValueError(
            f"Offset {offset} is not a GT frame ID; available range is "
            f"[{min(gt_ids)}, {max(gt_ids)}]"
        )
    frame_pairs = [
        (pred_id, pred_id + offset)
        for pred_id in sorted(predicted_ids)
        if pred_id + offset in gt_ids
    ]
    if not frame_pairs:
        raise ValueError(f"No overlapping predicted/GT frames for offset {offset}")
    return frame_pairs


def validate_mesh_sequences(
    gt_frames_dir: Path,
    epoch_dir: Path,
    offset: int,
    expected_gt_frames: Optional[int] = None,
) -> MeshSequence:
    gt_files = parse_gt_meshes(gt_frames_dir, expected_gt_frames)
    gt_ids = set(gt_files)

    human_files = parse_predicted_meshes(
        epoch_dir / "mesh_deformed", "predicted human mesh"
    )
    object_files = parse_predicted_meshes(
        epoch_dir / "mesh_obj", "predicted object mesh"
    )
    human_ids = set(human_files)
    object_ids = set(object_files)
    if human_ids != object_ids:
        raise ValueError(
            "Predicted human/object frame IDs differ; "
            f"human_only={sorted(human_ids - object_ids)[:10]}, "
            f"object_only={sorted(object_ids - human_ids)[:10]}"
        )
    frame_pairs = build_frame_pairs(human_ids, gt_ids, offset)
    return MeshSequence(gt_files, human_files, object_files, frame_pairs)
