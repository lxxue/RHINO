"""Step 11: Contact frame detection via hand-object proximity + object motion.

Hybrid heuristic (see rhino/preprocessing/contact_heuristic.py): per-frame
hand-object distance with hysteresis, seeded by proximity or object motion.
One global parameter set works across all BenchRHINO sequences (tuned
against volumetric-capture GT) — no per-sequence tuning.

Inputs (processed/): aligned_smplx.pkl, object_poses.npy,
object_poses_centered.npy, object_pcd_centered.ply.
Outputs: processed/contact_labels.npy (bool (F,)), plus
processed/contact_debug.npz with the per-frame signal traces for QA.

``--motion-only`` falls back to the simpler motion+padding heuristic (no SMPL-X
or hand data needed), for sequences with unreliable hand estimates.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rhino.preprocessing.contact_heuristic import (
    hand_object_distances,
    hybrid_labels,
    motion_flags,
    relative_object_motion,
    to_spans,
)


@dataclass
class DetectContactFramesArgs:
    data_dir: Path
    """Path to the sequence data directory."""
    seq_name: str = ""
    """Sequence name for gender lookup in sequence_take_mapping.json."""
    gender: str = ""
    """SMPL-X gender ('male'/'female'/'neutral'). Overrides the seq_name lookup."""
    percentile: int = 5
    """Percentile of per-hand vertex distances used as the hand-object distance."""
    tau_on: float = 0.02
    """Proximity seed threshold (meters): closer than this counts as contact."""
    tau_off: float = 0.08
    """Hysteresis threshold (meters): spans grow while distance stays below this,
    and object motion only counts while hands are within it."""
    median_window: int = 1
    """Median filter window over the motion magnitudes (1 = off)."""
    gap_fill: int = 5
    """Fill non-contact gaps shorter than this many frames."""
    min_span: int = 15
    """Drop contact spans shorter than this many frames."""
    motion_only: bool = False
    """Use the motion+padding fallback; no hand data needed."""
    device: str = "cuda"
    """Device for the SMPL-X forward pass."""


# Span-padding parameters, used only by the --motion-only fallback
_FALLBACK_MIN_SPAN = 10
_FALLBACK_EXTEND_BEFORE = 55
_FALLBACK_EXTEND_AFTER = 35


def motion_only_labels(object_poses: np.ndarray) -> np.ndarray:
    """Motion-only fallback: motion thresholds + short-span filter + padding."""
    rel_trans, rel_rot = relative_object_motion(object_poses)
    contact_labels = motion_flags(rel_trans, rel_rot)
    contact_labels[0] = False
    num_frames = len(contact_labels)

    # Filter short spans
    prev_label = contact_labels[0]
    start_idx = 0
    current_count = _FALLBACK_MIN_SPAN
    i = 1
    while i < num_frames:
        if contact_labels[i] == prev_label:
            current_count += 1
            i += 1
        else:
            if current_count < _FALLBACK_MIN_SPAN:
                contact_labels[start_idx:i] = not prev_label
                i = 1
                prev_label = contact_labels[0]
                start_idx = 0
                current_count = 1
            prev_label = contact_labels[i]
            current_count = 1
            start_idx = i
            i += 1
    if current_count < _FALLBACK_MIN_SPAN:
        contact_labels[start_idx:] = not prev_label

    # Asymmetric temporal extension
    extended = np.zeros(num_frames, dtype=bool)
    for i in range(num_frames):
        if i - _FALLBACK_EXTEND_BEFORE >= 0 and contact_labels[i - _FALLBACK_EXTEND_BEFORE]:
            extended[i] = True
        elif i + _FALLBACK_EXTEND_AFTER < num_frames and contact_labels[i + _FALLBACK_EXTEND_AFTER]:
            extended[i] = True
        else:
            extended[i] = contact_labels[i]
    return extended


def resolve_gender(args: DetectContactFramesArgs) -> str:
    if args.gender:
        return args.gender
    if args.seq_name:
        mapping_path = _REPO_ROOT / "rhino" / "confs" / "sequence_take_mapping.json"
        if mapping_path.exists():
            with open(mapping_path) as f:
                mapping = json.load(f)
            for seq in mapping.get("sequences", []):
                if seq["seq_name"] == args.seq_name and "gender" in seq:
                    return seq["gender"]
    raise RuntimeError(
        "SMPL-X gender is required for the proximity heuristic: pass --gender, "
        "add the sequence to rhino/confs/sequence_take_mapping.json, or use --motion-only."
    )


def main(args: DetectContactFramesArgs) -> None:
    processed_dir = args.data_dir.resolve() / "processed"
    object_poses = np.load(processed_dir / "object_poses.npy")
    num_frames = object_poses.shape[0]

    if args.motion_only:
        print("Detecting contact frames (motion-only fallback)...")
        labels = motion_only_labels(object_poses)
        debug = {}
    else:
        gender = resolve_gender(args)
        print(f"Detecting contact frames (proximity+motion, gender={gender}, "
              f"p{args.percentile}, tau_on={args.tau_on}, tau_off={args.tau_off})...")
        dist_pcts = hand_object_distances(
            processed_dir, gender, [args.percentile], device=args.device
        )
        d = dist_pcts[:, :, 0].min(axis=1)
        rel_trans, rel_rot = relative_object_motion(object_poses)
        labels = hybrid_labels(
            d, rel_trans, rel_rot,
            tau_on=args.tau_on, tau_off=args.tau_off,
            median_window=args.median_window,
            gap_fill=args.gap_fill, min_span=args.min_span,
        )
        debug = {
            "hand_object_distance": d,
            "dist_per_hand": dist_pcts[:, :, 0],
            "rel_trans": rel_trans,
            "rel_rot_deg": rel_rot,
        }

    np.save(processed_dir / "contact_labels.npy", labels)
    if debug:
        np.savez(processed_dir / "contact_debug.npz", **debug)

    spans = ", ".join(f"[{s}, {e}]" for s, e in to_spans(labels))
    print(f"Contact frames: {labels.sum()}/{num_frames} ({100 * labels.mean():.1f}%), spans: {spans}")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(DetectContactFramesArgs))
