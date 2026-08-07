"""Validate preprocessing steps 11-13 (contact detection + vertices) against a reference.

Structural invariants (shape, dtype, span count, contact-frame gating, hand-only
mask) are HARD failures; numeric drift (counts, IoU, Pearson) is logged as
ADVISORY because the RHINO contact heuristic intentionally differs from the
reference run's (proximity+motion vs motion+padding) and quantized InteractVLM can
perturb totals without breaking downstream consumers. Reference statistics are
derived from --reference-dir; absolute contact quality can additionally be
checked against volumetric-capture GT labels via --gt-labels/--gt-offset.
Writes validate_contact_report.json and exits 1 on any hard error, 0 otherwise.

Layout source of truth for the tensors compared here:
  - contact_labels.npy           : detect_contact_frames.py (1-D bool, length = num_frames)
  - contact_vertices.npy         : extract_contact_vertices.py (2-D bool, [num_frames, num_smplx_vertices],
                                   nonzero only on contact frames since step 12 runs contact-frames-only)
  - contact_vertices_refined.npy : refine_contact.py (2-D bool, [num_frames, num_smplx_vertices])
"""

import json
import pickle as pkl
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from rhino.preprocessing.extract_contact_vertices import check_interactvlm_assets
from rhino.preprocessing.tool_config import ToolConfig, require_tools

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HAND_IDS_PATH = _REPO_ROOT / "rhino" / "checkpoints" / "smplx" / "MANO_SMPLX_vertex_ids.pkl"

_ADVISORY_COUNT_TOL_STEP11 = 0.10
_ADVISORY_SPAN_FRAME_TOL = 8
_ADVISORY_COUNT_TOL_STEP12 = 0.25
_ADVISORY_COUNT_TOL_STEP13 = 0.25
_ADVISORY_NONZERO_FRAMES_MIN_FRAC = 0.95
_ADVISORY_PEARSON_MIN = 0.80
_ADVISORY_IOU_MIN = 0.60
_ADVISORY_GT_F1_MIN = 0.90


@dataclass
class ValidateContactArgs:
    rhino_dir: Path
    """Path to the RHINO processed/ directory (contains contact_*.npy)."""
    reference_dir: Optional[Path] = None
    """Optional read-only reference processed/ directory for advisory IoU/Pearson."""
    report_path: Optional[Path] = None
    """Path for the JSON report. Defaults to <rhino_dir parent>/validate_contact_report.json."""
    steps: str = "11,12,13"
    """Comma-separated subset of steps (11,12,13) to validate."""
    expected_frames: Optional[int] = None
    """Hard-checked frame count. Defaults to the processed/image/*.png count."""
    max_spans: int = 3
    """Hard-checked maximum number of contiguous contact spans (>= 1 required)."""
    gt_labels: Optional[Path] = None
    """Optional GT contact_labels.npy (volumetric capture) for an advisory F1 check."""
    gt_offset: int = 0
    """Frame offset for the GT labels: gt_frame = pred_frame + gt_offset."""
    num_smplx_vertices: int = 10475
    """Hard-checked SMPL-X vertex count."""
    contact_type: str = "hcontact"
    """InteractVLM contact_type used in step 12 (for asset preflight)."""
    skip_interactvlm_preflight: bool = False
    """Disable the hard InteractVLM asset preflight invoked from the step-12 block."""
    tools: ToolConfig = field(default_factory=ToolConfig)


def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _load_hand_ids() -> tuple[np.ndarray, np.ndarray]:
    with open(_HAND_IDS_PATH, "rb") as f:
        d = pkl.load(f)
    hand_ids = np.concatenate([d["left_hand"], d["right_hand"]]).astype(np.int64)
    non_hand_ids = np.setdiff1d(np.arange(10475), hand_ids).astype(np.int64)
    return hand_ids, non_hand_ids


def _count_spans(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    edges = np.diff(mask.astype(np.int8))
    starts = int((edges == 1).sum()) + (1 if mask[0] else 0)
    return starts


def _pct_delta(value: float, reference: float) -> float:
    if reference == 0:
        return float("inf") if value != 0 else 0.0
    return abs(value - reference) / reference


def _pearson(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return float(inter / max(union, 1))


def validate_step11(rhino_dir: Path, reference_dir: Optional[Path],
                    results: dict, errors: list, advisories: list,
                    expected_frames: int, max_spans: int = 3,
                    gt_labels: Optional[Path] = None, gt_offset: int = 0) -> None:
    section = "step11_contact_labels"
    results[section] = {"hard": {}, "advisory": {}}
    hard = results[section]["hard"]
    adv = results[section]["advisory"]

    path = rhino_dir / "contact_labels.npy"
    if not path.is_file():
        errors.append(f"{section}: missing artifact {path}")
        hard["structural_ok"] = False
        return

    arr = np.load(path)
    hard["path"] = str(path)
    hard["shape"] = list(arr.shape)
    hard["dtype"] = str(arr.dtype)

    structural_ok = True
    if arr.dtype != np.bool_:
        errors.append(f"{section}: dtype {arr.dtype} != bool")
        structural_ok = False
    if arr.shape != (expected_frames,):
        errors.append(f"{section}: shape {tuple(arr.shape)} != ({expected_frames},)")
        structural_ok = False

    if structural_ok:
        num_spans = _count_spans(arr)
        hard["num_contiguous_spans"] = num_spans
        if arr.any():
            idx = np.where(arr)[0]
            first, last, n_true = int(idx[0]), int(idx[-1]), int(arr.sum())
        else:
            first = last = -1
            n_true = 0
        hard["first_true"] = first
        hard["last_true"] = last
        hard["sum"] = n_true
        if not 1 <= num_spans <= max_spans:
            errors.append(
                f"{section}: expected 1..{max_spans} contiguous positive spans, got {num_spans}"
            )
            structural_ok = False

        if structural_ok and reference_dir is not None:
            ref_path = reference_dir / "contact_labels.npy"
            if ref_path.is_file():
                try:
                    ref_arr = np.load(ref_path)
                    if ref_arr.shape == arr.shape:
                        ref_idx = np.where(ref_arr)[0]
                        ref_sum = int(ref_arr.sum())
                        adv["sum"] = n_true
                        adv["ref_sum"] = ref_sum
                        adv["sum_delta_pct"] = _pct_delta(n_true, ref_sum)
                        if adv["sum_delta_pct"] > _ADVISORY_COUNT_TOL_STEP11:
                            advisories.append(
                                f"{section}: sum {n_true} differs from reference {ref_sum} "
                                f"by {adv['sum_delta_pct'] * 100:.1f}% "
                                f"(tol ±{_ADVISORY_COUNT_TOL_STEP11 * 100:.0f}%)"
                            )
                        if len(ref_idx) > 0 and n_true > 0:
                            first_delta = abs(first - int(ref_idx[0]))
                            last_delta = abs(last - int(ref_idx[-1]))
                            adv["first_delta_frames"] = first_delta
                            adv["last_delta_frames"] = last_delta
                            if first_delta > _ADVISORY_SPAN_FRAME_TOL:
                                advisories.append(
                                    f"{section}: span start {first} differs from reference "
                                    f"{int(ref_idx[0])} by {first_delta} frames "
                                    f"(tol ±{_ADVISORY_SPAN_FRAME_TOL})"
                                )
                            if last_delta > _ADVISORY_SPAN_FRAME_TOL:
                                advisories.append(
                                    f"{section}: span end {last} differs from reference "
                                    f"{int(ref_idx[-1])} by {last_delta} frames "
                                    f"(tol ±{_ADVISORY_SPAN_FRAME_TOL})"
                                )
                        adv["agreement_vs_reference"] = float((ref_arr == arr).mean())
                    else:
                        advisories.append(
                            f"{section}: reference shape {ref_arr.shape} != RHINO shape {arr.shape}"
                        )
                except Exception as ex:
                    advisories.append(f"{section}: failed to load reference {ref_path}: {ex}")

        if structural_ok and gt_labels is not None and gt_labels.is_file():
            gt = np.load(gt_labels).astype(bool)
            lo = max(0, -gt_offset)
            hi = min(len(arr), len(gt) - gt_offset)
            if hi > lo:
                p = arr[lo:hi]
                g = gt[lo + gt_offset:hi + gt_offset]
                tp = float((p & g).sum())
                prec = tp / max(p.sum(), 1)
                rec = tp / max(g.sum(), 1)
                f1 = 2 * prec * rec / max(prec + rec, 1e-8)
                adv["gt_f1"] = f1
                adv["gt_precision"] = prec
                adv["gt_recall"] = rec
                adv["gt_overlap_frames"] = int(hi - lo)
                if f1 < _ADVISORY_GT_F1_MIN:
                    advisories.append(
                        f"{section}: F1 vs GT labels {f1:.3f} below advisory floor "
                        f"{_ADVISORY_GT_F1_MIN:.2f} (P={prec:.3f} R={rec:.3f})"
                    )
            else:
                advisories.append(f"{section}: GT labels have no frame overlap (offset {gt_offset})")

    hard["structural_ok"] = structural_ok


def validate_step12(rhino_dir: Path, reference_dir: Optional[Path],
                    results: dict, errors: list, advisories: list,
                    expected_frames: int, num_smplx_vertices: int,
                    tools: Optional[ToolConfig] = None,
                    contact_type: str = "hcontact",
                    skip_interactvlm_preflight: bool = False) -> None:
    section = "step12_contact_vertices"
    results[section] = {"hard": {}, "advisory": {}}
    hard = results[section]["hard"]
    adv = results[section]["advisory"]

    preflight = {"ran": False, "ok": None, "skipped": skip_interactvlm_preflight}
    if not skip_interactvlm_preflight:
        cfg = tools if tools is not None else ToolConfig()
        preflight["ran"] = True
        preflight["interactvlm_root"] = str(cfg.interactvlm_root)
        preflight["contact_type"] = contact_type
        try:
            require_tools(cfg, "interactvlm")
            check_interactvlm_assets(cfg.interactvlm_root, contact_type)
            preflight["ok"] = True
        except Exception as ex:
            preflight["ok"] = False
            preflight["error"] = str(ex)
            errors.append(
                f"{section}: InteractVLM asset preflight failed: {ex}"
            )
    hard["interactvlm_preflight"] = preflight
    if preflight["ran"] and preflight["ok"] is False:
        hard["structural_ok"] = False
        return

    path = rhino_dir / "contact_vertices.npy"
    if not path.is_file():
        errors.append(f"{section}: missing artifact {path}")
        hard["structural_ok"] = False
        return

    arr = np.load(path)
    hard["path"] = str(path)
    hard["shape"] = list(arr.shape)
    hard["dtype"] = str(arr.dtype)

    structural_ok = True
    if arr.dtype != np.bool_:
        errors.append(f"{section}: dtype {arr.dtype} != bool")
        structural_ok = False

    image_dir = rhino_dir / "image"
    if image_dir.is_dir():
        num_images = len(sorted(image_dir.glob("*.png")))
        hard["processed_image_png_count"] = num_images
        if arr.ndim == 0:
            pass  # shape error already caught below
        elif arr.shape[0] != num_images:
            errors.append(
                f"{section}: frame count {arr.shape[0]} != processed/image/*.png count {num_images}"
            )
            structural_ok = False
    else:
        errors.append(f"{section}: missing {image_dir}; cannot verify frame-count alignment")
        structural_ok = False

    if arr.shape != (expected_frames, num_smplx_vertices):
        errors.append(
            f"{section}: shape {tuple(arr.shape)} != ({expected_frames}, {num_smplx_vertices})"
        )
        structural_ok = False

    labels_path = rhino_dir / "contact_labels.npy"
    if not labels_path.is_file():
        errors.append(f"{section}: missing contact_labels.npy required for contact-frame gating")
        hard["structural_ok"] = False
        return
    labels = np.load(labels_path).astype(bool)
    if structural_ok and labels.shape != (arr.shape[0],):
        errors.append(
            f"{section}: contact_labels shape {tuple(labels.shape)} != ({arr.shape[0]},)"
        )
        structural_ok = False

    if structural_ok:
        per_frame = arr.sum(axis=1).astype(np.int64)
        nonzero_frames = int((per_frame > 0).sum())
        total_positive = int(arr.sum())
        contact_frames = int(labels.sum())
        hard["total_positive"] = total_positive
        hard["nonzero_frames"] = nonzero_frames
        hard["contact_frames"] = contact_frames

        # Step 12 runs contact-frames-only: predictions outside the labels are stale
        outside = int(arr[~labels].sum())
        hard["positives_outside_contact_frames"] = outside
        if outside != 0:
            errors.append(
                f"{section}: {outside} positive vertices on non-contact frames "
                f"(step 12 must only predict on contact frames)"
            )
            structural_ok = False

        adv["total_positive"] = total_positive
        adv["nonzero_frames"] = nonzero_frames
        nonzero_frac = nonzero_frames / max(contact_frames, 1)
        adv["nonzero_contact_frames_frac"] = nonzero_frac
        if nonzero_frac < _ADVISORY_NONZERO_FRAMES_MIN_FRAC:
            advisories.append(
                f"{section}: {nonzero_frames}/{contact_frames} contact frames nonzero "
                f"({nonzero_frac * 100:.1f}%) below advisory floor "
                f"{_ADVISORY_NONZERO_FRAMES_MIN_FRAC * 100:.0f}%"
            )

        if reference_dir is not None:
            ref_path = reference_dir / "contact_vertices.npy"
            if ref_path.is_file():
                try:
                    ref_arr = np.load(ref_path)
                    if ref_arr.shape == arr.shape:
                        # Reference may cover all frames (the reference run used InteractVLM on
                        # every frame); compare on this run's contact frames only.
                        ref_total = int(ref_arr[labels].sum())
                        adv["ref_total_positive_contact_frames"] = ref_total
                        adv["total_positive_delta_pct"] = _pct_delta(total_positive, ref_total)
                        if adv["total_positive_delta_pct"] > _ADVISORY_COUNT_TOL_STEP12:
                            advisories.append(
                                f"{section}: total positive vertices {total_positive} differs from "
                                f"reference (contact frames) {ref_total} by "
                                f"{adv['total_positive_delta_pct'] * 100:.1f}% "
                                f"(tol ±{_ADVISORY_COUNT_TOL_STEP12 * 100:.0f}%)"
                            )
                        ref_per_frame = ref_arr.sum(axis=1).astype(np.int64)
                        pearson = _pearson(per_frame[labels], ref_per_frame[labels])
                        adv["per_frame_pearson_contact_frames"] = pearson
                        if pearson is None:
                            advisories.append(
                                f"{section}: per-frame Pearson undefined (zero variance on one side)"
                            )
                        elif pearson < _ADVISORY_PEARSON_MIN:
                            advisories.append(
                                f"{section}: per-frame Pearson {pearson:.3f} (contact frames) below "
                                f"advisory floor {_ADVISORY_PEARSON_MIN:.2f}"
                            )
                        adv["iou_vs_reference_contact_frames"] = _iou(arr[labels], ref_arr[labels])
                        if adv["iou_vs_reference_contact_frames"] < _ADVISORY_IOU_MIN:
                            advisories.append(
                                f"{section}: IoU vs reference (contact frames) "
                                f"{adv['iou_vs_reference_contact_frames']:.3f} below advisory floor "
                                f"{_ADVISORY_IOU_MIN:.2f}"
                            )
                    else:
                        advisories.append(
                            f"{section}: reference shape {ref_arr.shape} != RHINO shape {arr.shape}"
                        )
                except Exception as ex:
                    advisories.append(f"{section}: failed to load reference {ref_path}: {ex}")

    hard["structural_ok"] = structural_ok


def validate_step13(rhino_dir: Path, reference_dir: Optional[Path],
                    results: dict, errors: list, advisories: list,
                    expected_frames: int, num_smplx_vertices: int) -> None:
    section = "step13_contact_vertices_refined"
    results[section] = {"hard": {}, "advisory": {}}
    hard = results[section]["hard"]
    adv = results[section]["advisory"]

    path = rhino_dir / "contact_vertices_refined.npy"
    labels_path = rhino_dir / "contact_labels.npy"

    if not path.is_file():
        errors.append(f"{section}: missing artifact {path}")
        hard["structural_ok"] = False
        return
    if not labels_path.is_file():
        errors.append(f"{section}: missing contact_labels.npy required for gating invariant")
        hard["structural_ok"] = False
        return

    arr = np.load(path)
    labels = np.load(labels_path)
    hard["path"] = str(path)
    hard["shape"] = list(arr.shape)
    hard["dtype"] = str(arr.dtype)

    structural_ok = True
    if arr.dtype != np.bool_:
        errors.append(f"{section}: dtype {arr.dtype} != bool")
        structural_ok = False
    if arr.shape != (expected_frames, num_smplx_vertices):
        errors.append(
            f"{section}: shape {tuple(arr.shape)} != ({expected_frames}, {num_smplx_vertices})"
        )
        structural_ok = False
    if labels.shape != (expected_frames,):
        errors.append(
            f"{section}: contact_labels shape {tuple(labels.shape)} != ({expected_frames},)"
        )
        structural_ok = False

    try:
        hand_ids, non_hand_ids = _load_hand_ids()
    except FileNotFoundError:
        errors.append(
            f"{section}: MANO hand ids pkl not found at {_HAND_IDS_PATH}; cannot verify hand-only mask"
        )
        hard["structural_ok"] = False
        return

    hard["num_hand_vertices"] = int(len(hand_ids))

    if structural_ok:
        non_contact_mask = ~labels.astype(bool)
        non_contact_positives = int(arr[non_contact_mask].sum())
        non_hand_positives = int(arr[:, non_hand_ids].sum())
        hard["non_contact_frame_positives"] = non_contact_positives
        hard["non_hand_vertex_positives"] = non_hand_positives

        if non_contact_positives != 0:
            errors.append(
                f"{section}: {non_contact_positives} positive vertices on non-contact frames "
                f"(expected 0 — violates refined gating invariant)"
            )
            structural_ok = False
        if non_hand_positives != 0:
            errors.append(
                f"{section}: {non_hand_positives} positive vertices outside MANO hand mask "
                f"(expected 0 — refined tensor must be hand-only)"
            )
            structural_ok = False

        total_positive = int(arr.sum())
        nonzero_frames = int((arr.any(axis=1)).sum())
        adv["total_positive"] = total_positive
        adv["nonzero_frames"] = nonzero_frames
        adv["contact_frames"] = int(labels.sum())

        if reference_dir is not None:
            ref_path = reference_dir / "contact_vertices_refined.npy"
            if ref_path.is_file():
                try:
                    ref_arr = np.load(ref_path)
                    if ref_arr.shape == arr.shape:
                        ref_total = int(ref_arr.sum())
                        adv["ref_total_positive"] = ref_total
                        adv["total_positive_delta_pct"] = _pct_delta(total_positive, ref_total)
                        if adv["total_positive_delta_pct"] > _ADVISORY_COUNT_TOL_STEP13:
                            advisories.append(
                                f"{section}: total refined positives {total_positive} differs from "
                                f"reference {ref_total} by "
                                f"{adv['total_positive_delta_pct'] * 100:.1f}% "
                                f"(tol ±{_ADVISORY_COUNT_TOL_STEP13 * 100:.0f}%)"
                            )
                        iou = _iou(arr, ref_arr)
                        adv["iou_vs_reference"] = iou
                        if iou < _ADVISORY_IOU_MIN:
                            advisories.append(
                                f"{section}: IoU vs reference refined tensor {iou:.3f} below "
                                f"advisory floor {_ADVISORY_IOU_MIN:.2f}"
                            )
                    else:
                        advisories.append(
                            f"{section}: reference shape {ref_arr.shape} != RHINO shape {arr.shape}"
                        )
                except Exception as ex:
                    advisories.append(f"{section}: failed to load reference {ref_path}: {ex}")

    hard["structural_ok"] = structural_ok


_STEP_VALIDATORS = {
    11: validate_step11,
    12: validate_step12,
    13: validate_step13,
}


def main(args: ValidateContactArgs) -> None:
    rhino_dir = args.rhino_dir.resolve()
    if not rhino_dir.is_dir():
        print(f"ERROR: RHINO directory not found: {rhino_dir}", file=sys.stderr)
        sys.exit(1)

    reference_dir = args.reference_dir.resolve() if args.reference_dir else None
    if reference_dir is not None and not reference_dir.is_dir():
        print(
            f"WARNING: reference directory not found: {reference_dir}; "
            "advisory cross-references will be skipped.",
            file=sys.stderr,
        )
        reference_dir = None

    if args.report_path is not None:
        report_path = args.report_path.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        report_path = rhino_dir.parent / "validate_contact_report.json"

    try:
        requested = sorted({int(s) for s in args.steps.split(",") if s.strip()})
    except ValueError as ex:
        print(f"ERROR: invalid --steps value {args.steps!r}: {ex}", file=sys.stderr)
        sys.exit(1)

    unknown = [s for s in requested if s not in _STEP_VALIDATORS]
    if unknown:
        print(f"ERROR: unsupported step(s) {unknown}; valid: {sorted(_STEP_VALIDATORS)}", file=sys.stderr)
        sys.exit(1)

    print("=" * 72)
    print("Validating contact preprocessing steps 11-13")
    print(f"  RHINO dir:     {rhino_dir}")
    print(f"  Reference dir: {reference_dir if reference_dir else '(none — advisory xref skipped)'}")
    print(f"  Report path:   {report_path}")
    print(f"  Steps:         {requested}")
    print("=" * 72)

    results: dict = {}
    errors: list[str] = []
    advisories: list[str] = []

    expected_frames = args.expected_frames
    if expected_frames is None:
        image_dir = rhino_dir / "image"
        expected_frames = len(sorted(image_dir.glob("*.png"))) if image_dir.is_dir() else 0
        if expected_frames == 0:
            print(f"ERROR: cannot infer --expected-frames from {image_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"  Expected frames (from processed/image): {expected_frames}")

    for step in requested:
        print(f"\n--- Step {step} ---")
        validator = _STEP_VALIDATORS[step]
        if step == 11:
            validator(rhino_dir, reference_dir, results, errors, advisories,
                      expected_frames=expected_frames,
                      max_spans=args.max_spans,
                      gt_labels=args.gt_labels, gt_offset=args.gt_offset)
        elif step == 12:
            validator(rhino_dir, reference_dir, results, errors, advisories,
                      expected_frames=expected_frames,
                      num_smplx_vertices=args.num_smplx_vertices,
                      tools=args.tools,
                      contact_type=args.contact_type,
                      skip_interactvlm_preflight=args.skip_interactvlm_preflight)
        else:
            validator(rhino_dir, reference_dir, results, errors, advisories,
                      expected_frames=expected_frames,
                      num_smplx_vertices=args.num_smplx_vertices)
        section_key = next((k for k in results if k.startswith(f"step{step}")), None)
        if section_key is None:
            continue
        section = results[section_key]
        hard = section.get("hard", {})
        adv = section.get("advisory", {})
        print(f"  structural_ok: {hard.get('structural_ok')}")
        for k, v in hard.items():
            if k == "structural_ok":
                continue
            print(f"    hard.{k}: {v}")
        for k, v in adv.items():
            print(f"    adv.{k}: {v}")

    print("\n" + "=" * 72)
    if errors:
        print(f"HARD ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
    else:
        print("All hard invariants PASSED")
    if advisories:
        print(f"\nADVISORIES ({len(advisories)}):")
        for a in advisories:
            print(f"  ~ {a}")
    print("=" * 72)

    payload = {
        "rhino_dir": str(rhino_dir),
        "reference_dir": str(reference_dir) if reference_dir else None,
        "steps": requested,
        "pass": not errors,
        "results": results,
        "errors": errors,
        "advisories": advisories,
    }

    with open(report_path, "w") as f:
        json.dump(_sanitize(payload), f, indent=2)
    print(f"Report saved to {report_path}")

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    import tyro
    main(tyro.cli(ValidateContactArgs))
