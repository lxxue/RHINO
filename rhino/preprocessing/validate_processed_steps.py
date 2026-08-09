"""Validate preprocessing steps 4-10 against a reference directory.

Compares monocular cues, processed base, SMPL-X estimation/refinement, keypoints,
scene alignment, and object pose alignment between a RHINO output directory and a
reference run's directory.

All numeric thresholds are advisory: metrics are reported and deviations flagged,
but threshold violations alone do not constitute hard failures. Hard errors only
arise from missing/malformed structural artifacts.
"""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def expected_pose_filenames(n: int, ext: str = ".npy", width: int = 4) -> set[str]:
    return {f"{i:0{width}d}{ext}" for i in range(n)}


def geodesic_rotation_errors_mat(rots_a: np.ndarray, rots_b: np.ndarray) -> np.ndarray:
    """Per-element geodesic rotation error in degrees between two arrays of 3x3 rotations."""
    errs = []
    for ra, rb in zip(rots_a, rots_b):
        d = ra @ rb.T
        cos_t = (np.trace(d) - 1.0) / 2.0
        cos_t = float(np.clip(cos_t, -1.0, 1.0))
        errs.append(np.degrees(np.arccos(cos_t)))
    return np.asarray(errs)


def axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """Convert (..., 3) axis-angle vectors to (..., 3, 3) rotation matrices via Rodrigues."""
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    safe = np.where(theta > 1e-12, theta, 1.0)
    k = aa / safe
    K = np.zeros(aa.shape[:-1] + (3, 3), dtype=aa.dtype)
    K[..., 0, 1] = -k[..., 2]
    K[..., 0, 2] = k[..., 1]
    K[..., 1, 0] = k[..., 2]
    K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1]
    K[..., 2, 1] = k[..., 0]
    eye = np.eye(3, dtype=aa.dtype)
    sin_t = np.sin(theta)[..., None]
    cos_t = np.cos(theta)[..., None]
    R = eye + sin_t * K + (1 - cos_t) * (K @ K)
    is_zero = (theta < 1e-12)[..., None]
    R = np.where(is_zero, eye, R)
    return R


def sim3_align(src: np.ndarray, dst: np.ndarray):
    """Procrustes alignment with scale (Sim3). Returns aligned_src, scale, R, t, rmse."""
    src_c = src - src.mean(axis=0)
    dst_c = dst - dst.mean(axis=0)
    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    scale = np.trace(np.diag(S) @ D) / np.trace(src_c.T @ src_c)
    t = dst.mean(axis=0) - scale * (src.mean(axis=0) @ R.T)
    aligned = scale * (src @ R.T) + t
    rmse = float(np.sqrt(np.mean(np.sum((aligned - dst) ** 2, axis=1))))
    return aligned, float(scale), R, t, rmse


def load_pose_file(path: Path, expected_count: int) -> np.ndarray:
    arr = np.loadtxt(path, dtype=np.float64).flatten()
    if arr.size != expected_count:
        raise ValueError(f"{path}: expected {expected_count} values, got {arr.size}")
    return arr


# ---------------------------------------------------------------------------
# Step 4 — extract monocular cues
# ---------------------------------------------------------------------------


def _derive_image_hw(rhino_dir: Path) -> Optional[tuple]:
    """Derive (H, W) from the first available source image, or None if unavailable.

    Metric3D depth/normal maps inherit the input image resolution; hard-coding
    (960, 720) is brittle because sequences with other aspect ratios (portrait
    vs landscape, 480p sources, etc.) would spuriously fail the shape check.
    """
    candidates = [
        rhino_dir / "images" / "all_frames" / "0000.jpg",
        rhino_dir / "processed" / "image" / "0000.png",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            from PIL import Image  # local import to avoid requiring PIL at module load
            with Image.open(p) as im:
                w, h = im.size  # PIL returns (W, H)
                return (h, w)
        except Exception:  # noqa: BLE001
            continue
    # Last resort: inspect the first depth npy itself (internal consistency only).
    depth0 = rhino_dir / "monocular_geometry" / "depth" / "0000.npy"
    if depth0.exists():
        try:
            arr = np.load(depth0)
            if arr.ndim == 2:
                return tuple(arr.shape)
        except Exception:  # noqa: BLE001
            pass
    return None


def validate_step4(rhino_dir: Path, ref_dir: Path, results: dict, errors: list,
                   advisories: list, expected_frames: int = 579) -> None:
    section = "step4_monocular"
    results[section] = {}
    rhino_geo = rhino_dir / "monocular_geometry"
    ref_geo = ref_dir / "monocular_geometry"

    # Derive expected (H, W) from the first available image so we are not tied
    # to a single (960, 720) resolution.
    image_hw = _derive_image_hw(rhino_dir)
    if image_hw is None:
        advisories.append(
            f"{section}: could not derive image (H, W); falling back to (960, 720)."
        )
        image_hw = (960, 720)
    results[section]["image_hw"] = list(image_hw)

    # Structural and shape/range checks
    for kind in ("depth", "normal"):
        rhino_kind = rhino_geo / kind
        ref_kind = ref_geo / kind
        sub = results[section].setdefault(kind, {})

        if not rhino_kind.exists():
            errors.append(f"{section}/{kind}: missing directory {rhino_kind}")
            continue

        npy_files = sorted(rhino_kind.glob("*.npy"))
        jpg_files = sorted(rhino_kind.glob("*.jpg"))
        sub["npy_count"] = len(npy_files)
        sub["jpg_count"] = len(jpg_files)

        # Filename pattern: %04d.npy
        expected_npy = expected_pose_filenames(expected_frames, ".npy")
        actual_npy = {p.name for p in npy_files}
        missing = sorted(expected_npy - actual_npy)
        extra = sorted(actual_npy - expected_npy)
        sub["npy_missing"] = missing[:10]
        sub["npy_extra"] = extra[:10]
        if missing:
            errors.append(f"{section}/{kind}: {len(missing)} missing .npy files (e.g., {missing[:3]})")
        if extra:
            errors.append(f"{section}/{kind}: {len(extra)} unexpected .npy filenames (e.g., {extra[:3]})")

        if len(jpg_files) != len(npy_files):
            errors.append(f"{section}/{kind}: jpg count {len(jpg_files)} != npy count {len(npy_files)}")

        # EXHAUSTIVE shape and range check over ALL expected frames.
        # Depth is (H, W); normals are (H, W, 3).
        bad_shape = []
        bad_range = []
        bad_nonfinite = []
        checked_count = 0
        expected_shape = image_hw if kind == "depth" else (*image_hw, 3)
        for idx in range(expected_frames):
            npy_path = rhino_kind / f"{idx:04d}.npy"
            if not npy_path.exists():
                continue
            try:
                arr = np.load(npy_path)
            except Exception as ex:  # noqa: BLE001
                errors.append(f"{section}/{kind}: failed to load {npy_path}: {ex}")
                continue
            checked_count += 1
            if arr.shape != expected_shape:
                bad_shape.append((idx, list(arr.shape)))
            # Finiteness check MUST come before min/max comparison — NaN/inf values
            # silently pass any threshold comparison (NaN < x and NaN > x both False),
            # so a corrupt all-NaN array would otherwise be reported as valid.
            if not np.all(np.isfinite(arr)):
                n_nan = int(np.sum(np.isnan(arr)))
                n_inf = int(np.sum(np.isinf(arr)))
                bad_nonfinite.append((idx, n_nan, n_inf))
                continue
            mn, mx = float(arr.min()), float(arr.max())
            if mn < -1e-6 or mx > 1.0 + 1e-6:
                bad_range.append((idx, mn, mx))
        sub["shape_range_checked_frames"] = checked_count
        sub["bad_shape_count"] = len(bad_shape)
        sub["bad_shape_samples"] = bad_shape[:5]
        sub["bad_range_count"] = len(bad_range)
        sub["bad_range_samples"] = bad_range[:5]
        sub["bad_nonfinite_count"] = len(bad_nonfinite)
        sub["bad_nonfinite_samples"] = bad_nonfinite[:5]
        if bad_shape:
            errors.append(
                f"{section}/{kind}: {len(bad_shape)} frames with wrong shape (first: {bad_shape[:3]})"
            )
        if bad_nonfinite:
            errors.append(
                f"{section}/{kind}: {len(bad_nonfinite)} frames with NaN/inf values "
                f"(first: {bad_nonfinite[:3]})"
            )
        if bad_range:
            errors.append(
                f"{section}/{kind}: {len(bad_range)} frames with values outside [0,1] (first: {bad_range[:3]})"
            )

    if errors and any(section in e for e in errors):
        # Skip numerical comparison if structural checks already failed
        return

    # Numerical similarity (advisory)
    rhino_depth = rhino_geo / "depth"
    ref_depth = ref_geo / "depth"
    rhino_normal = rhino_geo / "normal"
    ref_normal = ref_geo / "normal"

    if not (ref_depth.exists() and ref_normal.exists()):
        results[section]["reference_present"] = False
        return
    results[section]["reference_present"] = True

    depth_maes = []
    normal_angle_means = []
    for idx in range(expected_frames):
        fname = f"{idx:04d}.npy"
        rd = rhino_depth / fname
        rfd = ref_depth / fname
        rn = rhino_normal / fname
        rfn = ref_normal / fname
        if not (rd.exists() and rfd.exists() and rn.exists() and rfn.exists()):
            continue
        d_rh = np.load(rd).astype(np.float64)
        d_rf = np.load(rfd).astype(np.float64)
        if d_rh.shape != d_rf.shape:
            continue
        depth_maes.append(float(np.mean(np.abs(d_rh - d_rf))))

        n_rh = np.load(rn).astype(np.float64) * 2.0 - 1.0  # decode from [0,1]
        n_rf = np.load(rfn).astype(np.float64) * 2.0 - 1.0
        if n_rh.shape != n_rf.shape:
            continue
        # Renormalize to unit vectors (per pixel)
        n_rh /= np.maximum(np.linalg.norm(n_rh, axis=-1, keepdims=True), 1e-9)
        n_rf /= np.maximum(np.linalg.norm(n_rf, axis=-1, keepdims=True), 1e-9)
        cos = np.clip(np.sum(n_rh * n_rf, axis=-1), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos))
        normal_angle_means.append(float(np.mean(angle)))

    if depth_maes:
        depth_arr = np.asarray(depth_maes)
        results[section]["depth_mae_mean"] = float(depth_arr.mean())
        results[section]["depth_mae_max"] = float(depth_arr.max())
        results[section]["depth_mae_p95"] = float(np.percentile(depth_arr, 95))
        results[section]["depth_mae_compared_frames"] = len(depth_maes)
        if depth_arr.mean() >= 0.01:
            advisories.append(
                f"{section}: mean depth MAE {depth_arr.mean():.6f} >= 0.01 advisory threshold"
            )
    if normal_angle_means:
        ang_arr = np.asarray(normal_angle_means)
        results[section]["normal_angle_deg_mean"] = float(ang_arr.mean())
        results[section]["normal_angle_deg_max"] = float(ang_arr.max())
        results[section]["normal_angle_deg_p95"] = float(np.percentile(ang_arr, 95))
        results[section]["normal_angle_compared_frames"] = len(normal_angle_means)
        if ang_arr.mean() >= 2.0:
            advisories.append(
                f"{section}: mean normal angular error {ang_arr.mean():.4f}° >= 2° advisory threshold"
            )


# ---------------------------------------------------------------------------
# Step 5 — prepare base processed directory
# ---------------------------------------------------------------------------


def validate_step5(rhino_dir: Path, ref_dir: Path, results: dict, errors: list,
                   advisories: list, expected_frames: int = 579) -> None:
    section = "step5_processed_base"
    results[section] = {}
    rhino_proc = rhino_dir / "processed"
    ref_proc = ref_dir / "processed"

    # Directory structure
    image_dirs = ["image", "human_mask", "obj_mask", "hand_mask", "all_mask"]
    npy_dirs = ["depth", "normal"]
    structure = {}
    for sub in image_dirs + npy_dirs:
        rhino_sub = rhino_proc / sub
        ext = ".png" if sub in image_dirs else ".npy"
        if not rhino_sub.exists():
            errors.append(f"{section}: missing subdirectory {rhino_sub}")
            structure[sub] = {"exists": False}
            continue
        files = sorted(rhino_sub.glob(f"*{ext}"))
        expected = expected_pose_filenames(expected_frames, ext)
        actual = {p.name for p in files}
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        structure[sub] = {
            "count": len(files),
            "missing": missing[:5],
            "extra": extra[:5],
        }
        if len(files) != expected_frames:
            errors.append(f"{section}/{sub}: file count {len(files)} != {expected_frames}")
        if missing:
            errors.append(f"{section}/{sub}: {len(missing)} missing files")
        if extra:
            errors.append(f"{section}/{sub}: {len(extra)} unexpected filenames")
    results[section]["subdirs"] = structure

    # intrinsic.npy + c2ws.npy existence
    rhino_intr = rhino_proc / "intrinsic.npy"
    rhino_c2ws = rhino_proc / "c2ws.npy"
    if not rhino_intr.exists():
        errors.append(f"{section}: missing {rhino_intr}")
    if not rhino_c2ws.exists():
        errors.append(f"{section}: missing {rhino_c2ws}")

    # Intrinsic 3x3 matches reference 4x4 top-left 3x3
    ref_intr_path = ref_proc / "intrinsic.npy"
    if rhino_intr.exists() and ref_intr_path.exists():
        rh_intr = np.load(rhino_intr)
        rf_intr_full = np.load(ref_intr_path)
        rf_intr = rf_intr_full[:3, :3]
        results[section]["intrinsic_rhino_shape"] = list(rh_intr.shape)
        results[section]["intrinsic_ref_shape"] = list(rf_intr_full.shape)
        # Validate shape BEFORE the diff computation — a 4x4 regression in processed/intrinsic.npy
        # must be reported as a structural error, not abort the validator with a numpy
        # broadcasting ValueError.
        atol = 1e-4
        if rh_intr.shape != (3, 3):
            errors.append(f"{section}: intrinsic.npy shape {rh_intr.shape} != (3,3)")
        else:
            results[section]["intrinsic_max_abs_diff"] = float(np.max(np.abs(rh_intr - rf_intr)))
            if not np.allclose(rh_intr, rf_intr, atol=atol):
                advisories.append(
                    f"{section}: intrinsic differs from reference top-left 3x3 (max abs diff "
                    f"{results[section]['intrinsic_max_abs_diff']:.6e} > atol={atol})"
                )

    # c2ws direct comparison (stage uses AprilTag-aligned absolute frame)
    ref_c2ws_path = ref_proc / "c2ws.npy"
    if rhino_c2ws.exists() and ref_c2ws_path.exists():
        rh_c2ws = np.load(rhino_c2ws)
        rf_c2ws = np.load(ref_c2ws_path)
        results[section]["c2ws_rhino_shape"] = list(rh_c2ws.shape)
        results[section]["c2ws_ref_shape"] = list(rf_c2ws.shape)
        if rh_c2ws.shape != (expected_frames, 4, 4):
            errors.append(f"{section}: c2ws.npy shape {rh_c2ws.shape} != ({expected_frames},4,4)")
        elif rh_c2ws.shape == rf_c2ws.shape:
            rh_R = rh_c2ws[:, :3, :3].astype(np.float64)
            rf_R = rf_c2ws[:, :3, :3].astype(np.float64)
            rot_errs = geodesic_rotation_errors_mat(rh_R, rf_R)
            trans_errs = np.linalg.norm(rh_c2ws[:, :3, 3] - rf_c2ws[:, :3, 3], axis=1)
            results[section]["c2ws_rot_err_mean_deg"] = float(rot_errs.mean())
            results[section]["c2ws_rot_err_max_deg"] = float(rot_errs.max())
            results[section]["c2ws_rot_err_p95_deg"] = float(np.percentile(rot_errs, 95))
            results[section]["c2ws_trans_err_mean"] = float(trans_errs.mean())
            results[section]["c2ws_trans_err_max"] = float(trans_errs.max())
            results[section]["c2ws_trans_err_p95"] = float(np.percentile(trans_errs, 95))
            if rot_errs.mean() >= 0.1:
                advisories.append(
                    f"{section}: mean c2ws rotation error {rot_errs.mean():.4f}° >= 0.1° advisory threshold"
                )
            if trans_errs.mean() >= 1e-4:
                advisories.append(
                    f"{section}: mean c2ws translation error {trans_errs.mean():.6e} >= 1e-4 advisory threshold"
                )
        else:
            errors.append(
                f"{section}: c2ws shape mismatch RHINO {rh_c2ws.shape} vs ref {rf_c2ws.shape}"
            )


# ---------------------------------------------------------------------------
# Step 6 — AiOS SMPL-X estimation (structural-only)
# ---------------------------------------------------------------------------


SMPLX_PARAM_COUNT = 172  # 3 + 63 + 45 + 45 + 3 + 10 + 3
SMPLX_BETAS_FULL = 300


def validate_step6(rhino_dir: Path, ref_dir: Path, results: dict, errors: list,
                   advisories: list, expected_frames: int = 579,
                   ) -> None:
    section = "step6_smplx_aios"
    results[section] = {}
    rhino_backend = rhino_dir / "smplx" / "aios"
    return _validate_step6_smplx_txt(
        rhino_backend / "smplx_params", rhino_backend / "mean_shape.txt",
        section, results, errors, expected_frames, layout="aios",
    )


def _validate_step6_smplx_txt(params_dir: Path, mean_shape_path: Path, section: str,
                               results: dict, errors: list, expected_frames: int,
                               layout: str) -> None:
    """Validate the AiOS SMPL-X .txt layout."""
    results[section]["layout"] = layout
    if not params_dir.exists():
        errors.append(f"{section}: missing {params_dir}")
        return

    txt_files = sorted(params_dir.glob("*.txt"))
    results[section]["params_count"] = len(txt_files)
    expected = expected_pose_filenames(expected_frames, ".txt")
    actual = {p.name for p in txt_files}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    results[section]["missing_files"] = missing[:10]
    results[section]["missing_count"] = len(missing)
    results[section]["extra_files"] = extra[:10]
    if missing:
        errors.append(f"{section}: {len(missing)} missing pose .txt files (e.g., {missing[:3]})")
    if extra:
        errors.append(f"{section}: {len(extra)} unexpected filenames (e.g., {extra[:3]})")

    # EXHAUSTIVE 172-param count check over ALL expected frames
    bad_count = []
    checked_count = 0
    for idx in range(expected_frames):
        p = params_dir / f"{idx:04d}.txt"
        if not p.exists():
            continue
        try:
            arr = np.loadtxt(p, dtype=np.float64).flatten()
        except Exception as ex:  # noqa: BLE001
            errors.append(f"{section}: failed to load {p}: {ex}")
            continue
        checked_count += 1
        if arr.size != SMPLX_PARAM_COUNT:
            bad_count.append((idx, int(arr.size)))
    results[section]["param_count_checked_frames"] = checked_count
    results[section]["bad_param_count_total"] = len(bad_count)
    results[section]["bad_param_count"] = bad_count[:10]
    if bad_count:
        errors.append(
            f"{section}: {len(bad_count)} pose files with wrong param count "
            f"(expected {SMPLX_PARAM_COUNT}; first: {bad_count[:3]})"
        )

    # mean_shape.txt existence + 300-D length
    if not mean_shape_path.exists():
        errors.append(f"{section}: missing {mean_shape_path}")
    else:
        try:
            mean_shape = np.loadtxt(mean_shape_path, dtype=np.float64).flatten()
            results[section]["mean_shape_length"] = int(mean_shape.size)
            if mean_shape.size != SMPLX_BETAS_FULL:
                errors.append(
                    f"{section}: mean_shape length {mean_shape.size} != {SMPLX_BETAS_FULL}"
                )
        except Exception as ex:  # noqa: BLE001
            errors.append(f"{section}: failed to load mean_shape.txt: {ex}")


# ---------------------------------------------------------------------------
# Step 7 — Sapiens keypoint estimation + conversion
# ---------------------------------------------------------------------------


def validate_step7(rhino_dir: Path, ref_dir: Path, results: dict, errors: list,
                   advisories: list, expected_frames: int = 579) -> None:
    section = "step7_keypoints"
    results[section] = {}
    sapiens_dir = rhino_dir / "keypoints" / "sapiens"
    smplx_kp_dir = rhino_dir / "keypoints" / "sapiens_smplx"

    # Raw Sapiens JSON outputs
    if not sapiens_dir.exists():
        errors.append(f"{section}: missing {sapiens_dir}")
    else:
        json_files = sorted(sapiens_dir.glob("*.json"))
        results[section]["sapiens_json_count"] = len(json_files)
        if len(json_files) != expected_frames:
            errors.append(f"{section}: sapiens JSON count {len(json_files)} != {expected_frames}")

    # Converted sapiens_smplx npys + mask
    if not smplx_kp_dir.exists():
        errors.append(f"{section}: missing {smplx_kp_dir}")
        return

    npy_files = sorted(p for p in smplx_kp_dir.glob("*.npy") if p.name != "mask.npy")
    results[section]["smplx_kp_count"] = len(npy_files)
    if len(npy_files) != expected_frames:
        errors.append(f"{section}: sapiens_smplx npy count {len(npy_files)} != {expected_frames}")

    mask_path = smplx_kp_dir / "mask.npy"
    if not mask_path.exists():
        errors.append(f"{section}: missing {mask_path}")
    else:
        mask = np.load(mask_path)
        results[section]["mask_shape"] = list(mask.shape)
        if mask.shape != (144,):
            errors.append(f"{section}: mask shape {mask.shape} != (144,)")

    # EXHAUSTIVE (144,3) shape check over ALL expected frames
    bad_shape = []
    checked_count = 0
    for idx in range(expected_frames):
        p = smplx_kp_dir / f"{idx:04d}.npy"
        if not p.exists():
            continue
        try:
            arr = np.load(p)
        except Exception as ex:  # noqa: BLE001
            errors.append(f"{section}: failed to load {p}: {ex}")
            continue
        checked_count += 1
        if arr.shape != (144, 3):
            bad_shape.append((idx, list(arr.shape)))
    results[section]["shape_checked_frames"] = checked_count
    results[section]["bad_shape_count"] = len(bad_shape)
    results[section]["bad_shape_samples"] = bad_shape[:10]
    if bad_shape:
        errors.append(
            f"{section}: {len(bad_shape)} sapiens_smplx files with wrong shape "
            f"(expected (144,3); first: {bad_shape[:3]})"
        )


# ---------------------------------------------------------------------------
# Step 8 — refined SMPL-X (aios_refined)
# ---------------------------------------------------------------------------


def _load_smplx_param_file(path: Path) -> dict:
    """Decompose a 172-float SMPL-X param file into named fields.

    Layout (matches refine_smplx.py):
      global_orient (3) | body_pose (63) | left_hand_pose (45) |
      right_hand_pose (45) | transl (3) | expression (10) | jaw_pose (3)
    Per-frame betas are NOT in this file — they live in mean_shape.txt.
    """
    arr = np.loadtxt(path, dtype=np.float64).flatten()
    if arr.size != SMPLX_PARAM_COUNT:
        raise ValueError(f"{path}: size {arr.size} != {SMPLX_PARAM_COUNT}")
    idx = 0
    out = {}
    for name, n in [
        ("global_orient", 3),
        ("body_pose", 63),
        ("left_hand_pose", 45),
        ("right_hand_pose", 45),
        ("transl", 3),
        ("expression", 10),
        ("jaw_pose", 3),
    ]:
        out[name] = arr[idx : idx + n]
        idx += n
    return out


def validate_step8(rhino_dir: Path, ref_dir: Path, results: dict, errors: list,
                   advisories: list, expected_frames: int = 579) -> None:
    section = "step8_smplx_refined"
    results[section] = {}
    rhino_ref_dir = rhino_dir / "smplx" / "aios_refined"
    params_dir = rhino_ref_dir / "smplx_params"
    mean_shape_path = rhino_ref_dir / "mean_shape.txt"

    if not params_dir.exists():
        errors.append(f"{section}: missing {params_dir}")
        return

    # Structure
    txt_files = sorted(params_dir.glob("*.txt"))
    results[section]["params_count"] = len(txt_files)
    expected = expected_pose_filenames(expected_frames, ".txt")
    actual = {p.name for p in txt_files}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    results[section]["missing_count"] = len(missing)
    if missing:
        errors.append(f"{section}: {len(missing)} missing refined pose files (e.g., {missing[:3]})")
    if extra:
        errors.append(f"{section}: {len(extra)} unexpected refined filenames")

    # EXHAUSTIVE 172-param count check over ALL expected refined frames
    bad_count = []
    checked_count = 0
    for idx in range(expected_frames):
        p = params_dir / f"{idx:04d}.txt"
        if not p.exists():
            continue
        try:
            arr = np.loadtxt(p, dtype=np.float64).flatten()
        except Exception as ex:  # noqa: BLE001
            errors.append(f"{section}: failed to load {p}: {ex}")
            continue
        checked_count += 1
        if arr.size != SMPLX_PARAM_COUNT:
            bad_count.append((idx, int(arr.size)))
    results[section]["param_count_checked_frames"] = checked_count
    results[section]["bad_param_count_total"] = len(bad_count)
    results[section]["bad_param_count"] = bad_count[:10]
    if bad_count:
        errors.append(
            f"{section}: {len(bad_count)} refined files with wrong param count "
            f"(expected {SMPLX_PARAM_COUNT}; first: {bad_count[:3]})"
        )

    if not mean_shape_path.exists():
        errors.append(f"{section}: missing {mean_shape_path}")
    else:
        try:
            mean_shape = np.loadtxt(mean_shape_path, dtype=np.float64).flatten()
            results[section]["mean_shape_length"] = int(mean_shape.size)
            if mean_shape.size != SMPLX_BETAS_FULL:
                errors.append(
                    f"{section}: mean_shape length {mean_shape.size} != {SMPLX_BETAS_FULL}"
                )
        except Exception as ex:  # noqa: BLE001
            errors.append(f"{section}: failed to load mean_shape.txt: {ex}")

    # Temporal smoothness of translation
    translations = []
    loaded_all = True
    for idx in range(expected_frames):
        p = params_dir / f"{idx:04d}.txt"
        if not p.exists():
            loaded_all = False
            break
        try:
            params = _load_smplx_param_file(p)
        except Exception:
            loaded_all = False
            break
        translations.append(params["transl"])
    if loaded_all:
        translations = np.asarray(translations)
        steps = np.linalg.norm(np.diff(translations, axis=0), axis=1)
        results[section]["transl_step_median"] = float(np.median(steps))
        results[section]["transl_step_max"] = float(np.max(steps))
        results[section]["transl_step_p95"] = float(np.percentile(steps, 95))
        large_jumps = int(np.sum(steps > 0.1))
        results[section]["transl_jumps_over_0.1"] = large_jumps
        if large_jumps > 0:
            advisories.append(
                f"{section}: {large_jumps} frame-to-frame translation jumps > 0.1 (max {steps.max():.4f})"
            )

    # Numerical comparison vs reference refined_aios.
    ref_params_dir = ref_dir / "smpl" / "refined_aios" / "smplx_params"
    ref_mean_shape = ref_dir / "smpl" / "refined_aios" / "mean_shape.txt"
    if not ref_params_dir.exists():
        advisories.append(f"{section}: reference {ref_params_dir} not present, skipping numerical comparison")
        return

    rot_errs = []
    trans_errs = []
    compared = 0
    for idx in range(expected_frames):
        rh_p = params_dir / f"{idx:04d}.txt"
        rf_p = ref_params_dir / f"pose-{idx:06d}.txt"
        if not (rh_p.exists() and rf_p.exists()):
            continue
        try:
            rh = _load_smplx_param_file(rh_p)
            rf = _load_smplx_param_file(rf_p)
        except Exception:
            continue
        compared += 1
        rh_body = np.concatenate([
            rh["global_orient"], rh["body_pose"], rh["left_hand_pose"],
            rh["right_hand_pose"], rh["jaw_pose"],
        ]).reshape(-1, 3)
        rf_body = np.concatenate([
            rf["global_orient"], rf["body_pose"], rf["left_hand_pose"],
            rf["right_hand_pose"], rf["jaw_pose"],
        ]).reshape(-1, 3)
        R_rh = axis_angle_to_matrix(rh_body)
        R_rf = axis_angle_to_matrix(rf_body)
        per_joint_err = geodesic_rotation_errors_mat(R_rh, R_rf)
        rot_errs.append(float(per_joint_err.mean()))
        trans_errs.append(float(np.linalg.norm(rh["transl"] - rf["transl"])))
    results[section]["ref_compared_frames"] = compared
    if compared:
        rot_arr = np.asarray(rot_errs)
        tr_arr = np.asarray(trans_errs)
        results[section]["ref_rot_err_mean_deg"] = float(rot_arr.mean())
        results[section]["ref_rot_err_max_deg"] = float(rot_arr.max())
        results[section]["ref_rot_err_p95_deg"] = float(np.percentile(rot_arr, 95))
        results[section]["ref_transl_err_mean"] = float(tr_arr.mean())
        results[section]["ref_transl_err_max"] = float(tr_arr.max())
        results[section]["ref_transl_err_p95"] = float(np.percentile(tr_arr, 95))
        if rot_arr.mean() >= 5.0:
            advisories.append(
                f"{section}: mean per-joint rotation error {rot_arr.mean():.4f}° >= 5° advisory"
            )
        if tr_arr.mean() >= 0.02:
            advisories.append(
                f"{section}: mean translation error {tr_arr.mean():.6f} >= 0.02 advisory"
            )

    # Beta L2 (mean_shape) comparison
    if mean_shape_path.exists() and ref_mean_shape.exists():
        try:
            rh_ms = np.loadtxt(mean_shape_path, dtype=np.float64).flatten()
            rf_ms = np.loadtxt(ref_mean_shape, dtype=np.float64).flatten()
            beta_l2 = float(np.linalg.norm(rh_ms - rf_ms))
            results[section]["mean_shape_l2_diff"] = beta_l2
            if beta_l2 >= 0.05:
                advisories.append(
                    f"{section}: mean_shape L2 diff {beta_l2:.4f} >= 0.05 advisory"
                )
        except Exception as ex:  # noqa: BLE001
            advisories.append(f"{section}: mean_shape numerical comparison failed: {ex}")


# ---------------------------------------------------------------------------
# Step 9 — align_smplx outputs
# ---------------------------------------------------------------------------


def validate_step9(rhino_dir: Path, ref_dir: Path, results: dict, errors: list,
                   advisories: list, expected_frames: int = 579,
                   camera_pose_dir: Optional[Path] = None,
                   normalize: bool = False) -> None:
    """Validate step 9 outputs.

    When ``normalize=False`` (default, stage sequences), step 9 rewrites
    processed/{c2ws.npy, intrinsic.npy} identically to the reconstruction source,
    so the byte-equality assertion is a hard check.

    When ``normalize=True`` (in-the-wild sequences), step 9 centers+rescales the
    c2ws translations, so c2ws byte-equality is EXPECTED to fail; the diff is
    reported but not promoted to an error. Intrinsic is still untouched by the
    normalize branch (only translations/scales are transformed) and remains a
    hard check.

    """
    section = "step9_aligned_smplx"
    results[section] = {}
    results[section]["normalize_mode"] = bool(normalize)
    rhino_proc = rhino_dir / "processed"
    ref_proc = ref_dir / "processed"
    aligned_path = rhino_proc / "aligned_smplx.pkl"
    results[section]["aligned_path"] = str(aligned_path)
    scene_pcd = rhino_proc / "scene_pcd.ply"

    # aligned_smplx.pkl structure
    required_keys = {
        "scale", "global_orient", "body_pose", "left_hand_pose", "right_hand_pose",
        "transl", "expression", "jaw_pose", "betas",
    }
    if not aligned_path.exists():
        errors.append(f"{section}: missing {aligned_path}")
        return
    try:
        with open(aligned_path, "rb") as f:
            aligned = pickle.load(f)
    except Exception as ex:  # noqa: BLE001
        errors.append(f"{section}: failed to load aligned_smplx.pkl: {ex}")
        return

    present = set(aligned.keys())
    missing_keys = sorted(required_keys - present)
    results[section]["keys"] = sorted(present)
    results[section]["missing_keys"] = missing_keys
    if missing_keys:
        errors.append(f"{section}: aligned_smplx.pkl missing keys {missing_keys}")

    # Shape checks
    def _shape(v):
        if isinstance(v, np.ndarray):
            return list(v.shape)
        if hasattr(v, "__len__"):
            return [len(v)]
        return None

    for k in sorted(present):
        results[section][f"shape_{k}"] = _shape(aligned[k])

    if "betas" in aligned:
        beta_arr = np.asarray(aligned["betas"]).flatten()
        if beta_arr.size != SMPLX_BETAS_FULL:
            errors.append(f"{section}: betas size {beta_arr.size} != {SMPLX_BETAS_FULL}")
    if "transl" in aligned:
        tr = np.asarray(aligned["transl"])
        if tr.shape != (expected_frames, 3):
            errors.append(f"{section}: transl shape {tr.shape} != ({expected_frames},3)")

    # scene_pcd.ply existence
    if not scene_pcd.exists():
        errors.append(f"{section}: missing {scene_pcd}")

    # Scale validity
    if "scale" in aligned:
        scale_arr = np.asarray(aligned["scale"]).flatten()
        results[section]["scale_count"] = int(scale_arr.size)
        results[section]["scale_mean"] = float(scale_arr.mean())
        results[section]["scale_std"] = float(scale_arr.std())
        results[section]["scale_min"] = float(scale_arr.min())
        results[section]["scale_max"] = float(scale_arr.max())
        if scale_arr.std() > 1e-6:
            errors.append(f"{section}: scale std {scale_arr.std():.3e} > 1e-6 (should be constant per frame)")
        if scale_arr.size > 0:
            s = float(scale_arr.mean())
            if not (0.90 <= s <= 1.10):
                errors.append(f"{section}: scale {s:.4f} outside [0.90, 1.10] — alignment likely failed")
            elif not (0.95 <= s <= 1.10):
                advisories.append(f"{section}: scale {s:.4f} outside advisory [0.95, 1.10] band")
            # Compare against the reference sequence's alignment scale
            ref_pkl_path = ref_proc / "aligned_smplx.pkl"
            if ref_pkl_path.exists():
                with open(ref_pkl_path, "rb") as f:
                    ref_scale_arr = np.asarray(pickle.load(f).get("scale", [])).flatten()
                if ref_scale_arr.size > 0:
                    ref_scale = float(ref_scale_arr.mean())
                    if abs(s - ref_scale) >= 0.05:
                        advisories.append(
                            f"{section}: scale {s:.4f} differs from reference {ref_scale:.4f} by "
                            f"{abs(s - ref_scale):.4f} >= 0.05 advisory"
                        )
                    results[section]["scale_ref_abs_diff"] = float(abs(s - ref_scale))

    # (a) c2ws still consistent with reference (post-step-9 comparison, same as step 5).
    rhino_c2ws_path = rhino_proc / "c2ws.npy"
    ref_c2ws_path = ref_proc / "c2ws.npy"
    if rhino_c2ws_path.exists() and ref_c2ws_path.exists():
        rh_c = np.load(rhino_c2ws_path)
        rf_c = np.load(ref_c2ws_path)
        if rh_c.shape == rf_c.shape == (expected_frames, 4, 4):
            rot_errs = geodesic_rotation_errors_mat(
                rh_c[:, :3, :3].astype(np.float64), rf_c[:, :3, :3].astype(np.float64)
            )
            trans_errs = np.linalg.norm(rh_c[:, :3, 3] - rf_c[:, :3, 3], axis=1)
            results[section]["c2ws_rot_err_mean_deg"] = float(rot_errs.mean())
            results[section]["c2ws_rot_err_max_deg"] = float(rot_errs.max())
            results[section]["c2ws_trans_err_mean"] = float(trans_errs.mean())
            results[section]["c2ws_trans_err_max"] = float(trans_errs.max())
        else:
            errors.append(f"{section}: c2ws shape mismatch (RHINO {rh_c.shape} vs ref {rf_c.shape})")

    # (b) intrinsic consistency with reference (step 9 must not have silently mutated it).
    rhino_intr_path = rhino_proc / "intrinsic.npy"
    ref_intr_path = ref_proc / "intrinsic.npy"
    if rhino_intr_path.exists() and ref_intr_path.exists():
        rh_intr = np.load(rhino_intr_path)
        rf_intr_full = np.load(ref_intr_path)
        # Reference may be stored as 4x4; take top-left 3x3 for comparison.
        rf_intr = rf_intr_full[:3, :3] if rf_intr_full.ndim == 2 and rf_intr_full.shape[0] >= 3 else rf_intr_full
        results[section]["intrinsic_rhino_shape"] = list(rh_intr.shape)
        results[section]["intrinsic_ref_shape"] = list(rf_intr_full.shape)
        if rh_intr.shape != (3, 3):
            errors.append(f"{section}: intrinsic.npy shape {rh_intr.shape} != (3,3)")
        else:
            results[section]["intrinsic_max_abs_diff"] = float(np.max(np.abs(rh_intr - rf_intr)))
            atol = 1e-4
            if not np.allclose(rh_intr, rf_intr, atol=atol):
                advisories.append(
                    f"{section}: intrinsic differs from reference (max abs diff "
                    f"{results[section]['intrinsic_max_abs_diff']:.6e} > atol={atol})"
                )

        # Cross-check against step-5 intrinsic_max_abs_diff: if step 5's diff already
        # exists in results, step 9 must not have introduced any *additional* drift.
        step5_key = "step5_processed_base"
        step5_diff = (results.get(step5_key) or {}).get("intrinsic_max_abs_diff")
        if step5_diff is not None and "intrinsic_max_abs_diff" in results[section]:
            step9_diff = results[section]["intrinsic_max_abs_diff"]
            delta = float(abs(step9_diff - step5_diff))
            results[section]["intrinsic_step9_minus_step5_abs_diff"] = delta
            # Allow float32 round-trip slack.
            if delta > 1e-6:
                errors.append(
                    f"{section}: intrinsic diff changed between step 5 ({step5_diff:.6e}) "
                    f"and step 9 ({step9_diff:.6e}); delta={delta:.6e} — step 9 appears to "
                    f"have mutated processed/intrinsic.npy."
                )

    # (c) byte-equality check against reconstruction source (proves step 9 is a no-op
    # on cameras when --normalize is false). Requires camera_pose_dir.
    if camera_pose_dir is None:
        candidate = rhino_dir / "camera_poses" / "mast3r" / "glomap_sparse_15" / "0"
        camera_pose_dir = candidate if candidate.exists() else None
    results[section]["camera_pose_dir"] = str(camera_pose_dir) if camera_pose_dir else None
    if camera_pose_dir is not None and camera_pose_dir.exists():
        # Read the COLMAP sparse model directly via hloc (stdlib+numpy only) instead of
        # `rhino.preprocessing.align_utils`, which transitively pulls in open3d/torch/cv2
        # and would either (a) silently fall into the `except` on stripped environments
        # or (b) fail to resolve because `python validate_processed_steps.py` does not
        # put the repo root on sys.path. Inlining keeps the byte-equality check reachable.
        try:
            from hloc.utils.read_write_model import (
                read_cameras_binary,
                read_images_binary,
            )

            def _qvec2rotmat(q):
                w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
                return np.array(
                    [
                        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
                        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                        [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
                    ],
                    dtype=np.float64,
                )

            cameras = read_cameras_binary(str(camera_pose_dir / "cameras.bin"))
            if len(cameras) != 1:
                raise ValueError(f"expected a single COLMAP camera, got {len(cameras)}")
            cam = next(iter(cameras.values()))
            model = cam.model
            p = cam.params
            if model == "SIMPLE_PINHOLE":
                fx, fy, cx, cy = p[0], p[0], p[1], p[2]
            elif model == "PINHOLE":
                fx, fy, cx, cy = p[0], p[1], p[2], p[3]
            elif model == "SIMPLE_RADIAL":
                fx, fy, cx, cy = p[0], p[0], p[1], p[2]
            else:
                raise NotImplementedError(f"unsupported COLMAP camera model: {model}")
            src_intr = np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
            )

            images = read_images_binary(str(camera_pose_dir / "images.bin"))
            bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
            c2ws_by_name = {}
            for img in images.values():
                R = _qvec2rotmat(img.qvec)
                t = np.asarray(img.tvec, dtype=np.float64).reshape(3, 1)
                w2c = np.concatenate([np.concatenate([R, t], axis=1), bottom], axis=0)
                c2w = np.linalg.inv(w2c)
                c2ws_by_name[img.name] = c2w.astype(np.float32)
            src_c2ws = np.array(
                [c2ws_by_name[k] for k in sorted(c2ws_by_name.keys())], dtype=np.float32
            )
        except Exception as ex:  # noqa: BLE001
            advisories.append(f"{section}: could not read reconstruction source for the byte-equality check: {ex}")
            src_intr = None
            src_c2ws = None

        if src_intr is not None and rhino_intr_path.exists():
            rh_intr = np.load(rhino_intr_path)
            if rh_intr.shape == src_intr.shape:
                intr_max_diff = float(np.max(np.abs(rh_intr.astype(np.float64) - src_intr.astype(np.float64))))
                results[section]["intrinsic_vs_source_max_abs_diff"] = intr_max_diff
                results[section]["intrinsic_vs_source_byte_equal"] = bool(
                    np.array_equal(rh_intr.astype(np.float32), src_intr.astype(np.float32))
                )
                if intr_max_diff > 1e-6:
                    errors.append(
                        f"{section}: intrinsic.npy differs from reconstruction source by "
                        f"{intr_max_diff:.6e} > 1e-6 — step 9 mutated intrinsic (unexpected with --normalize=False)."
                    )

        if src_c2ws is not None and (rhino_proc / "c2ws.npy").exists():
            rh_c = np.load(rhino_proc / "c2ws.npy")
            if rh_c.shape == src_c2ws.shape:
                c2w_max_diff = float(np.max(np.abs(rh_c.astype(np.float64) - src_c2ws.astype(np.float64))))
                results[section]["c2ws_vs_source_max_abs_diff"] = c2w_max_diff
                results[section]["c2ws_vs_source_byte_equal"] = bool(
                    np.array_equal(rh_c.astype(np.float32), src_c2ws.astype(np.float32))
                )
                if c2w_max_diff > 1e-6:
                    if normalize:
                        # --normalize=True intentionally centers+rescales c2ws translations;
                        # byte-equality is expected to fail. Report as advisory, not error.
                        advisories.append(
                            f"{section}: c2ws.npy differs from reconstruction source by "
                            f"{c2w_max_diff:.6e} — expected when --normalize=True (translations centered/rescaled)."
                        )
                    else:
                        errors.append(
                            f"{section}: c2ws.npy differs from reconstruction source by "
                            f"{c2w_max_diff:.6e} > 1e-6 — step 9 mutated c2ws (unexpected with --normalize=False)."
                        )

    # SMPL-X parameter comparison vs the canonical AiOS reference.
    ref_aligned_path = ref_proc / "aligned_smplx.pkl"
    results[section]["ref_aligned_path"] = str(ref_aligned_path)
    if ref_aligned_path.exists():
        try:
            with open(ref_aligned_path, "rb") as f:
                ref_aligned = pickle.load(f)
        except Exception as ex:  # noqa: BLE001
            advisories.append(
                f"{section}: failed to load reference {ref_aligned_path.name}: {ex}"
            )
            return

        rh_go = np.asarray(aligned.get("global_orient", []))
        rf_go = np.asarray(ref_aligned.get("global_orient", []))
        if rh_go.size and rf_go.size and rh_go.shape == rf_go.shape:
            def flatten_pose_blocks(d):
                blocks = [
                    np.asarray(d.get("global_orient", [])).reshape(expected_frames, -1, 3),
                    np.asarray(d.get("body_pose", [])).reshape(expected_frames, -1, 3),
                    np.asarray(d.get("left_hand_pose", [])).reshape(expected_frames, -1, 3),
                    np.asarray(d.get("right_hand_pose", [])).reshape(expected_frames, -1, 3),
                    np.asarray(d.get("jaw_pose", [])).reshape(expected_frames, -1, 3),
                ]
                return np.concatenate(blocks, axis=1)  # (F, J, 3)

            try:
                rh_all = flatten_pose_blocks(aligned)
                rf_all = flatten_pose_blocks(ref_aligned)
                if rh_all.shape == rf_all.shape:
                    Rh = axis_angle_to_matrix(rh_all)
                    Rf = axis_angle_to_matrix(rf_all)
                    # Per-frame mean joint error
                    per_frame = []
                    for i in range(rh_all.shape[0]):
                        per_frame.append(
                            float(geodesic_rotation_errors_mat(Rh[i], Rf[i]).mean())
                        )
                    per_frame = np.asarray(per_frame)
                    results[section]["smplx_rot_err_mean_deg"] = float(per_frame.mean())
                    results[section]["smplx_rot_err_max_deg"] = float(per_frame.max())
                    if per_frame.mean() >= 5.0:
                        advisories.append(
                            f"{section}: mean SMPL-X rotation error {per_frame.mean():.4f}° >= 5° advisory"
                        )
            except Exception as ex:  # noqa: BLE001
                advisories.append(f"{section}: pose comparison failed: {ex}")

        rh_tr = np.asarray(aligned.get("transl", []))
        rf_tr = np.asarray(ref_aligned.get("transl", []))
        if rh_tr.size and rf_tr.size and rh_tr.shape == rf_tr.shape:
            tr_err = np.linalg.norm(rh_tr - rf_tr, axis=-1)
            results[section]["smplx_transl_err_mean"] = float(tr_err.mean())
            results[section]["smplx_transl_err_max"] = float(tr_err.max())
            if tr_err.mean() >= 0.02:
                advisories.append(
                    f"{section}: mean SMPL-X translation error {tr_err.mean():.6f} >= 0.02 advisory"
                )

        rh_b = np.asarray(aligned.get("betas", [])).flatten()
        rf_b = np.asarray(ref_aligned.get("betas", [])).flatten()
        if rh_b.size == rf_b.size == SMPLX_BETAS_FULL:
            beta_l2 = float(np.linalg.norm(rh_b - rf_b))
            results[section]["betas_l2_diff"] = beta_l2
            if beta_l2 >= 0.05:
                advisories.append(
                    f"{section}: betas L2 diff {beta_l2:.4f} >= 0.05 advisory"
                )


# ---------------------------------------------------------------------------
# Step 10 — align_object_poses outputs
# ---------------------------------------------------------------------------


def validate_step10(rhino_dir: Path, ref_dir: Path, results: dict, errors: list,
                    advisories: list, expected_frames: int = 579) -> None:
    section = "step10_object_poses"
    results[section] = {}
    rhino_proc = rhino_dir / "processed"
    ref_proc = ref_dir / "processed"

    obj_path = rhino_proc / "object_poses.npy"
    obj_ctr_path = rhino_proc / "object_poses_centered.npy"
    pcd_path = rhino_proc / "object_pcd.ply"
    pcd_ctr_path = rhino_proc / "object_pcd_centered.ply"

    if not obj_path.exists():
        errors.append(f"{section}: missing {obj_path}")
    if not obj_ctr_path.exists():
        errors.append(f"{section}: missing {obj_ctr_path}")
    if not pcd_path.exists():
        errors.append(f"{section}: missing {pcd_path}")
    if not pcd_ctr_path.exists():
        errors.append(f"{section}: missing {pcd_ctr_path}")
    if not (obj_path.exists() and obj_ctr_path.exists()):
        return

    obj = np.load(obj_path)
    obj_ctr = np.load(obj_ctr_path)
    results[section]["object_poses_shape"] = list(obj.shape)
    results[section]["object_poses_centered_shape"] = list(obj_ctr.shape)
    target = (expected_frames, 4, 4)
    if obj.shape != target:
        errors.append(f"{section}: object_poses shape {obj.shape} != {target}")
    if obj_ctr.shape != target:
        errors.append(f"{section}: object_poses_centered shape {obj_ctr.shape} != {target}")

    # Centering consistency — translation offset must be constant across frames
    if obj.shape == obj_ctr.shape == target:
        offset = obj[:, :3, 3] - obj_ctr[:, :3, 3]
        offset_std = offset.std(axis=0)
        results[section]["centering_offset_std"] = offset_std.tolist()
        results[section]["centering_offset_mean"] = offset.mean(axis=0).tolist()
        if float(np.max(offset_std)) >= 1e-4:
            errors.append(
                f"{section}: centering offset not constant across frames "
                f"(max std {float(np.max(offset_std)):.3e} >= 1e-4)"
            )

    # Pose comparison with reference (advisory)
    ref_obj_path = ref_proc / "object_poses.npy"
    if ref_obj_path.exists() and obj.shape == target:
        rf_obj = np.load(ref_obj_path)
        if rf_obj.shape == obj.shape:
            # Per-frame rotation error
            rot_errs = geodesic_rotation_errors_mat(
                obj[:, :3, :3].astype(np.float64), rf_obj[:, :3, :3].astype(np.float64)
            )
            trans_errs = np.linalg.norm(obj[:, :3, 3] - rf_obj[:, :3, 3], axis=1)
            results[section]["ref_rot_err_mean_deg"] = float(rot_errs.mean())
            results[section]["ref_rot_err_max_deg"] = float(rot_errs.max())
            results[section]["ref_rot_err_p95_deg"] = float(np.percentile(rot_errs, 95))
            results[section]["ref_trans_err_mean"] = float(trans_errs.mean())
            results[section]["ref_trans_err_max"] = float(trans_errs.max())
            results[section]["ref_trans_err_p95"] = float(np.percentile(trans_errs, 95))

            # RMSE after Procrustes alignment on object centers (advisory)
            src = obj[:, :3, 3].astype(np.float64)
            dst = rf_obj[:, :3, 3].astype(np.float64)
            try:
                _, _, _, _, rmse = sim3_align(src, dst)
                results[section]["procrustes_rmse"] = float(rmse)
                if rmse >= 0.01:
                    advisories.append(
                        f"{section}: Procrustes-aligned translation RMSE {rmse:.4f} >= 0.01 advisory"
                    )
            except Exception as ex:  # noqa: BLE001
                advisories.append(f"{section}: Procrustes alignment failed: {ex}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@dataclass
class ValidateArgs:
    rhino_dir: Path
    """Path to RHINO output directory (e.g., .../RHINO/stage/S1_Take7)."""
    ref_dir: Path
    """Path to the reference run's sequence directory."""
    steps: str = "4,5,6,7,8,9,10"
    """Comma-separated step numbers to validate (subset of 4..10)."""
    expected_frames: Optional[int] = None
    """Expected number of frames in this sequence. Derived from
    processed/image (or images/selected_idxs.npy) when unset."""
    report_name: str = "processed_validation_report.json"
    """Filename for the JSON report (written under rhino_dir)."""
    camera_pose_dir: Optional[Path] = None
    """Reconstruction sparse dir for the step-9 byte-equality check. Defaults to
    rhino_dir/camera_poses/mast3r/glomap_sparse_15/0 if unset."""
    normalize: bool = False
    """Set to True if step 9 was run with --normalize (in-the-wild sequences).
    When True, c2ws byte-equality against the reconstruction source is expected
    to fail (normalize centers+rescales translations), so the diff is reported
    but does not produce a hard error. Intrinsic byte-equality still applies."""


def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    return obj


STEP_VALIDATORS = {
    4: validate_step4,
    5: validate_step5,
    6: validate_step6,
    7: validate_step7,
    8: validate_step8,
    9: validate_step9,
    10: validate_step10,
}


def main() -> None:
    import sys
    import tyro

    args = tyro.cli(ValidateArgs)
    if not args.rhino_dir.exists():
        print(f"ERROR: RHINO directory not found: {args.rhino_dir}")
        sys.exit(1)
    if not args.ref_dir.exists():
        print(f"ERROR: Reference directory not found: {args.ref_dir}")
        sys.exit(1)

    if args.expected_frames is None:
        image_dir = args.rhino_dir / "processed" / "image"
        n = len(list(image_dir.glob("*.png"))) if image_dir.is_dir() else 0
        if n == 0:
            sel = args.rhino_dir / "images" / "selected_idxs.npy"
            n = int(np.load(sel).shape[0]) if sel.exists() else 0
        if n == 0:
            print("ERROR: could not derive frame count; pass --expected-frames")
            sys.exit(1)
        args.expected_frames = n

    requested = sorted({int(s) for s in args.steps.split(",") if s.strip()})
    if not args.normalize and (
            args.rhino_dir / "processed" / "scene_normalization.json").exists():
        print("Note: scene_normalization.json found — treating as scene-normalized sequence")
        args.normalize = True
    results: dict = {}
    errors: list[str] = []
    advisories: list[str] = []

    print("=" * 72)
    print("Validating RHINO preprocessing steps 4-10 (selected)")
    print(f"  RHINO dir: {args.rhino_dir}")
    print(f"  Ref dir:   {args.ref_dir}")
    print(f"  Steps:     {requested}")
    print("=" * 72)

    for step in requested:
        validator = STEP_VALIDATORS.get(step)
        if validator is None:
            advisories.append(f"step{step}: no validator implemented yet")
            continue
        print(f"\n--- Step {step} ---")
        if step == 9:
            validator(args.rhino_dir, args.ref_dir, results, errors, advisories,
                      expected_frames=args.expected_frames,
                      camera_pose_dir=args.camera_pose_dir,
                      normalize=args.normalize)
        else:
            validator(args.rhino_dir, args.ref_dir, results, errors, advisories,
                      expected_frames=args.expected_frames)
        section_key = next(iter(k for k in results if k.startswith(f"step{step}")), None)
        if section_key:
            for k, v in results[section_key].items():
                if isinstance(v, dict):
                    print(f"  {k}:")
                    for kk, vv in v.items():
                        print(f"    {kk}: {vv}")
                else:
                    print(f"  {k}: {v}")

    print("\n" + "=" * 72)
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
    else:
        print("All structural checks PASSED")
    if advisories:
        print(f"\nADVISORIES ({len(advisories)}):")
        for a in advisories:
            print(f"  ~ {a}")
    print("=" * 72)

    report_path = args.rhino_dir / args.report_name
    if report_path.exists():
        try:
            with open(report_path) as f:
                existing = json.load(f)
            existing_results = existing.get("results", {})
            existing_results.update(results)
            existing_errors = list(existing.get("errors", []))
            existing_errors.extend(errors)
            existing_advisories = list(existing.get("advisories", []))
            existing_advisories.extend(advisories)
            payload = {
                "results": existing_results,
                "errors": existing_errors,
                "advisories": existing_advisories,
            }
        except json.JSONDecodeError:
            payload = {"results": results, "errors": errors, "advisories": advisories}
    else:
        payload = {"results": results, "errors": errors, "advisories": advisories}

    with open(report_path, "w") as f:
        json.dump(sanitize(payload), f, indent=2)
    print(f"\nReport saved to {report_path}")

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
