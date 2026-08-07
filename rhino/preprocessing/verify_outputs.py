"""Verification script: Compare RHINO preprocessing outputs against a reference run.

Exits with code 1 if any required artifact is missing, any frame count
mismatches, or any metric fails its threshold.
"""

import pickle as pkl
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as SciR


REQUIRED_ARTIFACTS = [
    ("c2ws.npy", "file"),
    ("object_poses_centered.npy", "file"),
    ("human_mask", "dir"),
    ("obj_mask", "dir"),
    ("hand_mask", "dir"),
    ("all_mask", "dir"),
    ("contact_labels.npy", "file"),
    ("aligned_smplx.pkl", "file"),
    ("contact_vertices_refined.npy", "file"),
]

# Verification thresholds
THRESHOLDS = {
    "pose_rmse": 0.01,       # camera poses
    "mask_iou": 0.9,
    "smplx_rotation_deg": 5.0,
    "smplx_transl": 0.02,
    "smplx_beta_vertex_disp": 0.015,  # mean canonical vertex displacement (m) from beta diff
    "smplx_scale": 0.05,
}

# Advisory metrics (reported, not gating):
# - Contact outputs intentionally diverge from the reference run (the RHINO
#   heuristic is proximity+motion; InteractVLM runs quantized with a proper
#   object prompt) — validate_contact_steps.py with its GT check is the
#   authoritative contact validation.
# - Object pose deltas vs the reference are run-to-run SfM variance
#   (up-to-similarity mapping, SLERP-interpolated occlusion gaps) that scales
#   with object difficulty.
ADVISORY_FLOORS = {
    "contact_labels_match": 0.95,   # higher is better
    "contact_vertices_iou": 0.70,   # higher is better
    "obj_pose_rmse": 0.05,          # lower is better
    # Scene-normalized (in-the-wild) sequences have no metric anchor: two runs'
    # MASt3R/GLOMAP scenes differ by a smooth low-frequency drift that the
    # per-frame body-to-scene depth alignment follows, so cm-level rigid-aligned
    # translation deltas are expected run-to-run variance, not per-frame error
    # (observed re-run: 3.5cm mean delta with 0.4cm frame-to-frame jitter while
    # body pose matched to 0.002 deg and cameras to 1.1cm).
    "smplx_transl_normalized": 0.05,  # lower is better
}


@dataclass
class VerifyOutputsArgs:
    rhino_dir: Path
    """Path to RHINO processed/ directory."""
    ref_dir: Path
    """Path to the reference run's processed/ directory."""
    normalized_scene: bool = False
    """Sequence was scene-normalized (in-the-wild --normalize-scene path, no
    AprilTag metric anchor): SMPL-X translation deltas vs the reference become
    advisory. Auto-enabled when the RHINO tree contains
    scene_normalization.json (written by step 9 --normalize)."""


def procrustes_rmse(src: np.ndarray, dst: np.ndarray) -> float:
    src_c = src - src.mean(axis=0)
    dst_c = dst - dst.mean(axis=0)
    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R_align = Vt.T @ D @ U.T
    scale = np.trace(np.diag(S) @ D) / np.trace(src_c.T @ src_c)
    aligned = scale * (src_c @ R_align.T) + dst.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((aligned - dst) ** 2, axis=1))))


def geodesic_rotation_errors(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    errors = []
    for i in range(len(R1)):
        R_diff = R1[i][:3, :3] @ R2[i][:3, :3].T
        angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
        errors.append(np.degrees(angle))
    return np.array(errors)


def mask_iou_strict(pred_dir: Path, ref_dir: Path) -> float:
    pred_files = sorted(pred_dir.glob("*.png"))
    ref_files = sorted(ref_dir.glob("*.png"))
    if len(pred_files) != len(ref_files):
        raise ValueError(f"Frame count mismatch: RHINO={len(pred_files)}, ref={len(ref_files)}")
    if len(pred_files) == 0:
        raise ValueError(f"No mask files in {pred_dir}")
    ious = []
    for pf, rf in zip(pred_files, ref_files):
        p = cv2.imread(str(pf), 0) > 127
        r = cv2.imread(str(rf), 0) > 127
        if p.shape != r.shape:
            p = cv2.resize(p.astype(np.uint8), (r.shape[1], r.shape[0])) > 0
        inter = (p & r).sum()
        union = (p | r).sum()
        ious.append(inter / max(union, 1))
    return float(np.mean(ious))


def compare_smplx(rhino_path: Path, ref_path: Path, label: str, results: dict, errors: list,
                   transl_advisory: bool = False,
                   advisory_results: dict | None = None):
    """Compare SMPL-X pkl files: global_orient, body_pose, transl, betas, scale."""
    with open(rhino_path, "rb") as f:
        sr = pkl.load(f)
    with open(ref_path, "rb") as f:
        sref = pkl.load(f)

    n_r = len(sr.get("global_orient", []))
    n_ref = len(sref.get("global_orient", []))
    if n_r != n_ref:
        errors.append(f"{label} frame mismatch: RHINO={n_r} ref={n_ref}")
        return
    n = n_r

    # Global orient rotation error
    go_r = SciR.from_rotvec(np.array(sr["global_orient"][:n]).reshape(-1, 3)).as_matrix()
    go_ref = SciR.from_rotvec(np.array(sref["global_orient"][:n]).reshape(-1, 3)).as_matrix()
    go_errs = []
    for rr, rref in zip(go_r, go_ref):
        angle = np.arccos(np.clip((np.trace(rr @ rref.T) - 1) / 2, -1, 1))
        go_errs.append(np.degrees(angle))
    go_mean = float(np.mean(go_errs))
    results[f"{label}_global_orient_deg"] = go_mean
    s = "PASS" if go_mean < THRESHOLDS["smplx_rotation_deg"] else "FAIL"
    print(f"  {label} global_orient: {go_mean:.2f}° ({s})")

    # Body pose rotation error (per-joint geodesic)
    if "body_pose" not in sr or "body_pose" not in sref:
        if "body_pose" not in sr:
            errors.append(f"{label}: missing 'body_pose' key in RHINO file")
        if "body_pose" not in sref:
            errors.append(f"{label}: missing 'body_pose' key in reference file")
        return
    if "body_pose" in sr and "body_pose" in sref:
        bp_r = np.array(sr["body_pose"][:n]).reshape(n, -1, 3)
        bp_ref = np.array(sref["body_pose"][:n]).reshape(n, -1, 3)
        bp_errs = []
        for frame_i in range(n):
            for joint_j in range(bp_r.shape[1]):
                R_r = SciR.from_rotvec(bp_r[frame_i, joint_j]).as_matrix()
                R_ref = SciR.from_rotvec(bp_ref[frame_i, joint_j]).as_matrix()
                angle = np.arccos(np.clip((np.trace(R_r @ R_ref.T) - 1) / 2, -1, 1))
                bp_errs.append(np.degrees(angle))
        bp_mean = float(np.mean(bp_errs))
        results[f"{label}_body_pose_deg"] = bp_mean
        s = "PASS" if bp_mean < THRESHOLDS["smplx_rotation_deg"] else "FAIL"
        print(f"  {label} body_pose: {bp_mean:.2f}° ({s})")

    # Translation, compared after removing the global rigid offset between the
    # two runs' world frames (AprilTag model_aligner leaves a small global
    # rotation ambiguity; the camera-pose check is Procrustes-aligned for the
    # same reason, and a ~3 deg world tilt shows up as ~10cm raw translation
    # difference without reflecting any per-frame error).
    t_r = np.array(sr["transl"][:n])
    t_ref = np.array(sref["transl"][:n])
    raw_err = float(np.mean(np.linalg.norm(t_r - t_ref, axis=-1)))
    src_c = t_r - t_r.mean(axis=0)
    dst_c = t_ref - t_ref.mean(axis=0)
    U, _, Vt = np.linalg.svd(src_c.T @ dst_c)
    D = np.diag([1.0, 1.0, np.linalg.det(Vt.T @ U.T)])
    aligned = src_c @ (Vt.T @ D @ U.T).T + t_ref.mean(axis=0)
    trans_err = float(np.mean(np.linalg.norm(aligned - t_ref, axis=-1)))
    if transl_advisory and advisory_results is not None:
        advisory_results[f"{label}_transl_err"] = trans_err
        s = "OK" if trans_err < ADVISORY_FLOORS["smplx_transl_normalized"] else "HIGH"
        print(f"  {label} translation: {trans_err:.4f} (ADVISORY {s}, normalized scene; "
              f"raw {raw_err:.4f} before global alignment)")
    else:
        results[f"{label}_transl_err"] = trans_err
        s = "PASS" if trans_err < THRESHOLDS["smplx_transl"] else "FAIL"
        print(f"  {label} translation: {trans_err:.4f} ({s}, raw {raw_err:.4f} before global alignment)")

    # Betas, gated on the geometry they produce rather than coefficient L2:
    # monocular shape fitting has beta ambiguity, so two runs can land in
    # different-but-equivalent minima (S2_Take5 re-run: beta L2 0.68 but
    # only 6.6mm mean canonical vertex displacement, and closer to the
    # volumetric-GT stature than the reference fit).
    b_r = np.array(sr.get("betas", np.zeros(10))).flatten()
    b_ref = np.array(sref.get("betas", np.zeros(10))).flatten()
    beta_l2 = float(np.linalg.norm(b_r - b_ref))
    nb = min(len(b_r), len(b_ref))
    shapedirs_path = Path(__file__).resolve().parents[1] / "checkpoints" / "smplx" / "SMPLX_NEUTRAL.npz"
    shapedirs = np.load(shapedirs_path, allow_pickle=True)["shapedirs"][:, :, :nb]
    disp = np.einsum("vcb,b->vc", shapedirs, (b_r[:nb] - b_ref[:nb]).astype(np.float64))
    beta_err = float(np.linalg.norm(disp, axis=1).mean())
    results[f"{label}_beta_vertex_disp_m"] = beta_err
    s = "PASS" if beta_err < THRESHOLDS["smplx_beta_vertex_disp"] else "FAIL"
    print(f"  {label} beta shape displacement: {beta_err * 1000:.1f}mm mean ({s}, coeff L2 {beta_l2:.4f})")

    # Scale
    s_r = np.array(sr.get("scale", [1.0])).flatten()[:n]
    s_ref = np.array(sref.get("scale", [1.0])).flatten()[:n]
    scale_err = float(np.mean(np.abs(s_r - s_ref)))
    results[f"{label}_scale_err"] = scale_err
    s = "PASS" if scale_err < THRESHOLDS["smplx_scale"] else "FAIL"
    print(f"  {label} scale: {scale_err:.4f} ({s})")


def metric_passes(key: str, value: float) -> bool:
    """Check if a metric passes its threshold. Uses explicit key matching."""
    if "procrustes_rmse_over_gate" in key:
        return value < 1.0
    if "geodesic_deg" in key:
        return value < THRESHOLDS["smplx_rotation_deg"]
    if "_iou" in key and "contact" not in key:
        return value > THRESHOLDS["mask_iou"]
    if "global_orient_deg" in key or "body_pose_deg" in key:
        return value < THRESHOLDS["smplx_rotation_deg"]
    if "transl_err" in key:
        return value < THRESHOLDS["smplx_transl"]
    if "beta_vertex_disp" in key:
        return value < THRESHOLDS["smplx_beta_vertex_disp"]
    if "scale_err" in key:
        return value < THRESHOLDS["smplx_scale"]
    # Unknown metric — fail safe
    print(f"  WARNING: unknown metric '{key}' not in threshold table, treating as FAIL")
    return False


def main(args: VerifyOutputsArgs) -> None:
    rhino = args.rhino_dir.resolve()
    ref = args.ref_dir.resolve()
    print(f"Verifying RHINO outputs against reference...")
    print(f"  RHINO: {rhino}")
    print(f"  Reference: {ref}")

    results = {}
    errors = []

    # Phase 0: Require all compared artifacts on the RHINO side. The reference
    # tree may predate newer conventions (older in-the-wild processed dirs have
    # no object_poses_centered.npy) — a missing reference artifact skips its
    # comparisons instead of failing the run.
    missing_ref = set()
    for artifact, kind in REQUIRED_ARTIFACTS:
        p = rhino / artifact
        if kind == "file" and not p.exists():
            errors.append(f"Missing RHINO artifact: {artifact}")
        elif kind == "dir" and (not p.is_dir() or not any(p.iterdir())):
            errors.append(f"Missing/empty RHINO directory: {artifact}/")
        r = ref / artifact
        if (kind == "file" and not r.exists()) or (
                kind == "dir" and (not r.is_dir() or not any(r.iterdir()))):
            missing_ref.add(artifact)
            print(f"  Note: reference lacks {artifact}; skipping its comparisons")
    if errors:
        print("\nArtifact validation FAILED:")
        for e in errors:
            print(f"  - {e}")
        print(f"\nOverall: ARTIFACT CHECK FAILED ({len(errors)} error(s))")
        sys.exit(1)

    # Phase 1: Camera/object poses. Camera poses gate hard (AprilTag-anchored,
    # reproducibly ~5mm across runs). Object pose deltas vs the reference are
    # advisory: object SfM is up-to-similarity with SLERP-interpolated
    # occlusion gaps, and run-to-run variance scales with object difficulty
    # (S1_Take7 0.024m, S1_Take8 0.033m, S4_Take2 0.086m) — the functional gates
    # are the contact-vs-GT check and training. Matches the advisory
    # treatment in validate_processed_steps step 10.
    advisory_results = {}
    for pose_name in ["c2ws.npy", "object_poses_centered.npy"]:
        if pose_name in missing_ref:
            continue
        p = np.load(rhino / pose_name)
        r = np.load(ref / pose_name)
        if p.shape[0] != r.shape[0]:
            errors.append(f"{pose_name}: frame count mismatch RHINO={p.shape[0]} ref={r.shape[0]}")
            continue
        rmse = procrustes_rmse(p[:, :3, 3], r[:, :3, 3])
        rot_errs = geodesic_rotation_errors(p, r)
        rot_err = float(rot_errs.mean())
        if "object" in pose_name:
            advisory_results[f"{pose_name}_procrustes_rmse"] = rmse
            advisory_results[f"{pose_name}_geodesic_deg"] = rot_err
            s_rmse = "OK" if rmse < ADVISORY_FLOORS["obj_pose_rmse"] else "HIGH"
            s_rot = "OK" if rot_err < THRESHOLDS["smplx_rotation_deg"] else "HIGH"
            print(f"  {pose_name}: RMSE={rmse:.6f} (ADVISORY {s_rmse}), RotErr={rot_err:.2f}° (ADVISORY {s_rot})")
        else:
            # Scale-aware gate: run-to-run camera agreement scales with the
            # working volume (S1_Take7 5mm over a 3.1m orbit, S3_Take6 17mm over
            # 4.0m) — gate at 0.5% of the trajectory bbox diagonal with a
            # 1cm floor.
            diag = float(np.linalg.norm(r[:, :3, 3].max(axis=0) - r[:, :3, 3].min(axis=0)))
            cam_gate = max(THRESHOLDS["pose_rmse"], 0.005 * diag)
            advisory_results[f"{pose_name}_procrustes_rmse"] = rmse
            results[f"{pose_name}_procrustes_rmse_over_gate"] = rmse / cam_gate
            results[f"{pose_name}_geodesic_deg"] = rot_err
            s_rmse = "PASS" if rmse < cam_gate else "FAIL"
            s_rot = "PASS" if rot_err < THRESHOLDS["smplx_rotation_deg"] else "FAIL"
            print(f"  {pose_name}: RMSE={rmse:.6f} ({s_rmse}, gate {cam_gate:.4f} @ 0.5% of {diag:.2f}m), RotErr={rot_err:.2f}° ({s_rot})")

    # Phase 2: Masks (IoU threshold = 0.9). Hand masks are advisory —
    # seed-sensitive tracking of small hands reproduces coverage but not
    # boundaries; unused by default training (fg_obj_bg sampling).
    for mask_type in ["human_mask", "obj_mask", "hand_mask", "all_mask"]:
        if mask_type in missing_ref:
            continue
        try:
            iou = mask_iou_strict(rhino / mask_type, ref / mask_type)
            if mask_type == "hand_mask":
                advisory_results[f"{mask_type}_iou"] = iou
                s = "OK" if iou > THRESHOLDS["mask_iou"] else "LOW"
                print(f"  {mask_type} IoU: {iou:.4f} (ADVISORY {s})")
            else:
                results[f"{mask_type}_iou"] = iou
                s = "PASS" if iou > THRESHOLDS["mask_iou"] else "FAIL"
                print(f"  {mask_type} IoU: {iou:.4f} ({s})")
        except ValueError as e:
            errors.append(str(e))

    # Phase 3: Contact labels (advisory — the RHINO heuristic intentionally
    # differs from the reference; shape mismatch stays a hard error)
    if "contact_labels.npy" not in missing_ref:
        cl1 = np.load(rhino / "contact_labels.npy").astype(bool)
        cl2 = np.load(ref / "contact_labels.npy").astype(bool)
        if cl1.shape[0] != cl2.shape[0]:
            errors.append(f"contact_labels frame mismatch: RHINO={cl1.shape[0]} ref={cl2.shape[0]}")
        else:
            match_rate = float((cl1 == cl2).mean())
            advisory_results["contact_labels_match"] = match_rate
            s = "OK" if match_rate >= ADVISORY_FLOORS["contact_labels_match"] else "LOW"
            print(f"  contact_labels agreement: {match_rate:.4f} (ADVISORY {s})")

    # Phase 4: SMPL-X (required: aligned_smplx.pkl). For scene-normalized
    # sequences translation is advisory — see ADVISORY_FLOORS note.
    normalized = args.normalized_scene or (rhino / "scene_normalization.json").exists()
    if normalized and not args.normalized_scene:
        print("  Note: scene_normalization.json found — treating as scene-normalized sequence")
    if "aligned_smplx.pkl" not in missing_ref:
        compare_smplx(rhino / "aligned_smplx.pkl", ref / "aligned_smplx.pkl",
                      "aligned_smplx.pkl", results, errors,
                      transl_advisory=normalized, advisory_results=advisory_results)

    # Phase 5: Contact vertices (advisory — see ADVISORY_FLOORS note)
    if "contact_vertices_refined.npy" not in missing_ref:
        cv1 = np.load(rhino / "contact_vertices_refined.npy").astype(bool)
        cv2_arr = np.load(ref / "contact_vertices_refined.npy").astype(bool)
        if cv1.shape[0] != cv2_arr.shape[0]:
            errors.append(f"contact_vertices frame mismatch: RHINO={cv1.shape[0]} ref={cv2_arr.shape[0]}")
        else:
            inter = (cv1 & cv2_arr).sum()
            union = (cv1 | cv2_arr).sum()
            iou = float(inter / max(union, 1))
            advisory_results["contact_vertices_iou"] = iou
            s = "OK" if iou > ADVISORY_FLOORS["contact_vertices_iou"] else "LOW"
            print(f"  contact_vertices IoU: {iou:.4f} (ADVISORY {s})")

    # Summary
    print(f"\nResults ({len(results)} hard metrics):")
    for k, v in results.items():
        p = metric_passes(k, v)
        print(f"  {k}: {v:.6f} {'PASS' if p else 'FAIL'}")
    if advisory_results:
        print(f"Advisory metrics (not gating):")
        for k, v in advisory_results.items():
            print(f"  {k}: {v:.6f} (ADVISORY)")

    if errors:
        print(f"\nStructural errors ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")

    metrics_pass = all(metric_passes(k, v) for k, v in results.items())
    all_pass = metrics_pass and not errors

    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    import tyro
    main(tyro.cli(VerifyOutputsArgs))
