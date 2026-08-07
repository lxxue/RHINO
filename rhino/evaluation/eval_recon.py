"""Per-frame ICP alignment + chamfer / Hausdorff / F-score metrics vs GT meshes."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
import tyro
from tqdm import tqdm

from rhino.evaluation.mesh_sequence import find_latest_epoch, validate_mesh_sequences
from rhino.evaluation.metrics import (
    compute_chamfer_distance_from_pcds,
    compute_f1_score_from_pcds,
    compute_hausdorff_distance_from_pcds,
)

ICP_MAX_CORRESPONDENCE_DISTANCE = 0.1
METRIC_NAMES = ("cd", "cd_p2g", "cd_g2p", "hd", "f1_1cm", "f1_2cm", "f1_5cm")


def _preflight(
    gt_dir: Path,
    pred_dir: Path,
    seq_name: str,
    subdir: str,
    version_name: str,
    take_name: str,
    offset: int,
    expected_gt_frames: Optional[int] = None,
):
    """Resolve and validate the complete predicted/GT sequence contract."""
    gt_frames_dir = gt_dir / take_name / "frames"
    version_dir = pred_dir / seq_name / subdir / version_name
    epoch_dir = find_latest_epoch(version_dir)
    sequence = validate_mesh_sequences(
        gt_frames_dir, epoch_dir, offset, expected_gt_frames
    )
    return epoch_dir, sequence


def eval_seq(
    seq_name: str,
    version_name: str,
    take_name: str,
    offset: int,
    gt_dir: Path,
    pred_dir: Path,
    output_dir: Path,
    subdir: str = "hoi",
    expected_gt_frames: Optional[int] = None,
):
    """Per-frame ICP + metrics for one (seq, subdir, version) triple."""
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)
    output_dir = Path(output_dir)
    print(
        f"Evaluating {seq_name}/{subdir}/{version_name} vs GT take "
        f"{take_name} (offset {offset})"
    )

    epoch_dir, sequence = _preflight(
        gt_dir,
        pred_dir,
        seq_name,
        subdir,
        version_name,
        take_name,
        offset,
        expected_gt_frames,
    )
    print(f"Predicted meshes: {epoch_dir}")

    num_frames_gt = len(sequence.gt_files)
    num_frames_pred = len(sequence.object_files)
    num_overlap = len(sequence.frame_pairs)

    aligned_mesh_dir = output_dir / seq_name / subdir / version_name / "aligned_meshes"
    gt_mesh_dir = output_dir / seq_name / "gt_meshes"
    aligned_mesh_dir.mkdir(exist_ok=True, parents=True)
    gt_mesh_dir.mkdir(exist_ok=True, parents=True)

    transformation = None
    init_trans = np.identity(4)
    metrics = {name: [] for name in METRIC_NAMES}

    for pred_id, gt_id in tqdm(sequence.frame_pairs):
        human_mesh = o3d.io.read_triangle_mesh(str(sequence.human_files[pred_id]))
        obj_mesh = o3d.io.read_triangle_mesh(str(sequence.object_files[pred_id]))
        if not human_mesh.has_vertices():
            raise ValueError(f"Human mesh is empty: {sequence.human_files[pred_id]}")
        if not obj_mesh.has_vertices():
            raise ValueError(f"Object mesh is empty: {sequence.object_files[pred_id]}")
        pred_mesh = o3d.geometry.TriangleMesh()
        pred_mesh += human_mesh
        pred_mesh += obj_mesh
        pred_pcd = o3d.geometry.PointCloud()
        pred_pcd.points = pred_mesh.vertices

        gt_mesh = o3d.io.read_triangle_mesh(str(sequence.gt_files[gt_id]))
        if not gt_mesh.has_vertices():
            raise ValueError(f"GT mesh is empty: {sequence.gt_files[gt_id]}")
        gt_mesh.vertices = o3d.utility.Vector3dVector(
            np.array(gt_mesh.vertices) / 1000.0
        )
        gt_mesh.compute_vertex_normals()
        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = gt_mesh.vertices
        gt_pcd.normals = gt_mesh.vertex_normals

        if transformation is None:
            reg = o3d.pipelines.registration.registration_icp(
                pred_pcd,
                gt_pcd,
                ICP_MAX_CORRESPONDENCE_DISTANCE,
                init_trans,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(
                    with_scaling=True
                ),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000),
            )
            transformation = reg.transformation
            init_trans = transformation
            np.save(
                output_dir / seq_name / subdir / version_name / "init_trans.npy",
                transformation,
            )

        pred_mesh.transform(transformation)
        o3d.io.write_triangle_mesh(
            str(aligned_mesh_dir / f"{pred_id:04d}.ply"), pred_mesh
        )
        gt_out = gt_mesh_dir / f"{pred_id:04d}_gt.ply"
        if not gt_out.exists():
            o3d.io.write_triangle_mesh(str(gt_out), gt_mesh)

        pred_pcd.transform(transformation)
        cd, cd_p2g, cd_g2p = compute_chamfer_distance_from_pcds(pred_pcd, gt_pcd)
        hd = compute_hausdorff_distance_from_pcds(pred_pcd, gt_pcd)
        f1_1cm, _, _ = compute_f1_score_from_pcds(pred_pcd, gt_pcd, threshold=0.01)
        f1_2cm, _, _ = compute_f1_score_from_pcds(pred_pcd, gt_pcd, threshold=0.02)
        f1_5cm, _, _ = compute_f1_score_from_pcds(pred_pcd, gt_pcd, threshold=0.05)

        frame_metrics = {
            "cd": cd,
            "cd_p2g": cd_p2g,
            "cd_g2p": cd_g2p,
            "hd": hd,
            "f1_1cm": f1_1cm,
            "f1_2cm": f1_2cm,
            "f1_5cm": f1_5cm,
        }
        if not np.isfinite(list(frame_metrics.values())).all():
            raise FloatingPointError(
                f"Nonfinite evaluation metric for predicted frame {pred_id}, GT frame {gt_id}"
            )
        for name, value in frame_metrics.items():
            metrics[name].append(float(value))

    for name in METRIC_NAMES:
        print(f"Average {name}: {np.mean(metrics[name]) * 100}")

    provenance = {
        "gt_dir": str(gt_dir),
        "take_name": take_name,
        "offset": int(offset),
        "pred_dir": str(pred_dir),
        "subdir": subdir,
        "version_name": version_name,
        "seq_name": seq_name,
        "epoch_dir": str(epoch_dir),
        "num_frames_gt": int(num_frames_gt),
        "num_frames_pred": int(num_frames_pred),
        "num_overlap": int(num_overlap),
        "alignment_mode": "first_frame_icp_with_scaling",
    }

    metric_dict = {**metrics, "_provenance": provenance}
    json_path = output_dir / seq_name / subdir / version_name / f"{seq_name}.json"
    with open(json_path, "w") as f:
        json.dump(metric_dict, f, indent=4)
    print(f"Wrote metrics: {json_path}")


if __name__ == "__main__":
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    tyro.cli(eval_seq)
