"""Step 9: SMPL-X scene alignment.

Non-interactive by default. Use --interactive for Open3D visualization.

When --normalize is set, the scene (cameras, point cloud) is centered and
scaled using the SMPL-X mean translation and scale. The (possibly
normalized) cameras and point cloud are written to data_dir/processed/
for use by downstream steps (e.g. step 10). Normalization is needed for
in-the-wild sequences; stage sequences with pre-calibrated cameras skip it.
"""

import json
import pickle as pkl
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm

from rhino.preprocessing.align_utils import (
    convert_smplx_to_world,
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_point_cloud_binary,
    read_smplx,
    solve_transformation,
    transform_verts_with_scale,
    visualize_smplx_and_scene,
)
from rhino.preprocessing.tool_config import ToolConfig
from rhino.rhino_smplx.smplx_server import SMPLXServer


@dataclass
class AlignSMPLXArgs:
    data_dir: Path
    """Path to the sequence data directory."""
    camera_pose_dir: Path
    """Path to COLMAP sparse reconstruction (with cameras.bin, images.bin, points3D.bin)."""
    interactive: bool = False
    """Enable Open3D visualization for manual scale adjustment."""
    manual_scale: float = -1.0
    """Manual scale override. If negative, auto-compute from ground plane."""
    normalize: bool = False
    """Normalize scene using SMPL-X mean scale/translation. Needed for in-the-wild sequences."""
    strict_scale: bool = True
    """Require the ground-plane scale to be near 1 (metric reconstructions). Use
    --no-strict-scale for non-metric SfM backends, where the scale is arbitrary."""
    tools: ToolConfig = field(default_factory=ToolConfig)


def main(args: AlignSMPLXArgs) -> None:
    data_dir = args.data_dir.resolve()
    camera_pose_dir = args.camera_pose_dir.resolve()
    print("Aligning refined AiOS SMPL-X estimates to the scene...")

    # Read SMPL-X estimates
    smplx_dir = data_dir / "smplx" / "aios_refined"
    raw_smplx = read_smplx(smplx_dir)
    betas = torch.from_numpy(raw_smplx["betas"]).float()
    smplx_server = SMPLXServer(gender="neutral", betas=betas[None])

    # Read scene reconstruction
    c2ws = read_extrinsics_binary(camera_pose_dir)
    scene_pcd = read_point_cloud_binary(camera_pose_dir)
    scene_pcd_legacy = scene_pcd.to_legacy() if hasattr(scene_pcd, "to_legacy") else scene_pcd

    # Fit ground plane (tight threshold, high iteration count)
    plane_model, inliers = scene_pcd_legacy.segment_plane(
        distance_threshold=0.0005, ransac_n=3, num_iterations=10000
    )
    print(f"Ground plane: {plane_model}")

    # Convert all frames to world coordinates
    new_smplxs = {
        "scale": [], "global_orient": [], "body_pose": [],
        "left_hand_pose": [], "right_hand_pose": [],
        "transl": [], "expression": [], "jaw_pose": [],
    }

    num_frames = len(raw_smplx["global_orient"])
    c2w_keys = sorted(c2ws.keys())
    suffix = c2w_keys[0].split(".")[-1]

    # Compute per-frame scale and alignment
    scales = []
    for idx in tqdm(range(num_frames), desc="Aligning frames"):
        fname = f"{idx:04d}.{suffix}"
        if fname not in c2ws:
            raise RuntimeError(f"No camera pose for frame {idx} ({fname})")
        c2w = c2ws[fname]

        transformed_verts, scale_i = solve_transformation(
            raw_smplx["smplx_verts"][idx], plane_model, c2w
        )

        if args.manual_scale > 0:
            scale_i = args.manual_scale
            transformed_verts = transform_verts_with_scale(
                raw_smplx["smplx_verts"][idx], c2w, scale_i
            )

        convert_smplx_to_world(
            smplx_server, raw_smplx, idx, scale_i,
            c2w, transformed_verts[:, :3], new_smplxs
        )
        scales.append(scale_i)

    # Use 20th percentile as robust mean scale, then override all per-frame scales
    mean_scale = float(np.percentile(scales, 20))
    print(f"Per-frame scales: min={min(scales):.4f}, max={max(scales):.4f}, "
          f"mean={np.mean(scales):.4f}, p20={mean_scale:.4f}")
    # Stage sequences are metric-scaled via AprilTag model_aligner, so the
    # SMPL-X alignment scale must come out near 1. In-the-wild scenes have
    # arbitrary SfM scale; --normalize exists precisely to remove it.
    if not args.normalize and not 0.90 <= mean_scale <= 1.10:
        msg = (
            f"mean_scale {mean_scale:.4f} outside expected range [0.90, 1.10]. "
            f"Check that the scene reconstruction is at human scale."
        )
        if args.strict_scale:
            raise RuntimeError(msg + " Pass --no-strict-scale for non-metric SfM backends.")
        print(f"WARNING: {msg} Continuing because --no-strict-scale is set.")
    for i in range(num_frames):
        new_smplxs["scale"][i] = mean_scale

    # Keep list format; add betas
    new_smplxs["betas"] = raw_smplx["betas"]

    # Normalize scene using SMPL-X alignment parameters
    intrinsic = read_intrinsics_binary(camera_pose_dir)
    if args.normalize:
        # Compute center from world-space SMPL-X translations
        translations_world = []
        for i in range(num_frames):
            t_w = np.array(new_smplxs["transl"][i], dtype=np.float64)
            s_i = float(new_smplxs["scale"][i])
            translations_world.append(t_w * s_i)
        center = np.mean(translations_world, axis=0)

        # Normalize c2ws
        for key in c2ws:
            c2ws[key][:3, 3] -= center.astype(np.float32)
            c2ws[key][:3, 3] /= mean_scale

        # Normalize scene point cloud
        scene_pcd_legacy.translate(-center, relative=True)
        scene_pcd_legacy.scale(1.0 / mean_scale, center=np.zeros((3, 1)))

        # Normalize SMPL-X params
        for i in range(num_frames):
            s_i = float(new_smplxs["scale"][i])
            t_i = np.array(new_smplxs["transl"][i], dtype=np.float64)
            new_smplxs["scale"][i] = np.float32(s_i / mean_scale)
            new_smplxs["transl"][i] = ((t_i * s_i - center) / mean_scale).astype(np.float32)

        # Record normalization provenance: validators switch to
        # normalized-scene comparison semantics (no metric anchor) based on
        # this marker, so it must exist exactly when --normalize ran.
        norm_record = {"center": [float(x) for x in center], "scale": float(mean_scale)}
        (data_dir / "processed").mkdir(exist_ok=True)
        with open(data_dir / "processed" / "scene_normalization.json", "w") as f:
            json.dump(norm_record, f, indent=2)
        print(f"Normalized scene: center={center}, scale={mean_scale:.4f}")
    else:
        print("Skipping scene normalization (use --normalize for in-the-wild sequences)")

    # Update processed/ with (possibly normalized) cameras and point cloud.
    # This overwrites the step-5 c2ws.npy so that all downstream consumers
    # (step 10 object poses, training dataset loader) use the same frame.
    processed_dir = data_dir / "processed"
    c2ws_arr = np.array([c2ws[k] for k in sorted(c2ws.keys())], dtype=np.float32)
    np.save(processed_dir / "c2ws.npy", c2ws_arr)
    np.save(processed_dir / "intrinsic.npy", intrinsic)
    o3d.io.write_point_cloud(str(processed_dir / "scene_pcd.ply"), scene_pcd_legacy)
    print(f"Scene cameras written to {processed_dir}")

    if args.interactive:
        visualize_smplx_and_scene(
            smplx_server, [scene_pcd_legacy], new_smplxs, sphere_radius=3.0
        )

    # Write aligned SMPL-X to processed/
    output_dir = data_dir / "processed"
    output_dir.mkdir(exist_ok=True)

    out_name = "aligned_smplx.pkl"

    with open(output_dir / out_name, "wb") as f:
        pkl.dump(new_smplxs, f)

    print(f"Aligned SMPL-X saved to {output_dir / out_name}")
    print(f"  Scale: {mean_scale}")
    print(f"  Frames: {num_frames}")


if __name__ == "__main__":
    import tyro
    main(tyro.cli(AlignSMPLXArgs))
