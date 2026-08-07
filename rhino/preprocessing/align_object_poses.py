"""Step 10: Object pose alignment + centering.

Aligns object SfM reconstruction to scene frame using RANSAC, computes object poses as obj_c2w_aligned @ inv(scene_c2w),
interpolates missing poses with SLERP, and centers the canonical space.

Scene cameras are read from data_dir/processed/ (written by step 9).
Object cameras are read from the object COLMAP sparse reconstruction.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R, Slerp

from rhino.preprocessing.align_utils import (
    read_extrinsics_binary,
    read_point_cloud_binary,
)
from rhino.preprocessing.hoi_utils import ransac


@dataclass
class AlignObjectPosesArgs:
    data_dir: Path
    """Path to the sequence data directory."""
    object_camera_pose_dir: Path
    """Path to object COLMAP sparse reconstruction."""
    ransac_threshold: float = 0.03
    """RANSAC inlier threshold."""
    ransac_max_iter: int = 10000
    """Maximum RANSAC iterations."""


def _slerp_poses(obj_c2ws, keyframe_idxs, num_frames, suffix):
    """Interpolate full trajectory from sparse keyframes using SLERP + linear translation.

    Pads endpoints, SLERPs rotations, linearly interpolates translations.
    """
    poses_np = np.array([obj_c2ws[f"{i:04d}.{suffix}"] for i in keyframe_idxs])
    all_frame_idxs = np.arange(num_frames)

    kf = list(keyframe_idxs)
    if kf[0] > 0:
        kf = [0] + kf
        poses_np = np.concatenate([poses_np[0:1], poses_np], axis=0)
    if kf[-1] < num_frames - 1:
        kf = kf + [num_frames - 1]
        poses_np = np.concatenate([poses_np, poses_np[-1:]], axis=0)

    key_rots = R.from_matrix(poses_np[:, :3, :3])
    key_trans = poses_np[:, :3, 3]
    slerp = Slerp(kf, key_rots)
    interp_rots = slerp(all_frame_idxs).as_matrix()

    interp_trans = np.zeros((num_frames, 3))
    for dim in range(3):
        interp_trans[:, dim] = np.interp(all_frame_idxs, kf, key_trans[:, dim])

    result = np.zeros((num_frames, 4, 4))
    result[:, :3, :3] = interp_rots
    result[:, :3, 3] = interp_trans
    result[:, 3, 3] = 1
    return result


def compute_robust_center(pts: np.ndarray, lo: float = 5, hi: float = 95) -> np.ndarray:
    """Center of per-axis [lo, hi] percentile range."""
    low = np.percentile(pts, lo, axis=0)
    high = np.percentile(pts, hi, axis=0)
    return (low + high) / 2


def center_canonical(object_poses, object_pcd_points):
    """Center the object canonical space at the origin.

    Recentering: x_c_new = x_c - c => t_new = t - c. Rotation unchanged.
    """
    center = compute_robust_center(object_pcd_points)
    centered_pcd = object_pcd_points - center
    centered_poses = object_poses.copy()
    centered_poses[:, :3, 3] -= center
    return centered_poses, centered_pcd


def main(args: AlignObjectPosesArgs) -> None:
    data_dir = args.data_dir.resolve()
    print("Aligning object poses to scene frame + centering...")

    processed_dir = data_dir / "processed"

    # Read scene cameras from processed/ (written/updated by step 9)
    c2ws_path = processed_dir / "c2ws.npy"
    if not c2ws_path.exists():
        raise RuntimeError(
            f"processed/c2ws.npy not found at {processed_dir}. "
            "Run step 9 (align_smplx) first."
        )
    scene_c2ws_np = np.load(c2ws_path)
    num_frames = scene_c2ws_np.shape[0]

    # Read object cameras from COLMAP
    obj_c2ws = read_extrinsics_binary(args.object_camera_pose_dir.resolve())

    # Read object point cloud
    obj_pcd = read_point_cloud_binary(args.object_camera_pose_dir.resolve())

    # Read scene point cloud from processed/ if available
    scene_pcd_path = processed_dir / "scene_pcd.ply"
    if scene_pcd_path.exists():
        scene_pcd_legacy = o3d.io.read_point_cloud(str(scene_pcd_path))
    else:
        scene_pcd_legacy = None

    # Build object c2w arrays — SLERP-interpolate if some frames are missing
    obj_keys = sorted(obj_c2ws.keys())
    obj_suffix = obj_keys[0].split(".")[-1]
    num_valid = len(obj_c2ws)

    keyframe_idxs = []
    for i in range(num_frames):
        key = f"{i:04d}.{obj_suffix}"
        if key in obj_c2ws:
            keyframe_idxs.append(i)

    if len(keyframe_idxs) < 4:
        raise RuntimeError(f"Only {len(keyframe_idxs)} object frames registered. Need at least 4.")

    if num_valid == num_frames:
        # All frames registered — build arrays directly
        obj_c2ws_all = np.array(
            [obj_c2ws[f"{i:04d}.{obj_suffix}"] for i in range(num_frames)]
        )
    else:
        # Interpolate missing frames (SLERP rotations, lerp translations)
        print(f"Object: {num_valid}/{num_frames} frames registered, interpolating missing")
        obj_c2ws_all = _slerp_poses(obj_c2ws, keyframe_idxs, num_frames, obj_suffix)

    # RANSAC: object camera centers → scene camera centers (all frames)
    obj_cam_centers = obj_c2ws_all[:, :3, 3]
    scene_cam_centers = scene_c2ws_np[:, :3, 3]

    rot, scale, trans, inliers = ransac(
        obj_cam_centers, scene_cam_centers,
        threshold=args.ransac_threshold,
        max_iter=args.ransac_max_iter,
    )
    print(f"RANSAC: {len(inliers)}/{num_frames} inliers, scale={scale:.4f}")

    # Align object c2ws to scene frame:
    #   obj_c2ws_aligned[i][:3,:3] = R @ obj_c2ws[i][:3,:3]
    #   obj_c2ws_aligned[i][:3, 3] = s * R @ obj_c2ws[i][:3, 3] + t
    # Use the original COLMAP c2ws where available; SLERP-interpolated poses
    # fill the gaps
    obj_c2ws_aligned = np.zeros((num_frames, 4, 4), dtype=np.float64)
    object_poses = np.zeros((num_frames, 4, 4), dtype=np.float64)
    valid_mask = np.zeros(num_frames, dtype=bool)

    for i in range(num_frames):
        key = f"{i:04d}.{obj_suffix}"
        if key in obj_c2ws:
            src = obj_c2ws[key]
        else:
            src = obj_c2ws_all[i]

        obj_c2ws_aligned[i] = np.eye(4)
        obj_c2ws_aligned[i][:3, :3] = rot @ src[:3, :3]
        obj_c2ws_aligned[i][:3, 3] = scale * rot @ src[:3, 3] + trans

        object_poses[i] = obj_c2ws_aligned[i] @ np.linalg.inv(scene_c2ws_np[i])
        if key in obj_c2ws:
            valid_mask[i] = True

    print(f"Object poses: {valid_mask.sum()}/{num_frames} valid")

    # Save uncentered poses
    np.save(processed_dir / "object_poses.npy", object_poses.astype(np.float32))

    # Transform and save point clouds
    rigid_matrix = np.eye(4)
    rigid_matrix[:3, :3] = rot
    rigid_matrix[:3, 3] = trans
    s_matrix = np.eye(4)
    s_matrix[:3, :3] *= scale
    whole_transform = rigid_matrix @ s_matrix

    obj_pcd_legacy = obj_pcd.to_legacy() if hasattr(obj_pcd, "to_legacy") else obj_pcd
    obj_pcd_transformed = obj_pcd_legacy.transform(whole_transform)
    o3d.io.write_point_cloud(str(processed_dir / "object_pcd.ply"), obj_pcd_transformed)
    if scene_pcd_legacy is not None:
        o3d.io.write_point_cloud(str(processed_dir / "scene_pcd.ply"), scene_pcd_legacy)

    # Center canonical space
    obj_points = np.asarray(obj_pcd_transformed.points)
    centered_poses, centered_pcd_points = center_canonical(object_poses, obj_points)
    np.save(processed_dir / "object_poses_centered.npy", centered_poses.astype(np.float32))

    centered_pcd = o3d.geometry.PointCloud()
    centered_pcd.points = o3d.utility.Vector3dVector(centered_pcd_points)
    o3d.io.write_point_cloud(str(processed_dir / "object_pcd_centered.ply"), centered_pcd)

    print(f"Object poses saved to {processed_dir}")
    print(f"  object_poses.npy: {object_poses.shape}")
    print(f"  object_poses_centered.npy: {centered_poses.shape}")


if __name__ == "__main__":
    import tyro
    main(tyro.cli(AlignObjectPosesArgs))
