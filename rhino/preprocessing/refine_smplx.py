"""Step 8: SMPL-X refinement using Sapiens keypoint constraints.

Two-stage per-frame optimization:
1. Pre-smooth poses (body + hands + jaw) with Savitzky-Golay
2. Per frame: opt_transl (200 iters) → opt_full (300 iters)
3. Average per-frame betas → mean_shape.txt
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from rhino.preprocessing.smplx_utils import GMoF
from rhino.preprocessing.tool_config import ToolConfig

# AiOS uses focal_length=5000 internally for its camera model.
# Translation Z must be rescaled: transl_z *= actual_focal / 5000.
_AIOS_FOCAL_LENGTH = 5000.0


@dataclass
class RefineSMPLXArgs:
    data_dir: Path
    """Path to the sequence data directory."""
    transl_iters: int = 200
    """Iterations for translation-only optimization stage."""
    full_iters: int = 300
    """Iterations for full parameter optimization stage."""
    temporal_weight: float = 20.0
    """Weight for temporal smoothness loss."""
    j2d_weight: float = 1e-2
    """Weight for 2D keypoint reprojection loss."""
    smooth_window: int = 21
    """Savitzky-Golay window for pre-smoothing poses."""
    max_kpt_idx: int = 76
    """Max keypoint index to use (76 = body+hands, discards face contour)."""
    smplx_model_path: str = ""
    """Path to SMPL-X model directory. Auto-detected from ToolConfig if empty."""
    tools: ToolConfig = field(default_factory=ToolConfig)


def fix_quaternions(quats):
    """Enforce quaternion continuity across time dimension."""
    result = quats.copy()
    dot_products = np.sum(quats[1:] * quats[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
    result[1:][mask] *= -1
    return result


def smoothen_poses(poses, window_length=21, polyorder=2):
    """Smooth joint rotation vectors via quaternion SLERP + Savitzky-Golay."""
    n_joints = poses.shape[1] // 3
    if poses.shape[0] < window_length:
        return poses

    qs = Rotation.as_quat(Rotation.from_rotvec(np.reshape(poses, (-1, 3))))
    qs = qs.reshape((-1, n_joints, 4))
    qs = fix_quaternions(qs)

    qs_smooth = []
    for j in range(n_joints):
        qss = savgol_filter(qs[:, j], window_length=window_length, polyorder=polyorder, axis=0)
        qs_smooth.append(qss[:, np.newaxis])
    qs_clean = np.concatenate(qs_smooth, axis=1)
    qs_clean = qs_clean / np.linalg.norm(qs_clean, axis=-1, keepdims=True)

    ps_clean = Rotation.as_rotvec(Rotation.from_quat(np.reshape(qs_clean, (-1, 4))))
    ps_clean = np.reshape(ps_clean, [-1, n_joints * 3])
    return ps_clean


def _project_joints(joints_3d, fx, fy, cx, cy):
    """Project 3D joints to 2D using pinhole camera model."""
    x = joints_3d[:, 0] / joints_3d[:, 2] * fx + cx
    y = joints_3d[:, 1] / joints_3d[:, 2] * fy + cy
    return torch.stack([x, y], dim=-1)


def _j2d_loss(robustifier, proj_2d, target_2d, conf, selected_inds):
    """2D keypoint loss with confidence^2 weighting."""
    diff = robustifier(target_2d[selected_inds, :2] - proj_2d[selected_inds])
    w = conf[selected_inds].unsqueeze(-1) ** 2
    return torch.mean(w * diff)


def _smplx_forward(smplx_fn, betas, global_orient, body_pose,
                   left_hand_pose, right_hand_pose, transl,
                   expression, jaw_pose):
    """Run SMPL-X forward pass with all params."""
    return smplx_fn(
        betas=betas[None], global_orient=global_orient[None],
        body_pose=body_pose[None], left_hand_pose=left_hand_pose[None],
        right_hand_pose=right_hand_pose[None], transl=transl[None],
        expression=expression[None], jaw_pose=jaw_pose[None],
    )


def optimize_transl(smplx_fn, fx, fy, cx, cy, target_2d, conf, selected_inds,
                    shape, global_orient, body_pose, left_hand_pose,
                    right_hand_pose, transl, expression, jaw_pose,
                    last_pose, num_iters, j2d_weight, temporal_weight,
                    robustifier):
    """Stage 1: Optimize translation only to correct AiOS depth."""
    opt_transl = transl.clone().detach().cuda().requires_grad_(True)
    optimizer = torch.optim.Adam([{"params": opt_transl, "lr": 5e-2}],
                                 lr=1e-3, betas=(0.9, 0.999))

    betas_g = shape.cuda()
    go_g = global_orient.cuda()
    bp_g = body_pose.cuda()
    lh_g = left_hand_pose.cuda()
    rh_g = right_hand_pose.cuda()
    expr_g = expression.cuda()
    jp_g = jaw_pose.cuda()

    for _ in range(num_iters):
        optimizer.zero_grad()
        output = _smplx_forward(smplx_fn, betas_g, go_g, bp_g, lh_g, rh_g,
                                opt_transl, expr_g, jp_g)
        proj_2d = _project_joints(output.joints[0], fx, fy, cx, cy)
        loss = j2d_weight * _j2d_loss(robustifier, proj_2d, target_2d, conf, selected_inds)

        if last_pose is not None:
            loss += temporal_weight * torch.mean(
                (opt_transl - last_pose["transl"].cuda()) ** 2
            )

        loss.backward()
        optimizer.step()

    return opt_transl.detach()


def optimize_full(smplx_fn, fx, fy, cx, cy, target_2d, conf, selected_inds,
                  shape, global_orient, body_pose, left_hand_pose,
                  right_hand_pose, transl, expression, jaw_pose,
                  last_pose, num_iters, j2d_weight, temporal_weight,
                  robustifier):
    """Stage 2: Optimize all parameters (betas, poses, translation)."""
    opt_betas = shape.clone().detach().cuda().requires_grad_(True)
    opt_go = global_orient.clone().detach().cuda().requires_grad_(True)
    opt_bp = body_pose.clone().detach().cuda().requires_grad_(True)
    opt_lh = left_hand_pose.clone().detach().cuda().requires_grad_(True)
    opt_rh = right_hand_pose.clone().detach().cuda().requires_grad_(True)
    opt_transl = transl.clone().detach().cuda().requires_grad_(True)
    expr_g = expression.cuda()
    jp_g = jaw_pose.cuda()

    optimizer = torch.optim.Adam([
        {"params": opt_betas, "lr": 1e-3},
        {"params": opt_go, "lr": 1e-3},
        {"params": opt_bp, "lr": 1e-3},
        {"params": opt_lh, "lr": 1e-3},
        {"params": opt_rh, "lr": 1e-3},
        {"params": opt_transl, "lr": 1e-4},
    ], lr=1e-3, betas=(0.9, 0.999))

    for _ in range(num_iters):
        optimizer.zero_grad()
        output = _smplx_forward(smplx_fn, opt_betas, opt_go, opt_bp, opt_lh,
                                opt_rh, opt_transl, expr_g, jp_g)
        proj_2d = _project_joints(output.joints[0], fx, fy, cx, cy)
        loss = j2d_weight * _j2d_loss(robustifier, proj_2d, target_2d, conf, selected_inds)

        if last_pose is not None:
            for key, opt_val in [
                ("betas", opt_betas), ("global_orient", opt_go),
                ("body_pose", opt_bp), ("left_hand_pose", opt_lh),
                ("right_hand_pose", opt_rh), ("transl", opt_transl),
            ]:
                loss += temporal_weight * torch.mean(
                    (opt_val - last_pose[key].cuda()) ** 2
                )

        loss.backward()
        optimizer.step()

    return {
        "betas": opt_betas.detach().cpu(),
        "global_orient": opt_go.detach().cpu(),
        "body_pose": opt_bp.detach().cpu(),
        "left_hand_pose": opt_lh.detach().cpu(),
        "right_hand_pose": opt_rh.detach().cpu(),
        "transl": opt_transl.detach().cpu(),
    }


def main(args: RefineSMPLXArgs) -> None:
    data_dir = args.data_dir.resolve()
    print("Refining AiOS SMPL-X estimates with keypoint constraints...")

    # Load intrinsics
    intrinsic_path = data_dir / "processed" / "intrinsic.npy"
    if not intrinsic_path.exists():
        raise RuntimeError(f"Intrinsic not found at {intrinsic_path}. Run prepare_base first.")
    intrinsic = np.load(intrinsic_path)
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    focal_length = fx  # Use x focal length for AiOS rescaling

    # Load SMPL-X params
    smplx_dir = data_dir / "smplx" / "aios"
    if not smplx_dir.exists():
        raise RuntimeError(f"SMPL-X estimates not found at {smplx_dir}")
    param_files = sorted((smplx_dir / "smplx_params").glob("*.txt"))
    if not param_files:
        raise RuntimeError(f"No SMPL-X parameter files found in {smplx_dir / 'smplx_params'}")
    num_frames = len(param_files)

    all_params = []
    for pf in param_files:
        all_params.append(np.loadtxt(pf).astype(np.float32))
    all_params = np.array(all_params)

    # Load per-frame shapes from AiOS output
    shape_dir = smplx_dir / "selected_frames" / "params"
    all_shapes = []
    for i in range(num_frames):
        shape_file = shape_dir / f"shape-{i:06d}.txt"
        if shape_file.exists():
            shape = np.loadtxt(shape_file).astype(np.float32)
        else:
            betas_path = smplx_dir / "mean_shape.txt"
            shape = np.loadtxt(betas_path).astype(np.float32) if betas_path.exists() else np.zeros(10, dtype=np.float32)
        if len(shape) < 300:
            shape = np.pad(shape, (0, 300 - len(shape)), mode='constant')
        all_shapes.append(shape)

    # Pre-smooth poses (body + hands + jaw) before optimization
    poses_combined = np.column_stack([
        all_params[:, 3:66],     # body_pose (63)
        all_params[:, 66:111],   # left_hand_pose (45)
        all_params[:, 111:156],  # right_hand_pose (45)
        all_params[:, 169:172],  # jaw_pose (3)
    ])
    smoothed_combined = smoothen_poses(poses_combined, window_length=args.smooth_window)

    # Load converted SMPL-X keypoints (from estimate_keypoints)
    keypoint_dir = data_dir / "keypoints" / "sapiens_smplx"
    if not keypoint_dir.exists():
        raise RuntimeError(
            f"SMPL-X keypoints not found at {keypoint_dir}. "
            f"Run estimate_keypoints (step 7) first."
        )
    kpt_files = sorted(f for f in keypoint_dir.glob("*.npy") if f.name != "mask.npy")
    if len(kpt_files) < num_frames:
        raise RuntimeError(
            f"Insufficient keypoint files: {len(kpt_files)} found for {num_frames} frames."
        )

    # Load mask, limit to first max_kpt_idx joints (body+hands, no face contour)
    mask_path = keypoint_dir / "mask.npy"
    keypoints_mask = np.load(mask_path) if mask_path.exists() else np.ones(args.max_kpt_idx)
    selected_inds = torch.where(
        torch.tensor(keypoints_mask[:args.max_kpt_idx])
    )[0].cuda()

    # Create SMPL-X model
    try:
        from smplx.body_models import SMPLX as SMPLXModel
    except ImportError:
        raise RuntimeError(
            "smplx package required for SMPL-X refinement. "
            "Install it: pip install smplx==0.1.28"
        )
    model_path = args.smplx_model_path or str(args.tools.smplx_model_dir)
    smplx_fn = SMPLXModel(model_path, gender="neutral", use_pca=False,
                          num_pca_comps=45, num_betas=300,
                          flat_hand_mean=False, use_face_contour=True,
                          batch_size=1).cuda()
    robustifier = GMoF(rho=100).cuda()

    print(f"Focal length: {focal_length:.2f} (AiOS Z scale: {focal_length / _AIOS_FOCAL_LENGTH:.4f})")
    print(f"Running two-stage optimization "
          f"({args.transl_iters} transl + {args.full_iters} full iters/frame)...")

    last_pose = None
    refined_shapes = []
    optimized_params = []

    for i in tqdm(range(num_frames)):
        params_i = all_params[i]
        shape_i = torch.from_numpy(all_shapes[i]).float()

        # Raw global_orient from AiOS
        global_orient = torch.from_numpy(params_i[:3]).float()
        expression = torch.from_numpy(params_i[159:169]).float()

        # Rescale translation Z for AiOS (uses focal_length=5000 internally).
        transl_raw = params_i[156:159].copy()
        transl_raw[2] *= focal_length / _AIOS_FOCAL_LENGTH
        transl = torch.from_numpy(transl_raw).float()

        # Use PRE-SMOOTHED poses (body + hands + jaw)
        sm = smoothed_combined[i]
        body_pose = torch.from_numpy(sm[:63]).float()
        left_hand_pose = torch.from_numpy(sm[63:108]).float()
        right_hand_pose = torch.from_numpy(sm[108:153]).float()
        jaw_pose = torch.from_numpy(sm[153:156]).float()

        # Load sapiens keypoints and process confidence
        kpts = np.load(kpt_files[i])
        kpts_2d = torch.from_numpy(kpts).float().cuda()
        conf = kpts_2d[:, 2].clone() if kpts_2d.shape[-1] == 3 else torch.ones(kpts_2d.shape[0]).cuda()
        conf[conf < 0.5] = 0.0
        conf = torch.pow(conf, 8)

        # Stage 1: Optimize translation only (anchors body at correct depth)
        opt_transl = optimize_transl(
            smplx_fn, fx, fy, cx, cy, kpts_2d, conf, selected_inds,
            shape_i, global_orient, body_pose, left_hand_pose, right_hand_pose,
            transl, expression, jaw_pose,
            last_pose, args.transl_iters, args.j2d_weight, args.temporal_weight,
            robustifier,
        )

        # Stage 2: Optimize full (betas, poses, translation)
        opt_result = optimize_full(
            smplx_fn, fx, fy, cx, cy, kpts_2d, conf, selected_inds,
            shape_i, global_orient, body_pose, left_hand_pose, right_hand_pose,
            opt_transl.cpu(), expression, jaw_pose,
            last_pose, args.full_iters, args.j2d_weight, args.temporal_weight,
            robustifier,
        )

        last_pose = opt_result
        refined_shapes.append(opt_result["betas"].numpy().squeeze())

        # Save optimized params (jaw_pose = smoothed version)
        refined = np.concatenate([
            opt_result["global_orient"].numpy(),
            opt_result["body_pose"].numpy(),
            opt_result["left_hand_pose"].numpy(),
            opt_result["right_hand_pose"].numpy(),
            opt_result["transl"].numpy(),
            params_i[159:169],  # expression (original)
            sm[153:156],        # jaw_pose (smoothed)
        ])
        optimized_params.append(refined)

    # Write refined params (no post-smoothing needed; pre-smoothing was applied)
    output_dir = data_dir / "smplx" / "aios_refined"
    (output_dir / "smplx_params").mkdir(parents=True, exist_ok=True)

    for i, params in enumerate(optimized_params):
        np.savetxt(output_dir / "smplx_params" / f"{i:04d}.txt", params)

    # Average refined betas across all frames
    betas = np.array(refined_shapes).mean(axis=0)
    np.savetxt(output_dir / "mean_shape.txt", betas)

    print(f"Refined {num_frames} frames → {output_dir}")


if __name__ == "__main__":
    import tyro
    main(tyro.cli(RefineSMPLXArgs))
