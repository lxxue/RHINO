"""Hybrid contact-frame heuristic: hand-object proximity + object motion.

Used by step 11 (detect_contact_frames.py). Parameters were tuned against
volumetric-capture GT across the 7 BenchRHINO sequences.

Signals per frame i:
  d[i]      min over hands of the P-th percentile of SMPL-X hand-vertex
            NN distances to the object point cloud, both in the object
            canonical frame (X_can = object_poses_centered[i] @ X_world)
  motion[i] object movement between frames i-1 and i above the fixed motion
            thresholds (translation / rotation / combined score)

Rule:
  seed[i] = (d[i] < tau_on) OR (motion[i] AND d[i] < tau_off)
  seed spans grow outward while d < tau_off (hysteresis), non-contact gaps
  shorter than gap_fill are filled, spans shorter than min_span are dropped.
"""

import pickle
from pathlib import Path

import numpy as np

# Motion thresholds (fixed; not part of the tuned grid)
TRANS_THRESH = 0.02
ROT_THRESH = 0.6
COMBINED_THRESH = 0.7

_REPO_ROOT = Path(__file__).resolve().parents[2]


def to_spans(labels: np.ndarray) -> list:
    """Bool array -> list of inclusive (start, end) spans."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], labels.astype(int), [0]])))
    return [(int(s), int(e - 1)) for s, e in edges.reshape(-1, 2)]


def motion_flags(rel_trans: np.ndarray, rel_rot: np.ndarray, median_window: int = 1) -> np.ndarray:
    """Per-frame object-motion flags from relative translation/rotation magnitudes."""
    if median_window > 1:
        from scipy.ndimage import median_filter

        rel_trans = median_filter(rel_trans, size=median_window, mode="nearest")
        rel_rot = median_filter(rel_rot, size=median_window, mode="nearest")
    return (
        (rel_trans > TRANS_THRESH)
        | (rel_rot > ROT_THRESH)
        | (rel_trans * 20 + rel_rot > COMBINED_THRESH)
    )


def relative_object_motion(object_poses: np.ndarray) -> tuple:
    """Frame-to-frame object translation (m) and rotation (deg) magnitudes.

    object_poses maps world -> object canonical; its inverse's translation is
    the object position in the world. Index 0 is 0.
    """
    poses_inv = np.linalg.inv(object_poses.astype(np.float64))
    num_frames = len(poses_inv)
    rel_trans = np.zeros(num_frames, dtype=np.float32)
    rel_rot = np.zeros(num_frames, dtype=np.float32)
    for i in range(1, num_frames):
        rel_trans[i] = np.linalg.norm(poses_inv[i, :3, 3] - poses_inv[i - 1, :3, 3])
        rel = poses_inv[i, :3, :3] @ poses_inv[i - 1, :3, :3].T
        tr = np.clip((np.trace(rel) - 1) / 2, -1, 1)
        rel_rot[i] = np.rad2deg(np.arccos(tr))
    return rel_trans, rel_rot


def hybrid_labels(
    d: np.ndarray,
    rel_trans: np.ndarray,
    rel_rot: np.ndarray,
    tau_on: float = 0.02,
    tau_off: float = 0.08,
    median_window: int = 1,
    gap_fill: int = 5,
    min_span: int = 15,
) -> np.ndarray:
    """Apply the hybrid proximity+motion rule with hysteresis to the signals."""
    motion = motion_flags(rel_trans, rel_rot, median_window)
    near_off = d < tau_off
    labels = (d < tau_on) | (motion & near_off)

    n = len(labels)
    for s, e in to_spans(labels):
        i = s - 1
        while i >= 0 and near_off[i] and not labels[i]:
            labels[i] = True
            i -= 1
        i = e + 1
        while i < n and near_off[i] and not labels[i]:
            labels[i] = True
            i += 1

    spans = to_spans(labels)
    for (s1, e1), (s2, e2) in zip(spans[:-1], spans[1:]):
        if s2 - e1 - 1 < gap_fill:
            labels[e1 + 1:s2] = True

    for s, e in to_spans(labels):
        if e - s + 1 < min_span:
            labels[s:e + 1] = False

    return labels


def hand_object_distances(
    processed_dir: Path,
    gender: str,
    percentiles: list,
    device: str = "cuda",
) -> np.ndarray:
    """Per-frame per-hand distance percentiles to the object point cloud.

    Returns (num_frames, 2, len(percentiles)) float32: for each hand, the given
    percentiles of its 778 vertex NN distances in the object canonical frame.
    """
    import open3d as o3d
    import torch
    from scipy.spatial import cKDTree

    from rhino.rhino_smplx.smplx_server import SMPLXServer

    with open(processed_dir / "aligned_smplx.pkl", "rb") as f:
        smplx_params = pickle.load(f)
    centered_poses = np.load(processed_dir / "object_poses_centered.npy").astype(np.float64)
    pcd = o3d.io.read_point_cloud(str(processed_dir / "object_pcd_centered.ply"))
    kdtree = cKDTree(np.asarray(pcd.points))

    with open(_REPO_ROOT / "rhino" / "checkpoints" / "smplx" / "MANO_SMPLX_vertex_ids.pkl", "rb") as f:
        hand_ids = pickle.load(f)
    left_ids, right_ids = hand_ids["left_hand"], hand_ids["right_hand"]

    num_frames = len(smplx_params["global_orient"])
    torch_device = torch.device(device)
    # Construct on CPU: SMPLXServer runs a canonical forward pass in its
    # constructor, and the body-model buffers only reach the target device
    # via the .to() afterwards.
    betas = torch.tensor(np.asarray(smplx_params["betas"]), dtype=torch.float32)[None]
    server = SMPLXServer(gender=gender, betas=betas).to(torch_device)
    betas = betas.to(torch_device)

    def t(key, i):
        return torch.tensor(
            np.asarray(smplx_params[key][i]), dtype=torch.float32, device=torch_device
        )[None]

    dist_pcts = np.zeros((num_frames, 2, len(percentiles)), dtype=np.float32)
    with torch.no_grad():
        for i in range(num_frames):
            out = server.forward(
                scale=torch.tensor(
                    float(np.asarray(smplx_params["scale"]).reshape(-1)[i]), device=torch_device
                ),
                betas=betas,
                transl=t("transl", i),
                global_orient=t("global_orient", i),
                body_pose=t("body_pose", i),
                left_hand_pose=t("left_hand_pose", i),
                right_hand_pose=t("right_hand_pose", i),
                expression=t("expression", i),
                jaw_pose=t("jaw_pose", i),
                leye_pose=torch.zeros((1, 3), device=torch_device),
                reye_pose=torch.zeros((1, 3), device=torch_device),
            )
            verts_w = out["smplx_verts"][0].cpu().numpy().astype(np.float64)
            hom = np.concatenate([verts_w, np.ones((verts_w.shape[0], 1))], axis=1)
            verts_can = (centered_poses[i] @ hom.T).T[:, :3]
            for h, ids in enumerate([left_ids, right_ids]):
                dists, _ = kdtree.query(verts_can[ids])
                dist_pcts[i, h] = np.percentile(dists, percentiles)
    return dist_pcts
