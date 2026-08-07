"""Step 13: Contact prediction refinement with temporal smoothing.

Refinement procedure:
  1. Mask non-hand vertices (keep only MANO hand verts) in contact frames
  2. Zero all vertices in non-contact frames
  3. In-place temporal averaging with window ±window_size, threshold 0.3
  4. Re-mask non-hand vertices after smoothing
"""

import pickle as pkl
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import trange

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class RefineContactArgs:
    data_dir: Path
    """Path to the sequence data directory."""
    window_size: int = 3
    """Temporal averaging half-window (range is [i-window_size, i+window_size])."""
    contact_threshold: float = 0.3
    """Threshold for binarizing temporally-averaged contact predictions."""


def main(args: RefineContactArgs) -> None:
    data_dir = args.data_dir.resolve()
    processed_dir = data_dir / "processed"
    print("Refining contact predictions...")

    # Load predictions and contact labels
    contact_vertices = np.load(processed_dir / "contact_vertices.npy").copy()
    contact_labels = np.load(processed_dir / "contact_labels.npy")
    assert contact_vertices.shape[0] == contact_labels.shape[0]
    num_frames = contact_vertices.shape[0]

    # Load MANO hand vertex IDs
    hand_ids_path = _REPO_ROOT / "rhino" / "checkpoints" / "smplx" / "MANO_SMPLX_vertex_ids.pkl"
    with open(hand_ids_path, "rb") as f:
        hand_ids_dict = pkl.load(f)
    hand_ids = np.concatenate([hand_ids_dict["left_hand"], hand_ids_dict["right_hand"]])
    non_hand_ids = np.setdiff1d(np.arange(10475), hand_ids)

    # Pre-pass: mask non-hand vertices in contact frames, zero non-contact frames
    for i in range(num_frames):
        if contact_labels[i]:
            contact_vertices[i, non_hand_ids] = False
        else:
            contact_vertices[i, :] = False

    # In-place causal temporal smoothing
    ws = args.window_size
    for i in trange(num_frames, desc="Refining contacts"):
        if not contact_labels[i]:
            contact_vertices[i, :] = False
        else:
            neighbors = contact_vertices[max(0, i - ws):min(num_frames, i + ws + 1)]
            contact_vertices[i] = neighbors.mean(axis=0) > args.contact_threshold
            contact_vertices[i, non_hand_ids] = False

    np.save(processed_dir / "contact_vertices_refined.npy", contact_vertices)
    print(f"Refined contact vertices: {processed_dir / 'contact_vertices_refined.npy'}")
    print(f"  Shape: {contact_vertices.shape}")
    print(f"  Frames with refined contact: {contact_vertices.any(axis=1).sum()}/{num_frames}")


if __name__ == "__main__":
    import tyro
    main(tyro.cli(RefineContactArgs))
