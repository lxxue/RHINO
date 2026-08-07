"""Step 12: InteractVLM contact vertex extraction.

Invokes InteractVLM inference (quantized to fit 24GB GPUs) on the contact
frames detected by step 11, then parses .npz output files for per-vertex
contact predictions. Non-contact frames are skipped: step 13 zeroes them
anyway, so predicting them would be wasted compute.
"""

from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import sys

# Ensure rhino package is importable when invoked with external Python interpreter
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
from tqdm import tqdm

from rhino.preprocessing.tool_config import ToolConfig, require_tools


@dataclass
class ExtractContactVerticesArgs:
    data_dir: Path
    """Path to the sequence data directory."""
    contact_type: str = "hcontact"
    """Contact type: 'hcontact' for human contact, 'ocontact' for object contact."""
    model_version: str = "xinlai/LISA-13B-llama2-v1"
    """InteractVLM trained-model identifier (HF hub id or path relative to interactvlm_root).

    Set this to the InteractVLM trained-model directory (e.g.
    ``./trained_models/interactvlm-3d-hcontact-damon`` for hcontact) when running
    4-bit inference; the raw LISA-13B base cannot load the InteractVLM head.
    The upstream default is kept to match `run_demo.py` semantics — the
    orchestrator (`run_pipeline.py`) is the right place to pin the checkpoint."""
    quant: str = "4bit"
    """Quantization mode: '4bit' (BitsAndBytes NF4), '8bit' (LLM.int8), or 'none'.

    The 13B model does not fit a 24GB GPU unquantized; 'none' requires ~30GB."""
    precision: str = "bf16"
    """Inference precision: 'bf16', 'fp16', or 'fp32'."""
    object_name: str = ""
    """Object name for the contact prompt ('Which body parts are in contact with
    the {object}?'). Empty falls back to run_demo.py's filename parsing, which
    degrades to the frame number for numbered frames — always set this."""
    num_smplx_vertices: int = 10475
    """Number of SMPL-X vertices."""
    tools: ToolConfig = field(default_factory=ToolConfig)


def stage_images_for_interactvlm(img_dir: Path, staging_dir: Path,
                                 contact_labels: np.ndarray) -> None:
    """Stage contact-frame images into InteractVLM's expected folder layout.

    InteractVLM folder mode expects: staging_dir/<name>/<name>.jpg. Only frames
    whose contact label is True are staged, and stale sample dirs from an
    earlier staging (e.g. all-frames, or different labels) are removed so
    parse_contact_outputs cannot resurrect outdated predictions.
    """
    import shutil
    staging_dir.mkdir(parents=True, exist_ok=True)
    img_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    staged = set()
    for img_path in img_paths:
        try:
            idx = int(img_path.stem)
        except ValueError:
            continue
        if idx >= len(contact_labels) or not contact_labels[idx]:
            continue
        staged.add(img_path.stem)
        sample_dir = staging_dir / img_path.stem
        sample_dir.mkdir(exist_ok=True)
        dst = sample_dir / f"{img_path.stem}.jpg"
        if not dst.exists():
            shutil.copy2(img_path, dst)
    for sub in staging_dir.iterdir():
        if sub.is_dir() and sub.name.isdigit() and sub.name not in staged:
            shutil.rmtree(sub)


def check_interactvlm_assets(interactvlm_root: Path, contact_type: str) -> None:
    """Validate InteractVLM runtime assets exist before launching inference.

    InteractVLM requires internal data files (body template renderings,
    SMPL-to-SMPL-X mapping) and trained model weights that must be downloaded
    separately via fetch_data.sh.
    """
    missing = []

    # SMPL-to-SMPL-X vertex mapping (loaded unconditionally by run_demo.py)
    smpl_to_smplx = interactvlm_root / "data" / "smpl_to_smplx.pkl"
    if not smpl_to_smplx.is_file():
        missing.append(str(smpl_to_smplx))

    # Vitruvian body template renderings (required for hcontact 3D prediction)
    if "hcontact" in contact_type:
        vitruvian_dir = interactvlm_root / "data" / "hcontact_vitruvian"
        if not vitruvian_dir.is_dir():
            missing.append(str(vitruvian_dir) + "/")

    # Trained model weights
    trained_models = interactvlm_root / "trained_models"
    if not trained_models.is_dir() or not any(trained_models.iterdir()):
        missing.append(str(trained_models) + "/")

    if missing:
        raise RuntimeError(
            f"InteractVLM runtime assets not found:\n"
            + "\n".join(f"  - {p}" for p in missing)
            + f"\n\nDownload them by running:\n"
            f"  cd {interactvlm_root} && bash fetch_data.sh\n"
            f"(Requires registration at https://interactvlm.is.tue.mpg.de)"
        )


def run_interactvlm_inference(args: ExtractContactVerticesArgs,
                              contact_labels: np.ndarray) -> Path:
    """Run InteractVLM inference on contact frames and return the staging directory."""
    require_tools(args.tools, "interactvlm")
    check_interactvlm_assets(args.tools.interactvlm_root, args.contact_type)
    data_dir = args.data_dir.resolve()
    img_dir = data_dir / "images" / "selected_frames"

    # Stage contact-frame images into InteractVLM's expected layout
    staging_dir = data_dir / "interactvlm_staging"
    print(f"Staging {int(contact_labels.sum())} contact frames for InteractVLM at {staging_dir}...")
    stage_images_for_interactvlm(img_dir, staging_dir, contact_labels)

    cmd = [
        args.tools.interactvlm_python,
        str(args.tools.interactvlm_root / "run_demo.py"),
        "--version", args.model_version,
        "--contact_type", args.contact_type,
        "--img_folder", str(staging_dir),
        "--input_mode", "folder",
        "--precision", args.precision,
    ]
    if args.quant == "4bit":
        cmd.append("--load_in_4bit")
    elif args.quant == "8bit":
        cmd.append("--load_in_8bit")
    elif args.quant != "none":
        raise ValueError(f"Unknown quant mode {args.quant!r}; expected '4bit', '8bit', or 'none'")
    if args.object_name:
        cmd.extend(["--object_name", args.object_name])

    print(f"Running InteractVLM ({args.contact_type}, {args.quant}/{args.precision}, object={args.object_name or '<filename>'})...")
    # Run from InteractVLM root so relative paths (./data/, ./trained_models/) resolve
    subprocess.run(cmd, check=True, cwd=str(args.tools.interactvlm_root))

    # InteractVLM writes .npz to each sample subdirectory
    # and .obj to staging_dir/contact_output/
    return staging_dir


def parse_contact_outputs(staging_dir: Path, contact_type: str,
                          num_frames: int, num_vertices: int,
                          contact_labels: np.ndarray) -> np.ndarray:
    """Parse InteractVLM .npz output files into per-vertex contact boolean array.

    Only contact frames (per contact_labels) are filled; all other rows stay zero.
    """
    contact_vertices = np.zeros((num_frames, num_vertices), dtype=bool)

    # In folder mode, InteractVLM saves .npz in each sample subdirectory:
    # staging_dir/<sample_name>/<sample_name>_{contact_type}_vertices.npz
    npz_files = sorted(staging_dir.rglob(f"*_{contact_type}_vertices.npz"))

    # Also check contact_output directory for .obj files
    contact_output_dir = staging_dir / "contact_output"
    obj_files = sorted(contact_output_dir.glob("*.obj")) if contact_output_dir.exists() else []

    if not npz_files and not obj_files:
        raise RuntimeError(
            f"No InteractVLM output found. Expected .npz in {staging_dir} subdirs "
            f"or .obj in {contact_output_dir}"
        )

    if npz_files:
        for npz_file in tqdm(npz_files, desc="Parsing contact .npz files"):
            # Extract frame index from filename
            stem = npz_file.stem.replace(f"_{contact_type}_vertices", "")
            try:
                idx = int(stem)
            except ValueError:
                continue
            if idx >= num_frames or not contact_labels[idx]:
                continue

            data = np.load(npz_file)
            if contact_type == "hcontact" and "pred_contact_3d_smplx" in data:
                pred = data["pred_contact_3d_smplx"]
                contact_mask = pred > 0.3
                n_verts = min(len(contact_mask), num_vertices)
                contact_vertices[idx, :n_verts] = contact_mask[:n_verts]
            elif "pred_contact_3d" in data:
                pred = data["pred_contact_3d"]
                contact_mask = pred > 0.3
                n_verts = min(len(contact_mask), num_vertices)
                contact_vertices[idx, :n_verts] = contact_mask[:n_verts]
    elif obj_files:
        # Fallback: parse colored .obj meshes
        import trimesh
        for obj_file in tqdm(obj_files, desc="Parsing contact .obj files"):
            stem = obj_file.stem.split("_")[0]
            try:
                idx = int(stem)
            except ValueError:
                continue
            if idx >= num_frames or not contact_labels[idx]:
                continue

            mesh = trimesh.load(obj_file, process=False)
            if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
                colors = mesh.visual.vertex_colors[:, :3]
                contact_mask = (colors[:, 0] > 200) & (colors[:, 1] < 50) & (colors[:, 2] < 50)
                n_verts = min(len(contact_mask), num_vertices)
                contact_vertices[idx, :n_verts] = contact_mask[:n_verts]

    return contact_vertices


def main(args: ExtractContactVerticesArgs) -> None:
    data_dir = args.data_dir.resolve()
    processed_dir = data_dir / "processed"

    num_frames = len(sorted((processed_dir / "image").glob("*.png")))
    if num_frames == 0:
        raise RuntimeError(f"No images in {processed_dir / 'image'}. Run prepare_base first.")

    contact_labels_path = processed_dir / "contact_labels.npy"
    if not contact_labels_path.exists():
        raise RuntimeError(
            f"{contact_labels_path} not found. Run step 11 (detect_contact_frames) first; "
            f"step 12 only runs InteractVLM on contact frames."
        )
    contact_labels = np.load(contact_labels_path).astype(bool)
    if len(contact_labels) != num_frames:
        raise RuntimeError(
            f"contact_labels.npy has {len(contact_labels)} frames but processed/image has {num_frames}"
        )

    staging_dir = run_interactvlm_inference(args, contact_labels)
    contact_vertices = parse_contact_outputs(
        staging_dir, args.contact_type, num_frames, args.num_smplx_vertices, contact_labels
    )

    np.save(processed_dir / "contact_vertices.npy", contact_vertices)
    print(f"Contact vertices saved: {processed_dir / 'contact_vertices.npy'}")
    print(f"  Shape: {contact_vertices.shape}")
    print(f"  Frames with contact: {(contact_vertices.any(axis=1)).sum()}/{num_frames}")


if __name__ == "__main__":
    import tyro
    main(tyro.cli(ExtractContactVerticesArgs))
