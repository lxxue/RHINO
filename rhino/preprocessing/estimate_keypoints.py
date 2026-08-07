"""Step 7: Whole-body keypoint estimation using Sapiens."""

from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import sys

# Ensure rhino package is importable when invoked with external Python interpreter
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


from rhino.preprocessing.tool_config import ToolConfig, require_tools


@dataclass
class EstimateKeypointsArgs:
    data_dir: Path
    """Path to the sequence data directory."""
    det_config: str = ""
    """RTMDet config file path (relative to sapiens root). Auto-detected if empty."""
    det_checkpoint: str = ""
    """RTMDet checkpoint path. Auto-detected if empty."""
    pose_config: str = ""
    """Sapiens pose config file path. Auto-detected if empty."""
    pose_checkpoint: str = ""
    """Sapiens pose checkpoint path. Auto-detected if empty."""
    tools: ToolConfig = field(default_factory=ToolConfig)


def main(args: EstimateKeypointsArgs) -> None:
    require_tools(args.tools, "sapiens", "aios")  # AiOS needed for detrsmpl keypoint conversion
    data_dir = args.data_dir.resolve()

    img_dir = data_dir / "images" / "selected_frames"
    output_dir = data_dir / "keypoints" / "sapiens"
    output_dir.mkdir(parents=True, exist_ok=True)

    sapiens_root = args.tools.sapiens_root
    sapiens_pose_dir = sapiens_root / "pose"

    # Auto-detect configs and checkpoints if not specified
    det_config = args.det_config
    det_checkpoint = args.det_checkpoint
    pose_config = args.pose_config
    pose_checkpoint = args.pose_checkpoint

    ckpt_root = args.tools.sapiens_checkpoint_root or sapiens_root

    if not det_config:
        # Expected: pose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person_no_nms.py
        mmdet_cfg_dir = sapiens_pose_dir / "demo" / "mmdetection_cfg"
        candidates = list(mmdet_cfg_dir.glob("rtmdet_m*no_nms*.py"))
        if not candidates:
            candidates = list(sapiens_root.rglob("rtmdet_m*no_nms*.py"))
        if not candidates:
            candidates = list(sapiens_root.rglob("rtmdet_*.py"))
        if candidates:
            det_config = str(candidates[0])
        else:
            raise RuntimeError("Cannot auto-detect RTMDet config. Specify --det-config.")
    if not det_checkpoint:
        candidates = list(ckpt_root.rglob("rtmdet*.pth"))
        if not candidates:
            candidates = list(sapiens_root.rglob("rtmdet*.pth"))
        if candidates:
            det_checkpoint = str(candidates[0])
        else:
            raise RuntimeError("Cannot auto-detect RTMDet checkpoint. Specify --det-checkpoint.")
    if not pose_config:
        candidates = list((sapiens_pose_dir / "configs").rglob("sapiens_2b*wholebody*.py"))
        if not candidates:
            candidates = list((sapiens_pose_dir / "configs").rglob("sapiens*wholebody*.py"))
        if candidates:
            pose_config = str(candidates[0])
        else:
            raise RuntimeError("Cannot auto-detect Sapiens pose config. Specify --pose-config.")
    if not pose_checkpoint:
        candidates = list(ckpt_root.rglob("sapiens*wholebody*.pth")) + \
                     list(ckpt_root.rglob("sapiens*wholebody*.pt2"))
        if not candidates:
            candidates = list(sapiens_root.rglob("sapiens*wholebody*.pth")) + \
                         list(sapiens_root.rglob("sapiens*wholebody*.pt2"))
        if candidates:
            pose_checkpoint = str(candidates[0])
        else:
            raise RuntimeError("Cannot auto-detect Sapiens pose checkpoint. Specify --pose-checkpoint.")

    # Run Sapiens demo_vis.py
    cmd = [
        args.tools.sapiens_python,
        str(sapiens_pose_dir / "demo" / "demo_vis.py"),
        det_config,
        det_checkpoint,
        pose_config,
        pose_checkpoint,
        "--input", str(img_dir),
        "--output-root", str(output_dir),
        "--save-predictions",
    ]
    print(f"Running Sapiens keypoint estimation...")
    subprocess.run(cmd, check=True)

    # Verify Sapiens produced JSON outputs
    json_files = sorted(output_dir.glob("*.json"))
    if not json_files:
        raise RuntimeError(
            f"No JSON output files found in {output_dir}. "
            f"Sapiens demo_vis.py may have failed or produced no detections."
        )
    print(f"Sapiens produced {len(json_files)} JSON keypoint files.")

    # Convert coco_wholebody → smplx using AiOS env (has detrsmpl)
    # This runs as a separate subprocess because the sapiens env lacks detrsmpl
    smplx_dir = data_dir / "keypoints" / "sapiens_smplx"
    convert_script = Path(__file__).parent / "convert_keypoints_to_smplx.py"
    convert_cmd = [
        args.tools.aios_python,
        str(convert_script),
        "--sapiens-dir", str(output_dir),
        "--output-dir", str(smplx_dir),
    ]
    print(f"Converting keypoints to SMPL-X format (using AiOS env)...")
    subprocess.run(convert_cmd, check=True)
    print(f"Keypoints saved to {smplx_dir}")


if __name__ == "__main__":
    import tyro
    main(tyro.cli(EstimateKeypointsArgs))
