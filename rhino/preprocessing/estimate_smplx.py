"""Step 6: SMPL-X body estimation using AiOS."""

from dataclasses import dataclass, field
from pathlib import Path
import re
import shutil
import subprocess
import sys

# Ensure rhino package is importable when invoked with external Python interpreter
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from rhino.preprocessing.tool_config import ToolConfig, require_tools


@dataclass
class EstimateSMPLXArgs:
    data_dir: Path
    """Path to the sequence data directory."""
    threshold: float = 0.3
    """Detection confidence threshold (AiOS default: 0.3)."""
    num_person: int = 1
    """Maximum number of persons to detect per frame (AiOS)."""
    checkpoint: str = "data/checkpoint/aios_checkpoint.pth"
    """Path to AiOS checkpoint (relative to AiOS root)."""
    tools: ToolConfig = field(default_factory=ToolConfig)


def normalize_aios_outputs(output_dir: Path, num_frames: int | None = None) -> None:
    """Normalize AiOS demo outputs into RHINO's expected layout.

    AiOS demo writes: output_dir/<img_dir_name>/params/pose-{id:06d}.txt + shape-{id:06d}.txt
    RHINO expects:    output_dir/smplx_params/{idx:04d}.txt + mean_shape.txt

    The pose-file id is the input frame index. Frames where AiOS detected no
    person have no pose file; those must NOT be silently skipped (that would
    shift every later frame against its image/keypoints), so they are filled
    from the nearest detected frame as an init for step-8 refinement.
    """
    params_dir = _aios_params_dir(output_dir)
    if params_dir is None:
        print("Warning: No AiOS params/ directory found to normalize.")
        return

    smplx_params_dir = output_dir / "smplx_params"
    smplx_params_dir.mkdir(exist_ok=True)

    # Read pose and shape files
    pose_files = sorted(params_dir.glob("pose-*.txt"))
    shape_files = sorted(params_dir.glob("shape-*.txt"))

    if not pose_files:
        print("Warning: No pose-*.txt files found in AiOS output.")
        return

    # Compute mean shape from all shape files
    # Pad to 300-D betas (SMPL-X full betas) as expected by downstream refine/align steps.
    shapes = []
    for sf in shape_files:
        shapes.append(np.loadtxt(sf).astype(np.float32))
    if shapes:
        mean_shape = np.mean(shapes, axis=0).flatten()
        padded = np.zeros(300, dtype=np.float32)
        padded[: mean_shape.size] = mean_shape
        np.savetxt(output_dir / "mean_shape.txt", padded)

    # Normalize pose files by their embedded frame id: pose-{id:06d}.txt →
    # smplx_params/{id:04d}.txt, filling detection gaps from the nearest frame.
    detected = {int(pf.stem.split("-")[1]): pf for pf in pose_files}
    if num_frames is None:
        num_frames = max(detected) + 1
    extra = sorted(i for i in detected if i >= num_frames)
    if extra:
        print(f"Warning: ignoring AiOS outputs beyond frame range: {extra}")

    detected_ids = sorted(i for i in detected if i < num_frames)
    filled = []
    for i in range(num_frames):
        if i in detected:
            src = detected[i]
        else:
            nearest = min(detected_ids, key=lambda j: abs(j - i))
            src = detected[nearest]
            filled.append(i)
        params = np.loadtxt(src).astype(np.float32)
        np.savetxt(smplx_params_dir / f"{i:04d}.txt", params)

    if filled:
        print(f"Warning: no AiOS detection for frames {filled}; "
              f"filled from nearest detected frame (refined in step 8).")
    print(f"Normalized {num_frames} AiOS outputs to {smplx_params_dir} "
          f"({len(detected_ids)} detected, {len(filled)} filled)")


def _aios_params_dir(output_dir: Path):
    """AiOS demo writes output_dir/<img_dir_name>/params/; skip retry scratch."""
    dirs = [d for d in output_dir.rglob("params") if "retry" not in d.parts]
    return dirs[0] if dirs else None


def _detected_frame_ids(params_dir: Path) -> set[int]:
    return {int(f.stem.split("-")[1]) for f in params_dir.glob("pose-*.txt")}


def _aios_cmd(args: EstimateSMPLXArgs, input_dir: Path, output_dir: Path,
              threshold: float) -> list[str]:
    aios_root = args.tools.aios_root
    # Use torchrun (env-var based) not torch.distributed.launch (arg-based)
    # to avoid --local-rank conflict with AiOS argparser
    torchrun = str(Path(args.tools.aios_python).parent / "torchrun")
    return [
        torchrun, "--nproc_per_node=1", "--master_port=29501",
        str(aios_root / "main.py"),
        "-c", str(aios_root / "config" / "aios_smplx_demo.py"),
        "--options",
        "batch_size=8",
        "backbone=resnet50",
        f"num_person={args.num_person}",
        f"threshold={threshold}",
        "--output_dir", str(output_dir),
        "--inference",
        "--inference_input", str(input_dir),
        "--resume", str(aios_root / args.checkpoint),
        "--eval",
    ]


_RETRY_THRESHOLDS = (0.2, 0.1, 0.05)


def retry_missing_aios_frames(args: EstimateSMPLXArgs, img_files: list,
                              output_dir: Path) -> None:
    """Re-run AiOS at lower detection thresholds on frames with no prediction.

    The pose-{id:06d} id is the frame's position in the (sorted) inference
    folder, so a retry on the subset of undetected images maps output position
    k back to missing[k] exactly, even when only some retries succeed. Frames
    still undetected after the ladder fall back to the nearest-frame fill in
    normalize_aios_outputs.
    """
    params_dir = _aios_params_dir(output_dir)
    if params_dir is None:
        return
    missing = sorted(set(range(len(img_files))) - _detected_frame_ids(params_dir))
    if not missing:
        return

    retry_root = output_dir / "retry"
    try:
        for threshold in _RETRY_THRESHOLDS:
            if threshold >= args.threshold:
                continue
            print(f"AiOS missed frames {missing}; retrying at threshold {threshold}...")
            stage_dir = retry_root / f"thr{threshold}" / "images"
            retry_out = retry_root / f"thr{threshold}" / "out"
            stage_dir.mkdir(parents=True, exist_ok=True)
            retry_out.mkdir(parents=True, exist_ok=True)
            for i in missing:
                shutil.copy2(img_files[i], stage_dir / img_files[i].name)
            subprocess.run(
                _aios_cmd(args, stage_dir, retry_out, threshold),
                check=True, cwd=str(args.tools.aios_root),
            )
            retry_params = next(iter(retry_out.rglob("params")), None)
            recovered = []
            if retry_params is not None:
                for src in sorted(retry_params.iterdir()):
                    m = re.match(r"([a-z_]+)-(\d+)\.(.+)$", src.name)
                    if m is None:
                        continue
                    prefix, k, suffix = m.group(1), int(m.group(2)), m.group(3)
                    if k >= len(missing):
                        continue
                    frame_id = missing[k]
                    shutil.copy2(src, params_dir / f"{prefix}-{frame_id:06d}.{suffix}")
                    if prefix == "pose":
                        recovered.append(frame_id)
            if recovered:
                print(f"Recovered AiOS predictions for frames {sorted(recovered)} "
                      f"at threshold {threshold}")
            missing = sorted(set(range(len(img_files))) - _detected_frame_ids(params_dir))
            if not missing:
                break
    finally:
        if retry_root.exists():
            shutil.rmtree(retry_root)
    if missing:
        print(f"AiOS still has no prediction for frames {missing} after retries at "
              f"thresholds {_RETRY_THRESHOLDS}; nearest-frame fill will apply.")


def run_aios(args: EstimateSMPLXArgs) -> None:
    require_tools(args.tools, "aios")
    data_dir = args.data_dir.resolve()
    img_dir = data_dir / "images" / "selected_frames"
    output_dir = data_dir / "smplx" / "aios"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running AiOS inference on {img_dir}...")
    subprocess.run(
        _aios_cmd(args, img_dir, output_dir, args.threshold),
        check=True, cwd=str(args.tools.aios_root),
    )

    # Recover undetected frames at lower thresholds, then normalize the
    # output layout to RHINO's expected contract.
    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    retry_missing_aios_frames(args, img_files, output_dir)
    normalize_aios_outputs(output_dir, num_frames=len(img_files))


def main(args: EstimateSMPLXArgs) -> None:
    run_aios(args)


if __name__ == "__main__":
    import tyro
    main(tyro.cli(EstimateSMPLXArgs))
