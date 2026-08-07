"""RHINO preprocessing pipeline orchestrator.

Dispatches steps 0-13 + finalize. Each step is invoked as a subprocess.
Use --steps to select which steps to run. Use --finalize to run validation.
Invalid step numbers are rejected with a clear error.
"""

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import tyro

from rhino.preprocessing.path_utils import resolve_camera_pose_dir, resolve_sequence_dir
from rhino.preprocessing.tool_config import ToolConfig

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]

VALID_STEPS = set(range(14))  # 0-13

STEP_NAMES = {
    0: "select_frames",
    1: "create_masks",
    2: "estimate_camera_poses",
    3: "estimate_object_poses",
    4: "extract_monocular_cues",
    5: "prepare_base",
    6: "estimate_smplx",
    7: "estimate_keypoints",
    8: "refine_smplx",
    9: "align_smplx",
    10: "align_object_poses",
    11: "detect_contact_frames",
    12: "extract_contact_vertices",
    13: "refine_contact",
}

# Dependencies: step N requires these steps to have produced output first
STEP_DEPS = {
    0: [],
    1: [0],
    2: [0, 1],
    3: [0, 1],
    4: [0],
    5: [0, 1, 2, 4],
    6: [0],
    7: [0],
    8: [5, 6, 7],
    9: [2, 5, 8],
    10: [3, 5, 9],
    11: [9, 10],
    12: [0, 5, 11],
    13: [11, 12],
}


def run_step(cmd: list[str], env: dict[str, str] | None = None) -> None:
    import os
    print(f"  CMD: {' '.join(cmd[:4])}...")
    run_env = {**os.environ}
    # Ensure the repo root is on PYTHONPATH so that subprocesses using
    # different conda interpreters can still import the rhino package.
    repo_root = str(_SCRIPT_DIR.parents[1])
    existing = run_env.get("PYTHONPATH", "")
    if repo_root not in existing.split(os.pathsep):
        run_env["PYTHONPATH"] = repo_root + (os.pathsep + existing if existing else "")
    if env:
        run_env.update(env)
    subprocess.run(cmd, check=True, env=run_env)


def _tools_args(tools: ToolConfig) -> list[str]:
    """Generate CLI args for non-default ToolConfig fields.

    Compares against a fresh ToolConfig() and returns --tools.field-name value
    pairs for any overridden fields. Used to forward tool configuration from
    the pipeline orchestrator to child subprocess scripts.
    """
    default = ToolConfig()
    args = []
    for name in ToolConfig.__dataclass_fields__:
        val = getattr(tools, name)
        default_val = getattr(default, name)
        if val != default_val and val is not None:
            cli_name = f"--tools.{name.replace('_', '-')}"
            args.extend([cli_name, str(val)])
    return args


def check_step_deps(step: int, data_dir: Path) -> None:
    """Check that prerequisite step outputs exist."""
    checks = {
        0: lambda: (data_dir / "images" / "selected_frames").is_dir(),
        1: lambda: (data_dir / "masks" / "sam2_human").is_dir(),
        2: lambda: True,  # Camera poses dir varies
        3: lambda: True,  # Object poses dir varies
        4: lambda: (data_dir / "monocular_geometry" / "depth").is_dir(),
        5: lambda: (data_dir / "processed" / "intrinsic.npy").exists(),
        6: lambda: (data_dir / "smplx").is_dir(),
        7: lambda: (data_dir / "keypoints").is_dir(),
        8: lambda: True,  # Refined SMPL-X checked inside
        9: lambda: (data_dir / "processed" / "aligned_smplx.pkl").exists(),
        10: lambda: (data_dir / "processed" / "object_poses.npy").exists(),
        11: lambda: (data_dir / "processed" / "contact_labels.npy").exists(),
        12: lambda: (data_dir / "processed" / "contact_vertices.npy").exists(),
        13: lambda: (data_dir / "processed" / "contact_vertices_refined.npy").exists(),
    }
    for dep in STEP_DEPS.get(step, []):
        check_fn = checks.get(dep)
        if check_fn and not check_fn():
            print(f"Warning: step {step} depends on step {dep} ({STEP_NAMES[dep]}), "
                  f"but its output was not found. It may fail.")


@dataclass
class PipelineArgs:
    data_dir: Optional[Path] = None
    """Path to the sequence data directory. Required unless --data-root + --subject are given."""
    data_root: Optional[Path] = None
    """Root directory containing stage/ and/or inthewild/. Resolves to --data-dir via --subject."""
    subject: Optional[str] = None
    """Sequence subject name under stage/ or inthewild/ inside --data-root."""
    steps: List[int] = field(default_factory=list)
    """Step numbers to run (0-13)."""
    step: Optional[int] = None
    """Single step number (0-13). Alias for --steps with one element."""
    finalize: bool = False
    """Run the finalize/validation step after all other steps."""

    # Step 0: Frame selection
    input_path: Optional[Path] = None
    """Input video or image directory (required for step 0)."""
    window_size: int = 10
    """Sharpness selection window size."""
    frame_start: int = 0
    frame_end: int = 1000000
    video_resize_factor: int = 2

    # Step 1: Masks
    text: str = "table."
    """Object text query for SAM2 (lowercase, end with dot)."""
    skip_human_mask: bool = False
    skip_object_mask: bool = False
    skip_hand_mask: bool = False
    hand_tracking_window: int = 10

    # Step 2-3: Camera/object poses
    ref_camera_centers: Optional[Path] = None
    """Path to reference camera centers (e.g. apriltag/camera_centers.txt) for colmap model_aligner.
    Used for stage sequences to align SfM to real-world coordinates. Skipped for in-the-wild."""
    model_aligner_max_error: float = 0.1
    """Max error for colmap model_aligner robust alignment."""
    camera_model: str = "SIMPLE_PINHOLE"
    sfm_tool: str = "hloc"
    feature_type: str = "superpoint_aachen"
    matching_method: Optional[str] = None
    matcher_type: str = "superpoint+lightglue"
    camera_use_mast3r: bool = False
    camera_use_glomap: bool = False
    camera_pair_window_size: int = 5
    record3d_num_ba_iterations: int = 3
    """Number of Record3D triangulation/bundle-adjustment rounds."""
    object_use_mast3r: bool = False
    object_use_glomap: bool = False
    object_pair_window_size: int = 5
    bg_color: List[int] = field(default_factory=lambda: [255, 255, 255])

    # Step 5: Prepare base
    prepare_base_resize_factor: int = 1

    # Step 6: SMPL-X estimation
    smplx_threshold: float = 0.3
    """AiOS detection confidence threshold."""
    smplx_num_person: int = 1
    """Maximum persons per frame for AiOS."""

    # Step 9: SMPL-X alignment
    interactive_alignment: bool = False
    normalize_scene: bool = False
    """Normalize scene using SMPL-X alignment (needed for in-the-wild sequences)."""

    # Step 11: Contact detection
    seq_name: str = ""
    """Sequence name for metadata lookup in sequence_take_mapping.json (gender, object_name)."""
    gender: str = ""
    """SMPL-X gender for the step-11 proximity heuristic. Overrides the seq_name lookup."""
    contact_motion_only: bool = False
    """Use the motion-only contact fallback (no SMPL-X hands needed) in step 11."""

    # Step 12: InteractVLM
    interactvlm_quant: str = "4bit"
    """InteractVLM quantization: '4bit' (NF4), '8bit' (LLM.int8), or 'none' (needs >24GB)."""
    interactvlm_model_version: str = "./trained_models/interactvlm-3d-hcontact-damon"
    """InteractVLM trained-model checkpoint for step 12 (path relative to interactvlm_root).

    Default matches InteractVLM's scripts/run_demo.sh hcontact mapping. Override for ocontact."""
    interactvlm_object_name: str = ""
    """Object name for the InteractVLM contact prompt (e.g. 'gray_box').

    Defaults to --text with the trailing period stripped and spaces replaced
    by underscores (matching how the reference data was generated)."""

    # Tool configuration
    tools: ToolConfig = field(default_factory=ToolConfig)


def main(args: PipelineArgs) -> None:
    resolved_data_dir = resolve_sequence_dir(args.data_dir, args.data_root, args.subject)

    # Merge --step alias into --steps
    steps = list(args.steps)
    if args.step is not None:
        if args.step not in steps:
            steps.append(args.step)
    steps = sorted(set(steps))

    # Validate step numbers
    invalid = set(steps) - VALID_STEPS
    if invalid:
        print(f"Error: Invalid step number(s): {sorted(invalid)}")
        print(f"Valid steps: 0-13")
        sys.exit(1)

    if not steps and not args.finalize:
        print("No steps selected. Use --steps N [N ...], --step N, and/or --finalize.")
        sys.exit(1)

    if 3 in steps and args.object_use_mast3r and not args.camera_use_mast3r:
        print(
            "Error: --object-use-mast3r requires --camera-use-mast3r: the mast3r "
            "object-pose path reads its focal length from the mast3r scene cameras.",
            file=sys.stderr,
        )
        sys.exit(1)

    data_dir = str(resolved_data_dir.resolve())
    matching_method = args.matching_method or ("poses" if args.sfm_tool == "record3d" else "sequential")

    for step in steps:
        print(f"\n=== Step {step}: {STEP_NAMES[step]} ===")
        check_step_deps(step, resolved_data_dir.resolve())

        if step == 0:
            if args.input_path is None:
                print("Error: --input-path required for step 0")
                sys.exit(1)
            run_step([
                sys.executable, str(_SCRIPT_DIR / "select_frames.py"),
                "--data_dir", data_dir,
                "--input_path", str(args.input_path.resolve()),
                "--window_size", str(args.window_size),
                "--frame_start", str(args.frame_start),
                "--frame_end", str(args.frame_end),
                "--video_resize_factor", str(args.video_resize_factor),
            ])

        elif step == 1:
            sam2_tool_args = [
                "--tools.sam2-root", str(args.tools.sam2_root),
                "--tools.sam2-checkpoint", str(args.tools.sam2_checkpoint),
            ]
            if not args.skip_human_mask:
                run_step([
                    args.tools.sam2_python, str(_SCRIPT_DIR / "create_masks.py"),
                    "--data_dir", data_dir, "--text", "human.",
                ] + sam2_tool_args)
            if not args.skip_object_mask:
                run_step([
                    args.tools.sam2_python, str(_SCRIPT_DIR / "create_masks.py"),
                    "--data_dir", data_dir, "--text", args.text,
                    "--is_object",
                    "--bg_color", str(args.bg_color[0]), str(args.bg_color[1]), str(args.bg_color[2]),
                ] + sam2_tool_args)
            if not args.skip_hand_mask:
                run_step([
                    args.tools.sam2_python, str(_SCRIPT_DIR / "create_masks.py"),
                    "--data_dir", data_dir, "--text", "hand.",
                    "--is_hand", "--tracking_window", str(args.hand_tracking_window),
                ] + sam2_tool_args)

        elif step == 2:
            camera_pose_dir_resolved = resolve_camera_pose_dir(
                resolved_data_dir.resolve(), args.sfm_tool, args.feature_type,
                matching_method, args.matcher_type,
                args.camera_use_mast3r, args.camera_use_glomap, args.camera_pair_window_size,
                args.record3d_num_ba_iterations,
            )
            if not args.camera_use_mast3r:
                run_step([
                    sys.executable, str(_SCRIPT_DIR / "estimate_camera_poses.py"),
                    "--data_dir", data_dir,
                    "--camera_model", args.camera_model,
                    "--sfm_tool", args.sfm_tool,
                    "--feature_type", args.feature_type,
                    "--matching_method", matching_method,
                    "--matcher_type", args.matcher_type,
                    "--video_resize_factor", str(args.video_resize_factor),
                    "--record3d-num-ba-iterations", str(args.record3d_num_ba_iterations),
                ])
            else:
                run_step([
                    args.tools.mast3r_python, str(_SCRIPT_DIR / "mast3r_matching.py"),
                    "--data_dir", data_dir,
                    "--pair_window_size", str(args.camera_pair_window_size),
                    "--no-is-object",
                    "--use-glomap-mapper" if args.camera_use_glomap else "--no-use-glomap-mapper",
                ])

            # Optional: align SfM to reference camera positions (stage sequences)
            if args.ref_camera_centers is not None:
                ref_path = args.ref_camera_centers.resolve()
                if not ref_path.exists():
                    print(f"Error: reference camera centers not found: {ref_path}")
                    sys.exit(1)
                original_dir = camera_pose_dir_resolved.parent / "original"
                if not original_dir.exists():
                    import shutil
                    print(f"  Moving raw SfM to {original_dir}")
                    shutil.copytree(camera_pose_dir_resolved, original_dir)
                run_step([
                    "colmap", "model_aligner",
                    "--input_path", str(original_dir),
                    "--output_path", str(camera_pose_dir_resolved),
                    "--ref_images_path", str(ref_path),
                    "--ref_is_gps", "0",
                    "--alignment_max_error", str(args.model_aligner_max_error),
                ])

        elif step == 3:
            camera_pose_dir_resolved = resolve_camera_pose_dir(
                resolved_data_dir.resolve(), args.sfm_tool, args.feature_type,
                matching_method, args.matcher_type,
                args.camera_use_mast3r, args.camera_use_glomap, args.camera_pair_window_size,
                args.record3d_num_ba_iterations,
            )
            if not args.object_use_mast3r:
                run_step([
                    sys.executable, str(_SCRIPT_DIR / "estimate_object_poses.py"),
                    "--data_dir", data_dir,
                    "--camera_model", args.camera_model,
                    "--sfm_tool", args.sfm_tool,
                    "--feature_type", args.feature_type,
                    "--matching_method", matching_method,
                    "--matcher_type", args.matcher_type,
                    "--camera_pose_dir", str(camera_pose_dir_resolved),
                ])
            else:
                run_step([
                    args.tools.mast3r_python, str(_SCRIPT_DIR / "mast3r_matching.py"),
                    "--data_dir", data_dir,
                    "--is-object",
                    "--pair_window_size", str(args.object_pair_window_size),
                    "--camera_pair_window_size", str(args.camera_pair_window_size),
                    "--pose_method", "mast3r",
                    "--use-glomap-mapper" if args.object_use_glomap else "--no-use-glomap-mapper",
                ])

        elif step == 4:
            run_step([
                args.tools.metric3d_python, str(_SCRIPT_DIR / "extract_monocular_cues.py"),
                "--data_dir", data_dir,
            ] + _tools_args(args.tools))

        elif step == 5:
            camera_pose_dir_resolved = resolve_camera_pose_dir(
                resolved_data_dir.resolve(), args.sfm_tool, args.feature_type,
                matching_method, args.matcher_type,
                args.camera_use_mast3r, args.camera_use_glomap, args.camera_pair_window_size,
                args.record3d_num_ba_iterations,
            )
            run_step([
                sys.executable, str(_SCRIPT_DIR / "prepare_base.py"),
                "--data_dir", data_dir,
                "--camera_pose_dir", str(camera_pose_dir_resolved),
                "--resize_factor", str(args.prepare_base_resize_factor),
            ])

        elif step == 6:
            run_step([
                args.tools.aios_python, str(_SCRIPT_DIR / "estimate_smplx.py"),
                "--data_dir", data_dir,
                "--threshold", str(args.smplx_threshold),
                "--num-person", str(args.smplx_num_person),
            ] + _tools_args(args.tools))

        elif step == 7:
            run_step([
                args.tools.sapiens_python, str(_SCRIPT_DIR / "estimate_keypoints.py"),
                "--data_dir", data_dir,
            ] + _tools_args(args.tools))

        elif step == 8:
            run_step([
                sys.executable, str(_SCRIPT_DIR / "refine_smplx.py"),
                "--data_dir", data_dir,
            ] + _tools_args(args.tools))

        elif step == 9:
            camera_pose_dir_resolved = resolve_camera_pose_dir(
                resolved_data_dir.resolve(), args.sfm_tool, args.feature_type,
                matching_method, args.matcher_type,
                args.camera_use_mast3r, args.camera_use_glomap, args.camera_pair_window_size,
                args.record3d_num_ba_iterations,
            )
            cmd = [
                sys.executable, str(_SCRIPT_DIR / "align_smplx.py"),
                "--data_dir", data_dir,
                "--camera_pose_dir", str(camera_pose_dir_resolved),
            ] + _tools_args(args.tools)
            if args.interactive_alignment:
                cmd.append("--interactive")
            if args.normalize_scene:
                cmd.append("--normalize")
            run_step(cmd)

        elif step == 10:
            # Resolve object pose dir
            if args.object_use_mast3r:
                mapper = "glomap" if args.object_use_glomap else "colmap"
                obj_pose_dir = resolved_data_dir.resolve() / "object_poses" / "mast3r" / f"{mapper}_sparse_{args.object_pair_window_size}" / "0"
            else:
                obj_pose_dir = resolved_data_dir.resolve() / "object_poses" / f"hloc-{args.feature_type}-{matching_method}-{args.matcher_type}"
            run_step([
                sys.executable, str(_SCRIPT_DIR / "align_object_poses.py"),
                "--data_dir", data_dir,
                "--object_camera_pose_dir", str(obj_pose_dir),
            ])

        elif step == 11:
            cmd_11 = [
                sys.executable, str(_SCRIPT_DIR / "detect_contact_frames.py"),
                "--data_dir", data_dir,
            ]
            if args.seq_name:
                cmd_11.extend(["--seq_name", args.seq_name])
            if args.gender:
                cmd_11.extend(["--gender", args.gender])
            if args.contact_motion_only:
                cmd_11.append("--motion-only")
            run_step(cmd_11)

        elif step == 12:
            object_name = args.interactvlm_object_name
            if not object_name and args.seq_name:
                mapping_path = _REPO_ROOT / "rhino" / "confs" / "sequence_take_mapping.json"
                if mapping_path.exists():
                    with open(mapping_path) as f:
                        for seq in json.load(f).get("sequences", []):
                            if seq["seq_name"] == args.seq_name and "object_name" in seq:
                                object_name = seq["object_name"]
                                break
            if not object_name and args.text:
                object_name = args.text.strip().rstrip(".").replace(" ", "_")
            cmd = [
                args.tools.interactvlm_python, str(_SCRIPT_DIR / "extract_contact_vertices.py"),
                "--data_dir", data_dir,
                "--model-version", args.interactvlm_model_version,
                "--quant", args.interactvlm_quant,
            ] + _tools_args(args.tools)
            if object_name:
                cmd.extend(["--object-name", object_name])
            run_step(cmd)

        elif step == 13:
            run_step([
                sys.executable, str(_SCRIPT_DIR / "refine_contact.py"),
                "--data_dir", data_dir,
            ])

    # Finalize
    if args.finalize:
        print(f"\n=== Finalize: validate processed/ ===")
        finalize_cmd = [
            sys.executable, str(_SCRIPT_DIR / "finalize_processed.py"),
            "--data_dir", data_dir,
            "--quantization-mode", args.interactvlm_quant,
        ]
        run_step(finalize_cmd)


if __name__ == "__main__":
    main(tyro.cli(PipelineArgs))
