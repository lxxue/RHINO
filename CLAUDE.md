# CLAUDE.md

## Project Overview

This is the **public release repo** for RHINO (Reconstructing Human Interactions with Novel Objects), accepted at CVPR 2026. It contains training code for jointly reconstructing humans (VolSDF + SMPL-X), objects (VolSDF + 6D pose), and background scenes (static VolSDF) from monocular video, plus the full preprocessing pipeline (steps 0-13 + finalize) and evaluation tooling.

## Key Facts

- **Training code**: Hydra + PyTorch Lightning; entry point `rhino/train.py`. The three implicit branches (human / object / background) live in `rhino/models/hoi_network.py`, losses in `rhino/models/hoi_loss.py`, the Lightning module in `rhino/models/hoi.py`.
- **Preprocessing**: steps 0-13 + finalize, orchestrated by `rhino/preprocessing/run_pipeline.py`; all CLI uses tyro. Steps that need dedicated tool environments (SAM2, MASt3R, Metric3D, AiOS, Sapiens, InteractVLM) take their interpreters from `tool_config.py` (`--tools.*` flags).
- **Contact detection (step 11)**: hybrid proximity+motion heuristic with hysteresis (`rhino/preprocessing/contact_heuristic.py`); one global config, no per-sequence tuning; `--motion-only` is the simpler fallback. Per-sequence gender/object_name metadata lives in `rhino/confs/sequence_take_mapping.json`.
- **InteractVLM (step 12)**: runs on contact frames only, prompts with the per-sequence object name, and defaults to 4-bit NF4 + bf16 quantization (`--interactvlm-quant`), which fits a 24GB GPU (~18GB). The dtype fix and `--object_name` support are baked into the pinned lxxue/InteractVLM fork.
- **Scene cameras**: step 9 (align_smplx) updates `processed/{c2ws.npy, intrinsic.npy, scene_pcd.ply}` with (possibly normalized) cameras from the reconstruction; step 10 reads scene c2ws from `processed/`. For stage sequences, `colmap model_aligner` with AprilTag references runs after step 2 (`--ref-camera-centers`).
- **Camera/pose conventions**: scene cameras live in `intrinsic.npy` + `c2ws.npy`; the default object pose file is `object_poses_centered.npy`.
- **In-the-wild sequences**: use `--normalize-scene` (step 9 centers/rescales the scene from the SMPL-X alignment; the near-1 scale sanity check applies only to non-normalized stage scenes). Step 6 keys AiOS outputs by the embedded frame id so missing detections cannot shift later frames; undetected frames are re-run through AiOS at descending thresholds (0.2/0.1/0.05), with nearest-frame fill only as a last resort. Step 9 records `processed/scene_normalization.json` on the normalize path; validators auto-detect normalized sequences from that marker (SMPL-X translation becomes advisory; step-2 raw model resolved per the reference layout).
- **Hand masks**: included but less robust than human/object masks; `create_masks.py` re-seeds Grounding DINO inside empty SAM2 runs to recover dropouts. Boundaries are seed-sensitive. Unused by the default training sampling (`fg_obj_bg`).
- **Validators**: `validate_raw_steps.py` (0-3), `validate_processed_steps.py` (4-10), `validate_contact_steps.py` (11-13), `verify_outputs.py` (end-to-end) compare a pipeline run against a reference run. Comparison semantics account for run-to-run variance: the camera gate scales with trajectory size (0.5% of bbox diagonal, 1cm floor); rotations and SMPL-X translations are compared after removing the global world-frame offset (AprilTag model_aligner ambiguity); betas are gated on induced canonical vertex displacement (15mm); object poses / contact / hand masks are advisory (SfM variance, heuristic sensitivity, seed-sensitive hand tracking).
- **Evaluation**: `rhino/evaluation/` (per-frame ICP + chamfer/Hausdorff/F-score, headless rendering, comparison videos). Requires volumetric-capture GT meshes.

## Repo Structure

```
third_party/
├── mast3r/                         # Git submodule (lxxue fork, main branch)
├── Grounded-SAM-2/                 # Git submodule (IDEA-Research)
├── Hierarchical-Localization/      # Git submodule (lxxue fork, image_mask branch)
├── InteractVLM/                    # Git submodule (lxxue fork: dtype fix + --object_name)
├── AiOS/                           # Git submodule (lxxue fork)
├── sapiens/                        # Git submodule (facebookresearch, official)
└── Metric3D/                       # Git submodule (YvanYin, official)

rhino/
├── train.py                    # Training entry point (Hydra + PyTorch Lightning)
├── models/
│   ├── hoi.py                  # HOI Lightning module
│   ├── hoi_network.py          # HOINetwork (VolSDF rendering)
│   ├── hoi_loss.py             # HOILoss
│   ├── training_schedule.py    # Deterministic optimizer/contact schedule
├── datasets/hoi.py             # HOI dataset loader
├── datasets/__init__.py        # Factory (HOI-only; dataset key "hoi_obj")
├── datasets/frame_utils.py     # Frame-selection contracts shared with training
├── components/
│   ├── ray_sampler.py          # VolSDF error-bound ray sampling
│   ├── network.py              # ImplicitNet, RenderingNet
│   ├── deformer.py             # RigidDeformer + skinning
│   ├── loss.py, density.py, embedder.py, body_sampler.py
│   ├── body_model_params.py, camera_pose_params.py, object_pose_params.py
├── preprocessing/              # Steps 0-13 + finalize
│   ├── select_frames.py        # Step 0: frame extraction
│   ├── create_masks.py         # Step 1: SAM2 + Grounding DINO masks
│   ├── estimate_camera_poses.py # Step 2: camera pose estimation (hloc/COLMAP)
│   ├── estimate_object_poses.py # Step 3: object pose estimation (hloc/COLMAP)
│   ├── extract_monocular_cues.py # Step 4: Metric3D depth + normals
│   ├── prepare_base.py         # Step 5: base processed/ assembly
│   ├── estimate_smplx.py       # Step 6: SMPL-X estimation (AiOS)
│   ├── estimate_keypoints.py   # Step 7: Sapiens whole-body keypoints
│   ├── convert_keypoints_to_smplx.py # Step 7 helper: coco_wholebody → SMPL-X (AiOS env)
│   ├── refine_smplx.py         # Step 8: SMPL-X refinement
│   ├── align_smplx.py          # Step 9: SMPL-X scene alignment
│   ├── align_object_poses.py   # Step 10: object pose alignment + centering
│   ├── detect_contact_frames.py # Step 11: contact frame detection
│   ├── contact_heuristic.py    # Shared step-11 signal/rule module
│   ├── extract_contact_vertices.py # Step 12: InteractVLM contact extraction
│   ├── refine_contact.py       # Step 13: contact prediction refinement
│   ├── finalize_processed.py   # Finalize: validate + manifest.json
│   ├── run_pipeline.py         # Orchestrator for steps 0-13 + finalize
│   ├── tool_config.py          # Centralized tool/interpreter configuration
│   ├── path_utils.py           # Shared sequence-dir resolution
│   ├── verify_outputs.py       # End-to-end verification vs a reference run
│   ├── validate_*_steps.py     # Per-step-group validators
│   ├── mast3r_matching.py      # MASt3R-based SfM (alt for steps 2+3)
│   ├── colmap_utils.py         # run_command + undistort_images
│   ├── record3d_utils.py       # Record3D camera loading & triangulation
│   ├── align_utils.py          # Scene alignment utilities
│   ├── hoi_utils.py            # RANSAC, similarity transforms
│   └── smplx_utils.py          # GMoF robust loss (step 8)
├── rhino_smplx/                # SMPL-X wrappers on top of the external smplx package
│   ├── smplx_server.py, smplx_deformer.py, cano_smplx_params.py
├── utils/
│   ├── mesh_utils.py, render_utils.py, idr_utils.py, misc.py, sample_utils.py
│   └── sdf_vis.py              # SDF slice visualization
├── evaluation/                 # eval_recon, render_mesh_seq, mesh_sequence contracts, video_utils, comparison videos
├── confs/
│   ├── base.yaml               # Root config (placeholder paths)
│   ├── model/hoi.yaml          # Model config (cls: rhino.models.hoi.HOI)
│   ├── dataset/template.yaml   # Dataset config template
│   └── sequence_take_mapping.json
└── checkpoints/
    ├── pose_mean_hand.npy      # Required by cano_smplx_params.py
    └── smplx_init_last.ckpt    # Bundled canonical-human SDF initialization

```

## Preprocessing Submodule Paths

Third-party submodules are at the repo root under `third_party/`. Preprocessing scripts resolve paths dynamically via `_REPO_ROOT = Path(__file__).resolve().parents[2]` (2 levels up from `rhino/preprocessing/`). SAM2 and MASt3R are put on `sys.path` from their configured roots at run time.

## Important: rhino_smplx wraps the external smplx package

`rhino/rhino_smplx/` contains no vendored SMPL-X code: `SMPLXServer` builds a
plain `smplx.SMPLX` from the pip package and recomputes the per-joint rigid
transforms and skinning weights via `smplx.lbs` public helpers. The directory
name avoids an import collision with the external `smplx` package; all
internal imports use `rhino.rhino_smplx.*`.

## Common Commands

```bash
conda activate rhino

# Training
cd rhino && python train.py dataset=<dataset_config> version=<version>
```
