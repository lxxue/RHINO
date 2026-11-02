# CLAUDE.md

Guidance for AI coding agents working in this repository.

## Project

RHINO (CVPR 2026) jointly reconstructs a human (SMPL-X + VolSDF), a rigid
manipulated object (VolSDF + per-frame 6D pose), and the static background
scene (VolSDF) from a monocular RGB video. The repo contains the training
code, the full preprocessing pipeline (steps 0-13 + finalize), evaluation
tooling, and download links for the BenchRHINO dataset. Installation, usage
commands, preprocessing setup, data formats, and the evaluation workflow are
documented in README.md (collapsible sections) — this file only covers what
the README does not.

## Run contract (training)

- Entry point: `cd rhino && python train.py dataset=<conf> data_root=...
  result_root=... version=<name>`.
- `version=` is required (train.py raises if unset). Hydra composes from
  `rhino/confs/base.yaml`; the run directory is
  `{result_root}/{subject}/hoi/{version}` and Hydra `chdir: true` makes ALL
  outputs (checkpoints/, tblogger/, epoch_*/ exports) relative to it.
- The latest `checkpoints/*.ckpt` in the run dir is auto-resolved for both
  `model.is_continue=True` and `test_only=True`.
- `test_only=True` exports per-frame meshes into `epoch_0000/`; add
  `model.test_render_image=True` for the (slow) volumetric image renders.
- `wandb: True` is the default logger; pass `wandb=False` for TensorBoard.

## Architecture map

- `rhino/train.py` — Hydra entry; builds dataloaders + `rhino.models.hoi.HOI`.
- `rhino/models/hoi.py` — Lightning module: optimization loop, per-frame
  SMPL-X / camera / object-pose parameter groups, validation and test
  rendering; deterministic schedule in `models/training_schedule.py`.
- `rhino/models/hoi_network.py` — `HOINetwork`: three implicit branches
  (human with SMPL-X-driven deformation via `components/deformer.py`, rigid
  object, static background) rendered with VolSDF using the error-bound
  sampler in `components/ray_sampler.py`.
- `rhino/models/hoi_loss.py` — photometric / geometric / contact losses.
- `rhino/datasets/hoi.py` — dataset key `hoi_obj`; pixel sampling strategies
  `fg_obj_bg` (default) and `fg_obj_hand_bg` (adds hand-mask pixels).
- `rhino/rhino_smplx/` — thin wrappers over the pip `smplx` package; no
  body-model code is vendored. The directory is named `rhino_smplx` to avoid
  an import collision with the external package — always import
  `rhino.rhino_smplx.*`. Needs the licensed SMPL-X assets the user downloads
  into `rhino/checkpoints/smplx/`.
- `rhino/confs/` — `dataset/<seq>.yaml` sets `subject`, which interpolates
  into `data_dir: stage|inthewild/<subject>/processed`; shipped configs are
  the BenchRHINO sequences plus `demo_table` and `template.yaml`.
- `rhino/preprocessing/` — every script is a tyro CLI; `run_pipeline.py`
  orchestrates steps 0-13 + finalize; `tool_config.py` centralizes external
  tool roots and interpreters (`--tools.*`); the heavy tools (SAM2, MASt3R,
  AiOS, Sapiens, Metric3D, InteractVLM) are `third_party/` submodules, some
  imported via `sys.path` rather than pip; `validate_*_steps.py` and
  `verify_outputs.py` compare a pipeline run against a reference run.
- `rhino/evaluation/` — `eval_recon.py` (first-frame scaled ICP + per-frame
  chamfer / Hausdorff / F-score vs the BenchRHINO GT `full/` meshes),
  `mesh_sequence.py` (filesystem contracts), `render_mesh_seq.py`
  (AITViewer, intentionally a separate environment; stage-camera views need
  a user-supplied calibration, none ships),
  `video_utils.py` + `create_video*.py` (comparison videos).

## Data facts

- BenchRHINO: 7 stage sequences `S1_Take3`, `S1_Take7`, `S1_Take8`,
  `S2_Take5`, `S2_Take9`, `S3_Take6`, `S4_Take2`; the in-the-wild demo
  sequence `demo_table` ships in the demo bundle.
- `rhino/confs/sequence_take_mapping.json`: per-sequence GT frame offset
  (`gt_frame = pred_frame + offset`), frame counts, SMPL-X `gender` (of the
  GT fits and the step-11 heuristic only — training always uses the neutral
  model, `models/hoi.py`), and the `object_name` used as the InteractVLM
  contact prompt.
- Pretrained checkpoints unpack to
  `<result_root>/<seq>/hoi/pretrained/checkpoints/`.

## Gotchas

- numpy must stay <2: the pinned kaolin 0.17.0 / pytorch3d wheels break
  against the numpy 2.x ABI (`import kaolin` fails with a dtype-size error).
- There is no test suite; verify changes by importing the touched modules and
  running a short training or `test_only` pass.
- Default object pose file is `object_poses_centered.npy`; scene cameras are
  `intrinsic.npy` + `c2ws.npy` in `processed/`.
- Preprocessing scripts resolve the repo root via
  `_REPO_ROOT = Path(__file__).resolve().parents[2]` — keep their directory
  depth if moving files.
- Step 1 defaults to the SAM 2.0 checkpoint `sam2_hiera_large.pt` and
  hardcodes the matching `sam2_hiera_l.yaml` config; SAM 2.1 weights would
  mismatch.
- InteractVLM (step 12) runs 4-bit quantized by default to fit 24 GB GPUs.
- Hand masks ship but are less robust than the human/object masks and are
  unused by the default `fg_obj_bg` sampling.
- Never commit SMPL-X body-model assets; they are licensed for
  non-commercial use and are user-supplied.
- Source builds against the pinned torch versions need `setuptools<70` and
  `pip --no-build-isolation` on current pip (pytorch3d, mmcv, chumpy,
  xtcocotools, the sapiens editables); several tool setup.py files still
  import `pkg_resources` at runtime too.
- Run everything with `PYTHONNOUSERSITE=1`: packages in `~/.local` otherwise
  leak into conda envs and make pip skip installing real dependencies.
- The validators' absolute pose checks assume run and reference share a
  processing mode (stage/AprilTag vs in-the-wild/normalize-scene world
  frames differ by a Sim(3) + axis flip); cross-mode runs legitimately flag
  there while masks/depth/contact/relative-motion checks stay meaningful.
