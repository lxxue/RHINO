from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List
import os
import sys
from pathlib import Path

if TYPE_CHECKING:
    from sam2.sam2_image_predictor import SAM2ImagePredictor

# Ensure rhino package is importable when invoked with an external interpreter
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rhino.preprocessing.tool_config import ToolConfig

import cv2
import imageio.v3 as iio
import numpy as np
import supervision as sv
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

# SAM2 lives outside site-packages; build_models() puts tools.sam2_root on
# sys.path before importing it, so a --tools.sam2-root override is honored.

# adapted from https://github.com/IDEA-Research/Grounded-SAM-2/blob/main/grounded_sam2_tracking_demo.py
# hand tracking logic additionally draws on:
# https://github.com/hongsukchoi/VideoMimic/blob/main/real2sim/stage0_preprocessing/sam2_segmentation.py

SAM2_CONFIG = "sam2_hiera_l.yaml"
GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
TEXT_THRESHOLD = 0.3

HAND_TRACK_IOU_THRESHOLD = 0.5
HAND_MIN_AREA = 200
HAND_MIN_FRAMES = 3
HAND_MIN_HUMAN_OVERLAP = 0.5
HAND_MAX_TRACKS = 2
HAND_BACKFILL_INVALID_STREAK = 3
HAND_HUMAN_DILATE_KERNEL = 15
HAND_DETECTION_DEDUP_IOU = 0.7
HAND_ANCHOR_REFINE_TOPK = 5
HAND_RECOVERY_MIN_RUN = 3
HAND_RECOVERY_MAX_ROUNDS = 12


def create_obj_images(data_dir, bg_color=[255, 255, 255]):
    image_dir = data_dir / "images" / "selected_frames"
    obj_mask_dir = data_dir / "masks" / "sam2_obj"
    colmap_mask_dir = data_dir / "masks" / "colmap_obj"
    colmap_mask_dir.mkdir(parents=False, exist_ok=True)
    obj_image_dir = data_dir / "images" / "objects"
    obj_image_dir.mkdir(parents=True, exist_ok=True)
    for image_path in image_dir.iterdir():
        obj_image_path = obj_image_dir / image_path.name
        obj_mask_path = colmap_mask_dir / (image_path.name + ".png")
        img = iio.imread(str(image_path))
        mask = iio.imread(str(obj_mask_dir / image_path.name))
        img[mask <= 127] = bg_color
        iio.imwrite(obj_image_path, img)
        iio.imwrite(obj_mask_path, mask)
    return


def create_colmap_mask(mask_obj_dir, mask_human_dir, mask_all_dir, mask_colmap_dir):
    human_mask_fnames = sorted(mask_human_dir.glob("*.jpg"))
    obj_mask_fnames = sorted(mask_obj_dir.glob("*.jpg"))
    assert len(human_mask_fnames) == len(obj_mask_fnames)

    mask_dilated_human_dir = mask_colmap_dir.parent / "sam2_vis_human_dilated"
    mask_dilated_human_dir.mkdir(exist_ok=True)
    image_dir = mask_colmap_dir.parents[1] / "images" / "selected_frames"

    for i in range(len(human_mask_fnames)):
        human_mask_fname = human_mask_fnames[i]
        obj_mask_fname = obj_mask_fnames[i]

        human_mask = cv2.imread(str(human_mask_fname))
        obj_mask = cv2.imread(str(obj_mask_fname))

        # Dilate the human mask more aggressively to keep dynamic content away from COLMAP.
        dilated_human_mask = cv2.dilate(human_mask, np.ones((5, 5), np.uint8), iterations=5)
        dilated_human_mask_vis = cv2.imread(str(image_dir / human_mask_fname.name))
        dilated_human_mask_vis[dilated_human_mask < 127] = 0
        cv2.imwrite(str(mask_dilated_human_dir / human_mask_fname.name), dilated_human_mask_vis)

        mask_all = np.maximum(human_mask, obj_mask)
        dilated_mask_all = cv2.dilate(mask_all, np.ones((5, 5), np.uint8), iterations=5)
        dilated_mask_scene = 255 - dilated_mask_all
        cv2.imwrite(str(mask_all_dir / human_mask_fname.name), mask_all)
        cv2.imwrite(str(mask_colmap_dir / (human_mask_fname.name + ".png")), dilated_mask_scene)


def get_frame_names(video_dir: Path) -> list[str]:
    frame_names = [
        p
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    return frame_names


def build_models(tools: ToolConfig = None):
    if tools is None:
        tools = ToolConfig()
    sam2_root = str(tools.sam2_root)
    if sam2_root not in sys.path:
        sys.path.insert(0, sam2_root)
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam2_checkpoint = str(tools.sam2_checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    video_predictor = build_sam2_video_predictor(SAM2_CONFIG, sam2_checkpoint, device=device)
    sam2_image_model = build_sam2(SAM2_CONFIG, sam2_checkpoint, device=device)
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL_ID)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GROUNDING_DINO_MODEL_ID
    ).to(device)
    return device, video_predictor, image_predictor, processor, grounding_model


def load_image_pil(video_dir: Path, frame_names: list[str], frame_idx: int) -> Image.Image:
    return Image.open(video_dir / frame_names[frame_idx])


def detect_masks_on_frame(
    frame_idx: int,
    text: str,
    video_dir: Path,
    frame_names: list[str],
    image_predictor: SAM2ImagePredictor,
    processor: AutoProcessor,
    grounding_model: AutoModelForZeroShotObjectDetection,
    device: str,
) -> list[np.ndarray]:
    image = load_image_pil(video_dir, frame_names, frame_idx)
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[image.size[::-1]],
    )
    input_boxes = results[0]["boxes"].cpu().numpy()
    if len(input_boxes) == 0:
        return []

    image_predictor.set_image(np.array(image.convert("RGB")))
    masks_pred, _, _ = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    if masks_pred.ndim == 3:
        masks_pred = masks_pred[None]
    elif masks_pred.ndim == 4:
        masks_pred = masks_pred.squeeze(1)

    masks = []
    for mask in masks_pred:
        if mask.ndim == 3:
            mask = mask[0]
        mask = mask.astype(bool)
        if mask.any():
            masks.append(mask)
    return masks


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def mask_overlap_ratio(mask: np.ndarray, support_mask: np.ndarray | None) -> float:
    if support_mask is None:
        return 1.0
    area = mask.sum()
    if area == 0:
        return 0.0
    overlap = np.logical_and(mask, support_mask).sum()
    return float(overlap / area)


def copy_track_masks(track_masks: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    return {track_id: mask.copy() for track_id, mask in track_masks.items()}


def deduplicate_hand_masks(masks: list[np.ndarray]) -> list[np.ndarray]:
    deduplicated = []
    for mask in sorted(masks, key=lambda m: int(m.sum()), reverse=True):
        if any(mask_iou(mask, kept_mask) >= HAND_DETECTION_DEDUP_IOU for kept_mask in deduplicated):
            continue
        deduplicated.append(mask.copy())
    return deduplicated


def select_valid_hand_masks(
    masks: list[np.ndarray], human_mask: np.ndarray | None, max_count: int = HAND_MAX_TRACKS
) -> list[np.ndarray]:
    valid_masks = []
    for mask in deduplicate_hand_masks(masks):
        area = int(mask.sum())
        if area < HAND_MIN_AREA:
            continue
        if mask_overlap_ratio(mask, human_mask) < HAND_MIN_HUMAN_OVERLAP:
            continue
        valid_masks.append(mask.copy())
    valid_masks.sort(key=lambda m: int(m.sum()), reverse=True)
    return valid_masks[:max_count]


def get_cached_hand_masks(
    frame_idx: int,
    detection_cache: dict[int, list[np.ndarray]],
    text: str,
    video_dir: Path,
    frame_names: list[str],
    human_masks: list[np.ndarray] | None,
    image_predictor: SAM2ImagePredictor,
    processor: AutoProcessor,
    grounding_model: AutoModelForZeroShotObjectDetection,
    device: str,
) -> list[np.ndarray]:
    if frame_idx not in detection_cache:
        human_mask = None if human_masks is None else human_masks[frame_idx]
        detection_cache[frame_idx] = select_valid_hand_masks(
            detect_masks_on_frame(
                frame_idx=frame_idx,
                text=text,
                video_dir=video_dir,
                frame_names=frame_names,
                image_predictor=image_predictor,
                processor=processor,
                grounding_model=grounding_model,
                device=device,
            ),
            human_mask=human_mask,
        )
    return [mask.copy() for mask in detection_cache[frame_idx]]


def make_scan_frame_indices(num_frames: int, scan_stride: int) -> list[int]:
    frame_indices = list(range(0, num_frames, scan_stride))
    if num_frames > 0 and frame_indices[-1] != num_frames - 1:
        frame_indices.append(num_frames - 1)
    return sorted(set(frame_indices))


def hand_seed_score(masks: list[np.ndarray]) -> tuple[int, int, int, int]:
    if not masks:
        return (0, 0, 0, 0)
    areas = [int(mask.sum()) for mask in masks[:HAND_MAX_TRACKS]]
    return (
        len(areas),
        min(areas) if len(areas) >= 2 else 0,
        sum(areas),
        max(areas),
    )


def select_best_hand_seed_frame(
    text: str,
    video_dir: Path,
    frame_names: list[str],
    human_masks: list[np.ndarray] | None,
    image_predictor: SAM2ImagePredictor,
    processor: AutoProcessor,
    grounding_model: AutoModelForZeroShotObjectDetection,
    device: str,
    scan_stride: int,
) -> tuple[int | None, list[np.ndarray], dict[int, list[np.ndarray]]]:
    detection_cache: dict[int, list[np.ndarray]] = {}
    coarse_frame_indices = make_scan_frame_indices(len(frame_names), scan_stride)
    coarse_scores = []
    for frame_idx in coarse_frame_indices:
        masks = get_cached_hand_masks(
            frame_idx=frame_idx,
            detection_cache=detection_cache,
            text=text,
            video_dir=video_dir,
            frame_names=frame_names,
            human_masks=human_masks,
            image_predictor=image_predictor,
            processor=processor,
            grounding_model=grounding_model,
            device=device,
        )
        score = hand_seed_score(masks)
        if score[0] > 0:
            coarse_scores.append((score, frame_idx))

    if not coarse_scores:
        return None, [], detection_cache

    coarse_scores.sort(reverse=True)
    best_frame_idx = None
    best_masks: list[np.ndarray] = []
    best_score = (0, 0, 0, 0)
    visited = set()
    for _, coarse_frame_idx in coarse_scores[:HAND_ANCHOR_REFINE_TOPK]:
        start = max(0, coarse_frame_idx - scan_stride + 1)
        end = min(len(frame_names), coarse_frame_idx + scan_stride)
        for frame_idx in range(start, end):
            if frame_idx in visited:
                continue
            visited.add(frame_idx)
            masks = get_cached_hand_masks(
                frame_idx=frame_idx,
                detection_cache=detection_cache,
                text=text,
                video_dir=video_dir,
                frame_names=frame_names,
                human_masks=human_masks,
                image_predictor=image_predictor,
                processor=processor,
                grounding_model=grounding_model,
                device=device,
            )
            score = hand_seed_score(masks)
            if score > best_score:
                best_frame_idx = frame_idx
                best_masks = masks
                best_score = score
    return best_frame_idx, best_masks, detection_cache


def select_additional_hand_seed_frame(
    frame_tracks: list[dict[int, np.ndarray]],
    detection_cache: dict[int, list[np.ndarray]],
    text: str,
    video_dir: Path,
    frame_names: list[str],
    human_masks: list[np.ndarray] | None,
    image_predictor: SAM2ImagePredictor,
    processor: AutoProcessor,
    grounding_model: AutoModelForZeroShotObjectDetection,
    device: str,
    scan_stride: int,
) -> tuple[int | None, np.ndarray | None]:
    coarse_frame_indices = make_scan_frame_indices(len(frame_names), scan_stride)
    coarse_scores = []
    for frame_idx in coarse_frame_indices:
        candidate_masks = get_cached_hand_masks(
            frame_idx=frame_idx,
            detection_cache=detection_cache,
            text=text,
            video_dir=video_dir,
            frame_names=frame_names,
            human_masks=human_masks,
            image_predictor=image_predictor,
            processor=processor,
            grounding_model=grounding_model,
            device=device,
        )
        if len(candidate_masks) <= len(frame_tracks[frame_idx]):
            continue
        unmatched_masks = [
            mask
            for mask in candidate_masks
            if not frame_tracks[frame_idx]
            or max(mask_iou(mask, existing_mask) for existing_mask in frame_tracks[frame_idx].values())
            < HAND_TRACK_IOU_THRESHOLD
        ]
        if not unmatched_masks:
            continue
        best_unmatched_mask = max(unmatched_masks, key=lambda mask: int(mask.sum()))
        coarse_scores.append(
            (
                (
                    len(candidate_masks),
                    int(best_unmatched_mask.sum()),
                    sum(int(mask.sum()) for mask in candidate_masks[:HAND_MAX_TRACKS]),
                ),
                frame_idx,
            )
        )

    if not coarse_scores:
        return None, None

    coarse_scores.sort(reverse=True)
    best_frame_idx = None
    best_mask = None
    best_score = (0, 0, 0)
    visited = set()
    for _, coarse_frame_idx in coarse_scores[:HAND_ANCHOR_REFINE_TOPK]:
        start = max(0, coarse_frame_idx - scan_stride + 1)
        end = min(len(frame_names), coarse_frame_idx + scan_stride)
        for frame_idx in range(start, end):
            if frame_idx in visited:
                continue
            visited.add(frame_idx)
            candidate_masks = get_cached_hand_masks(
                frame_idx=frame_idx,
                detection_cache=detection_cache,
                text=text,
                video_dir=video_dir,
                frame_names=frame_names,
                human_masks=human_masks,
                image_predictor=image_predictor,
                processor=processor,
                grounding_model=grounding_model,
                device=device,
            )
            if len(candidate_masks) <= len(frame_tracks[frame_idx]):
                continue
            unmatched_masks = [
                mask
                for mask in candidate_masks
                if not frame_tracks[frame_idx]
                or max(mask_iou(mask, existing_mask) for existing_mask in frame_tracks[frame_idx].values())
                < HAND_TRACK_IOU_THRESHOLD
            ]
            if not unmatched_masks:
                continue
            frame_best_mask = max(unmatched_masks, key=lambda mask: int(mask.sum()))
            score = (
                len(candidate_masks),
                int(frame_best_mask.sum()),
                sum(int(mask.sum()) for mask in candidate_masks[:HAND_MAX_TRACKS]),
            )
            if score > best_score:
                best_frame_idx = frame_idx
                best_mask = frame_best_mask.copy()
                best_score = score
    return best_frame_idx, best_mask


def collect_reference_tracks(
    frame_tracks: list[dict[int, np.ndarray]], frame_idx: int
) -> tuple[dict[int, np.ndarray], bool]:
    if frame_tracks[frame_idx]:
        return copy_track_masks(frame_tracks[frame_idx]), True
    if frame_idx > 0 and frame_tracks[frame_idx - 1]:
        return copy_track_masks(frame_tracks[frame_idx - 1]), False
    return {}, False


def assign_track_ids(
    detection_masks: list[np.ndarray],
    reference_tracks: dict[int, np.ndarray],
    next_track_id: int,
    keep_unmatched_reference: bool,
) -> tuple[dict[int, np.ndarray], int]:
    if not detection_masks:
        return copy_track_masks(reference_tracks), next_track_id

    scored_pairs = []
    for det_idx, det_mask in enumerate(detection_masks):
        for track_id, ref_mask in reference_tracks.items():
            iou = mask_iou(det_mask, ref_mask)
            if iou >= HAND_TRACK_IOU_THRESHOLD:
                scored_pairs.append((iou, det_idx, track_id))
    scored_pairs.sort(reverse=True)

    active_tracks = {}
    assigned_detections = set()
    matched_tracks = set()
    for _, det_idx, track_id in scored_pairs:
        if det_idx in assigned_detections or track_id in matched_tracks:
            continue
        active_tracks[track_id] = detection_masks[det_idx].copy()
        assigned_detections.add(det_idx)
        matched_tracks.add(track_id)

    if keep_unmatched_reference:
        for track_id, ref_mask in reference_tracks.items():
            if track_id not in matched_tracks:
                active_tracks[track_id] = ref_mask.copy()

    for det_idx, det_mask in enumerate(detection_masks):
        if det_idx in assigned_detections:
            continue
        active_tracks[next_track_id] = det_mask.copy()
        next_track_id += 1

    return active_tracks, next_track_id


def propagate_from_seed_masks(
    video_predictor,
    inference_state,
    start_frame_idx: int,
    seed_masks: dict[int, np.ndarray],
    max_frame_num_to_track: int,
    reverse: bool = False,
) -> dict[int, dict[int, np.ndarray]]:
    if not seed_masks:
        return {}

    video_predictor.reset_state(inference_state)
    for track_id, mask in seed_masks.items():
        video_predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=start_frame_idx,
            obj_id=track_id,
            mask=mask,
        )

    segments = {start_frame_idx: copy_track_masks(seed_masks)}
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
        inference_state,
        start_frame_idx=start_frame_idx,
        max_frame_num_to_track=max_frame_num_to_track,
        reverse=reverse,
    ):
        frame_segments = {}
        for i, out_obj_id in enumerate(out_obj_ids):
            out_mask = (out_mask_logits[i] > 0.0).cpu().numpy()
            if out_mask.ndim == 3:
                out_mask = out_mask[0]
            out_mask = out_mask.astype(bool)
            if out_mask.any():
                frame_segments[int(out_obj_id)] = out_mask
        segments[out_frame_idx] = frame_segments
    return segments


def update_frame_tracks(
    frame_tracks: list[dict[int, np.ndarray]],
    segments: dict[int, dict[int, np.ndarray]],
    num_frames: int,
) -> None:
    for frame_idx, track_masks in segments.items():
        if frame_idx < 0 or frame_idx >= num_frames:
            continue
        for track_id, mask in track_masks.items():
            frame_tracks[frame_idx][track_id] = mask.copy()


def load_dilated_human_masks(mask_human_dir: Path, frame_names: list[str]) -> list[np.ndarray] | None:
    if not mask_human_dir.exists():
        return None

    kernel = np.ones((HAND_HUMAN_DILATE_KERNEL, HAND_HUMAN_DILATE_KERNEL), np.uint8)
    dilated_masks = []
    for frame_name in frame_names:
        mask_path = mask_human_dir / frame_name
        if not mask_path.exists():
            return None
        human_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if human_mask is None:
            return None
        dilated = cv2.dilate(human_mask, kernel, iterations=1)
        dilated_masks.append(dilated > 127)
    return dilated_masks


def compute_track_stats(
    frame_tracks: list[dict[int, np.ndarray]], human_masks: list[np.ndarray] | None
) -> dict[int, dict[str, float]]:
    track_stats: dict[int, dict[str, float]] = {}
    for frame_idx, tracks in enumerate(frame_tracks):
        human_mask = None if human_masks is None else human_masks[frame_idx]
        for track_id, mask in tracks.items():
            area = int(mask.sum())
            if area == 0:
                continue
            stats = track_stats.setdefault(
                track_id,
                {
                    "first_seen": frame_idx,
                    "last_seen": frame_idx,
                    "frame_count": 0,
                    "total_area": 0.0,
                    "max_area": 0.0,
                    "total_overlap": 0.0,
                },
            )
            stats["first_seen"] = min(stats["first_seen"], frame_idx)
            stats["last_seen"] = max(stats["last_seen"], frame_idx)
            stats["frame_count"] += 1
            stats["total_area"] += area
            stats["max_area"] = max(stats["max_area"], area)
            stats["total_overlap"] += mask_overlap_ratio(mask, human_mask)

    for stats in track_stats.values():
        frame_count = max(1, int(stats["frame_count"]))
        stats["mean_overlap_with_dilated_human"] = stats["total_overlap"] / frame_count
    return track_stats


def select_hand_track_ids(track_stats: dict[int, dict[str, float]]) -> list[int]:
    def sort_key(track_id: int):
        stats = track_stats[track_id]
        return (
            stats["frame_count"],
            stats["total_area"],
            stats["max_area"],
        )

    strict_candidates = [
        track_id
        for track_id, stats in track_stats.items()
        if stats["frame_count"] >= HAND_MIN_FRAMES
        and stats["max_area"] >= HAND_MIN_AREA
        and stats["mean_overlap_with_dilated_human"] >= HAND_MIN_HUMAN_OVERLAP
    ]
    if strict_candidates:
        strict_candidates.sort(key=sort_key, reverse=True)
        return strict_candidates[:HAND_MAX_TRACKS]

    relaxed_candidates = [
        track_id
        for track_id, stats in track_stats.items()
        if stats["frame_count"] >= HAND_MIN_FRAMES and stats["max_area"] >= HAND_MIN_AREA
    ]
    relaxed_candidates.sort(key=sort_key, reverse=True)
    return relaxed_candidates[:HAND_MAX_TRACKS]


def backfill_hand_track(
    track_id: int,
    frame_tracks: list[dict[int, np.ndarray]],
    track_stats: dict[int, dict[str, float]],
    human_masks: list[np.ndarray] | None,
    tracking_window: int,
    video_predictor,
    inference_state,
) -> None:
    start_frame_idx = int(track_stats[track_id]["first_seen"])
    invalid_streak = 0
    while start_frame_idx > 0:
        seed_mask = frame_tracks[start_frame_idx].get(track_id)
        if seed_mask is None:
            break

        max_track = min(tracking_window + 1, start_frame_idx + 1)
        reverse_segments = propagate_from_seed_masks(
            video_predictor=video_predictor,
            inference_state=inference_state,
            start_frame_idx=start_frame_idx,
            seed_masks={track_id: seed_mask},
            max_frame_num_to_track=max_track,
            reverse=True,
        )
        earlier_frames = sorted(
            (frame_idx for frame_idx in reverse_segments if frame_idx < start_frame_idx),
            reverse=True,
        )
        if not earlier_frames:
            break

        earliest_valid_frame = None
        for frame_idx in earlier_frames:
            mask = reverse_segments[frame_idx].get(track_id)
            if mask is None:
                invalid_streak += 1
            else:
                human_mask = None if human_masks is None else human_masks[frame_idx]
                is_valid = (
                    mask.sum() >= HAND_MIN_AREA
                    and mask_overlap_ratio(mask, human_mask) >= HAND_MIN_HUMAN_OVERLAP
                )
                if is_valid:
                    frame_tracks[frame_idx][track_id] = mask.copy()
                    earliest_valid_frame = frame_idx
                    invalid_streak = 0
                else:
                    invalid_streak += 1
            if invalid_streak >= HAND_BACKFILL_INVALID_STREAK:
                return

        if earliest_valid_frame is None:
            break
        start_frame_idx = earliest_valid_frame


def save_masks_and_visualizations(
    frame_names: list[str],
    video_dir: Path,
    video_segments: list[dict[int, np.ndarray]],
    mask_vis_dir: Path,
    output_mask_dir: Path,
    label_name: str,
) -> None:
    for frame_idx, frame_name in enumerate(frame_names):
        image_path = video_dir / frame_name
        img = cv2.imread(str(image_path))
        segments = video_segments[frame_idx]
        if segments:
            object_ids = sorted(segments)
            masks = np.stack([segments[obj_id] for obj_id in object_ids], axis=0)
            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks),
                mask=masks,
                class_id=np.array(object_ids, dtype=np.int32),
            )
            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
            label_annotator = sv.LabelAnnotator()
            annotated_frame = label_annotator.annotate(
                annotated_frame,
                detections=detections,
                labels=[f"{label_name}_{obj_id}" for obj_id in object_ids],
            )
            mask_annotator = sv.MaskAnnotator()
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            mask_all = np.any(masks, axis=0)
        else:
            annotated_frame = img
            mask_all = np.zeros(img.shape[:2], dtype=bool)

        cv2.imwrite(str(mask_vis_dir / f"{frame_idx:04d}.jpg"), annotated_frame)
        cv2.imwrite(str(output_mask_dir / f"{frame_idx:04d}.jpg"), mask_all.astype(np.uint8) * 255)


def run_default_tracking(
    text: str,
    video_dir: Path,
    frame_names: list[str],
    video_predictor,
    image_predictor,
    processor,
    grounding_model,
    device: str,
) -> list[dict[int, np.ndarray]]:
    detection_masks = detect_masks_on_frame(
        frame_idx=0,
        text=text,
        video_dir=video_dir,
        frame_names=frame_names,
        image_predictor=image_predictor,
        processor=processor,
        grounding_model=grounding_model,
        device=device,
    )
    if not detection_masks:
        return [dict() for _ in frame_names]

    seed_masks = {object_id: mask for object_id, mask in enumerate(detection_masks, start=1)}
    inference_state = video_predictor.init_state(video_path=str(video_dir))
    video_segments = [dict() for _ in frame_names]

    reverse_segments = propagate_from_seed_masks(
        video_predictor=video_predictor,
        inference_state=inference_state,
        start_frame_idx=0,
        seed_masks=seed_masks,
        max_frame_num_to_track=len(frame_names),
        reverse=True,
    )
    update_frame_tracks(video_segments, reverse_segments, len(frame_names))

    forward_segments = propagate_from_seed_masks(
        video_predictor=video_predictor,
        inference_state=inference_state,
        start_frame_idx=0,
        seed_masks=seed_masks,
        max_frame_num_to_track=len(frame_names),
        reverse=False,
    )
    update_frame_tracks(video_segments, forward_segments, len(frame_names))
    return video_segments


def propagate_hand_seed_masks(
    frame_tracks: list[dict[int, np.ndarray]],
    seed_frame_idx: int,
    seed_masks: dict[int, np.ndarray],
    num_frames: int,
    video_predictor,
    inference_state,
) -> None:
    reverse_segments = propagate_from_seed_masks(
        video_predictor=video_predictor,
        inference_state=inference_state,
        start_frame_idx=seed_frame_idx,
        seed_masks=seed_masks,
        max_frame_num_to_track=seed_frame_idx + 1,
        reverse=True,
    )
    update_frame_tracks(frame_tracks, reverse_segments, num_frames)

    forward_segments = propagate_from_seed_masks(
        video_predictor=video_predictor,
        inference_state=inference_state,
        start_frame_idx=seed_frame_idx,
        seed_masks=seed_masks,
        max_frame_num_to_track=num_frames - seed_frame_idx,
        reverse=False,
    )
    update_frame_tracks(frame_tracks, forward_segments, num_frames)


def _frame_hand_area(tracks: dict[int, np.ndarray]) -> int:
    return sum(int(mask.sum()) for mask in tracks.values())


def find_dropout_runs(frame_tracks: list[dict[int, np.ndarray]], min_run: int) -> list[tuple[int, int]]:
    """Maximal inclusive runs of frames whose total hand-mask area is below HAND_MIN_AREA."""
    runs = []
    start = None
    for idx, tracks in enumerate(frame_tracks):
        if _frame_hand_area(tracks) < HAND_MIN_AREA:
            if start is None:
                start = idx
        elif start is not None:
            if idx - start >= min_run:
                runs.append((start, idx - 1))
            start = None
    if start is not None and len(frame_tracks) - start >= min_run:
        runs.append((start, len(frame_tracks) - 1))
    return runs


def recover_hand_dropouts(
    frame_tracks: list[dict[int, np.ndarray]],
    detection_cache: dict[int, list[np.ndarray]],
    text: str,
    video_dir: Path,
    frame_names: list[str],
    human_masks: list[np.ndarray] | None,
    image_predictor,
    processor,
    grounding_model,
    device: str,
    video_predictor,
    inference_state,
) -> list[dict[int, np.ndarray]]:
    """Re-seed and re-propagate through runs of frames where tracking lost the hands.

    SAM2 video propagation can drop small or fast-moving hands mid-sequence and
    the seed selection never revisits those frames. For each empty run, probe a
    few frames inside it with Grounding DINO, propagate the first valid
    detection across the run in both directions, and fill only frames that are
    still empty (the saved hand mask is the per-frame union of tracks, so
    track-id collisions with retained tracks are harmless there).
    """
    for _ in range(HAND_RECOVERY_MAX_ROUNDS):
        runs = find_dropout_runs(frame_tracks, HAND_RECOVERY_MIN_RUN)
        if not runs:
            break
        runs.sort(key=lambda r: r[1] - r[0], reverse=True)
        recovered = False
        for start, end in runs:
            span = end - start
            probe_idxs = sorted({start + span // 2, start, end, start + span // 4, start + 3 * span // 4})
            for probe_idx in probe_idxs:
                seeds = get_cached_hand_masks(
                    frame_idx=probe_idx,
                    detection_cache=detection_cache,
                    text=text,
                    video_dir=video_dir,
                    frame_names=frame_names,
                    human_masks=human_masks,
                    image_predictor=image_predictor,
                    processor=processor,
                    grounding_model=grounding_model,
                    device=device,
                )
                if not seeds:
                    continue
                seed_masks = {tid: m for tid, m in enumerate(seeds[:HAND_MAX_TRACKS], start=1)}
                for reverse, span_limit in ((False, end - probe_idx + 1), (True, probe_idx - start + 1)):
                    segments = propagate_from_seed_masks(
                        video_predictor,
                        inference_state,
                        probe_idx,
                        seed_masks,
                        max_frame_num_to_track=span_limit,
                        reverse=reverse,
                    )
                    for frame_idx, track_masks in segments.items():
                        if not (start <= frame_idx <= end):
                            continue
                        if _frame_hand_area(frame_tracks[frame_idx]) >= HAND_MIN_AREA:
                            continue
                        human_mask = None if human_masks is None else human_masks[frame_idx]
                        for track_id, mask in track_masks.items():
                            if int(mask.sum()) < HAND_MIN_AREA:
                                continue
                            if mask_overlap_ratio(mask, human_mask) < HAND_MIN_HUMAN_OVERLAP:
                                continue
                            frame_tracks[frame_idx][track_id] = mask.copy()
                print(f"  Hand tracking: re-seeded dropout run [{start}, {end}] from frame {probe_idx}")
                recovered = True
                break
            if recovered:
                break
        if not recovered:
            break
    remaining = find_dropout_runs(frame_tracks, HAND_RECOVERY_MIN_RUN)
    if remaining:
        print(f"  Hand tracking: {len(remaining)} dropout run(s) not recoverable: {remaining}")
    return frame_tracks


def run_hand_tracking(
    text: str,
    video_dir: Path,
    frame_names: list[str],
    mask_human_dir: Path,
    video_predictor,
    image_predictor,
    processor,
    grounding_model,
    device: str,
    tracking_window: int,
) -> list[dict[int, np.ndarray]]:
    # Automatically select strong hand seed frames, then propagate bidirectionally from them.
    num_frames = len(frame_names)
    frame_tracks = [dict() for _ in frame_names]
    human_masks = load_dilated_human_masks(mask_human_dir, frame_names)
    if human_masks is None:
        print("  Hand tracking: human masks unavailable, disabling overlap-based filtering.")
    inference_state = video_predictor.init_state(video_path=str(video_dir))
    scan_stride = max(1, tracking_window)

    seed_frame_idx, seed_masks, detection_cache = select_best_hand_seed_frame(
        text=text,
        video_dir=video_dir,
        frame_names=frame_names,
        human_masks=human_masks,
        image_predictor=image_predictor,
        processor=processor,
        grounding_model=grounding_model,
        device=device,
        scan_stride=scan_stride,
    )
    if seed_frame_idx is None or not seed_masks:
        print("  Hand tracking: no valid automatic seed frame found.")
        return frame_tracks

    initial_seed_masks = {
        track_id: mask for track_id, mask in enumerate(seed_masks[:HAND_MAX_TRACKS], start=1)
    }
    print(
        f"  Hand tracking: automatic seed frame {seed_frame_idx} with {len(initial_seed_masks)} hand(s)"
    )
    propagate_hand_seed_masks(
        frame_tracks=frame_tracks,
        seed_frame_idx=seed_frame_idx,
        seed_masks=initial_seed_masks,
        num_frames=num_frames,
        video_predictor=video_predictor,
        inference_state=inference_state,
    )

    next_track_id = len(initial_seed_masks) + 1
    while next_track_id <= HAND_MAX_TRACKS:
        extra_seed_frame_idx, extra_seed_mask = select_additional_hand_seed_frame(
            frame_tracks=frame_tracks,
            detection_cache=detection_cache,
            text=text,
            video_dir=video_dir,
            frame_names=frame_names,
            human_masks=human_masks,
            image_predictor=image_predictor,
            processor=processor,
            grounding_model=grounding_model,
            device=device,
            scan_stride=scan_stride,
        )
        if extra_seed_frame_idx is None or extra_seed_mask is None:
            break
        print(
            f"  Hand tracking: supplemental seed frame {extra_seed_frame_idx} for track {next_track_id}"
        )
        propagate_hand_seed_masks(
            frame_tracks=frame_tracks,
            seed_frame_idx=extra_seed_frame_idx,
            seed_masks={next_track_id: extra_seed_mask},
            num_frames=num_frames,
            video_predictor=video_predictor,
            inference_state=inference_state,
        )
        next_track_id += 1

    track_stats = compute_track_stats(frame_tracks, human_masks)
    retained_track_ids = select_hand_track_ids(track_stats)
    print(f"  Hand tracking: retained track ids {retained_track_ids}")

    filtered_tracks = [dict() for _ in frame_names]
    retained_track_id_set = set(retained_track_ids)
    for frame_idx, tracks in enumerate(frame_tracks):
        for track_id, mask in tracks.items():
            if track_id in retained_track_id_set:
                filtered_tracks[frame_idx][track_id] = mask

    filtered_tracks = recover_hand_dropouts(
        frame_tracks=filtered_tracks,
        detection_cache=detection_cache,
        text=text,
        video_dir=video_dir,
        frame_names=frame_names,
        human_masks=human_masks,
        image_predictor=image_predictor,
        processor=processor,
        grounding_model=grounding_model,
        device=device,
        video_predictor=video_predictor,
        inference_state=inference_state,
    )
    return filtered_tracks


@dataclass
class CreateMasksArgs:
    data_dir: Path
    """Path to the sequence data directory."""
    text: str = "human."
    """Text queries need to be lowercased + end with a dot."""
    is_object: bool = False
    """Create object masks instead of human masks."""
    is_hand: bool = False
    """Create hand masks with tracking."""
    bg_color: List[int] = field(default_factory=lambda: [255, 255, 255])
    """Background color for object images."""
    tracking_window: int = 10
    """For hand masks, stride used to scan candidate seed frames."""
    tools: ToolConfig = field(default_factory=ToolConfig)


def main():
    import tyro
    args = tyro.cli(CreateMasksArgs)

    data_dir = args.data_dir
    text = args.text
    is_object = args.is_object
    is_hand = args.is_hand
    bg_color = args.bg_color
    tracking_window = max(1, args.tracking_window)

    mask_dir = data_dir / "masks" / "sam2"
    mask_colmap_dir = data_dir / "masks" / "colmap"
    mask_obj_dir = data_dir / "masks" / "sam2_obj"
    mask_human_dir = data_dir / "masks" / "sam2_human"
    mask_hand_dir = data_dir / "masks" / "sam2_hand"
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_colmap_dir.mkdir(exist_ok=True)
    mask_obj_dir.mkdir(exist_ok=True)
    mask_human_dir.mkdir(exist_ok=True)
    mask_hand_dir.mkdir(exist_ok=True)

    if is_object:
        print("Generating object masks...")
        mask_vis_dir = data_dir / "masks" / "sam2_vis_obj"
        output_mask_dir = mask_obj_dir
    elif is_hand:
        print("Generating hand masks...")
        mask_vis_dir = data_dir / "masks" / "sam2_vis_hand"
        output_mask_dir = mask_hand_dir
    else:
        print("Generating human masks...")
        mask_vis_dir = data_dir / "masks" / "sam2_vis_human"
        output_mask_dir = mask_human_dir
    mask_vis_dir.mkdir(exist_ok=True)

    video_dir = data_dir / "images" / "selected_frames"
    frame_names = get_frame_names(video_dir)
    device, video_predictor, image_predictor, processor, grounding_model = build_models(args.tools)

    if is_hand:
        video_segments = run_hand_tracking(
            text=text,
            video_dir=video_dir,
            frame_names=frame_names,
            mask_human_dir=mask_human_dir,
            video_predictor=video_predictor,
            image_predictor=image_predictor,
            processor=processor,
            grounding_model=grounding_model,
            device=device,
            tracking_window=tracking_window,
        )
    else:
        video_segments = run_default_tracking(
            text=text,
            video_dir=video_dir,
            frame_names=frame_names,
            video_predictor=video_predictor,
            image_predictor=image_predictor,
            processor=processor,
            grounding_model=grounding_model,
            device=device,
        )

    save_masks_and_visualizations(
        frame_names=frame_names,
        video_dir=video_dir,
        video_segments=video_segments,
        mask_vis_dir=mask_vis_dir,
        output_mask_dir=output_mask_dir,
        label_name=text.rstrip("."),
    )

    if is_object:
        create_colmap_mask(mask_obj_dir, mask_human_dir, mask_dir, mask_colmap_dir)
        create_obj_images(data_dir, bg_color)


if __name__ == "__main__":
    main()
