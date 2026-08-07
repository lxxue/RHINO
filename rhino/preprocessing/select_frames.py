"""Step 0: select the sharpest source frame from each temporal window."""

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tyro
from tqdm import tqdm


def compute_variance_of_laplacian(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()


def get_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return compute_variance_of_laplacian(gray)


def extract_number(filename):
    match = re.search(r"(\d+)", filename)
    return int(match.group()) if match else -1


@dataclass
class SelectFramesArgs:
    data_dir: Path
    """Path to the sequence data directory."""
    input_path: Path
    """Input video file or image directory."""
    window_size: int = 10
    """Select the sharpest frame in every window of this size."""
    frame_start: int = 0
    """First source frame index to consider."""
    frame_end: int = 1000000
    """Last source frame index to consider (inclusive)."""
    video_resize_factor: int = 1
    """Integer factor used to resize input frames."""


def _resize_frame(image: np.ndarray, factor: int, source: Path) -> np.ndarray:
    if factor == 1:
        return image
    width = image.shape[1] // factor
    height = image.shape[0] // factor
    if width < 1 or height < 1:
        raise ValueError(
            f"Resize factor {factor} makes {source} smaller than one pixel"
        )
    return cv2.resize(image, (width, height))


def _read_image(path: Path, resize_factor: int) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return _resize_frame(image, resize_factor, path)


def _copy_or_convert_video(input_path: Path, destination: Path) -> None:
    suffix = input_path.suffix.lower()
    if suffix == ".mp4":
        shutil.copy2(input_path, destination)
    elif suffix == ".mov":
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(input_path),
                "-c:v",
                "copy",
                "-c:a",
                "copy",
                str(destination),
            ],
            check=True,
        )
    else:
        raise ValueError(f"Unsupported video format: {input_path.suffix}")


def _load_source_frames(
    input_path: Path, image_dir: Path, resize_factor: int
) -> tuple[list[np.ndarray], list[int]]:
    if input_path.is_dir():
        image_pattern = re.compile(r".*?(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)
        image_files = [path for path in input_path.iterdir() if image_pattern.fullmatch(path.name)]
        image_files.sort(key=lambda path: extract_number(path.name))
        if not image_files:
            raise ValueError(f"No numbered JPG/PNG images found in {input_path}")
        source_ids = [extract_number(path.name) for path in image_files]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError(f"Input image frame IDs are not unique in {input_path}")

        metadata_path = input_path.parent / "metadata.json"
        if metadata_path.is_file():
            print("Copying Record3D metadata.json")
            shutil.copy2(metadata_path, image_dir / "metadata.json")
        frames = [
            _read_image(path, resize_factor)
            for path in tqdm(image_files, desc="Reading images")
        ]
        return frames, source_ids

    if not input_path.is_file():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    video_path = image_dir / "input.mp4"
    _copy_or_convert_video(input_path, video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        raise ValueError(f"Video reports no frames: {video_path}")

    frames = []
    try:
        for frame_id in tqdm(range(frame_count), desc="Reading video"):
            success, image = capture.read()
            if not success or image is None:
                raise ValueError(f"Failed to read video frame {frame_id} from {video_path}")
            frames.append(_resize_frame(image, resize_factor, video_path))
    finally:
        capture.release()
    return frames, list(range(frame_count))


def _write_jpeg(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to write image: {path}")


def select_frames(args: SelectFramesArgs) -> np.ndarray:
    if args.window_size <= 0:
        raise ValueError(f"window_size must be positive, got {args.window_size}")
    if args.video_resize_factor <= 0:
        raise ValueError(
            f"video_resize_factor must be positive, got {args.video_resize_factor}"
        )
    if args.frame_start < 0 or args.frame_end < args.frame_start:
        raise ValueError(
            f"Invalid inclusive frame range [{args.frame_start}, {args.frame_end}]"
        )

    data_dir = args.data_dir.resolve()
    input_path = args.input_path.resolve()
    image_dir = data_dir / "images"
    all_frames_dir = image_dir / "all_frames"
    selected_frames_dir = image_dir / "selected_frames"
    all_frames_dir.mkdir(parents=True, exist_ok=True)
    selected_frames_dir.mkdir(parents=True, exist_ok=True)

    frames, source_ids = _load_source_frames(
        input_path, image_dir, args.video_resize_factor
    )
    if args.frame_start >= len(frames):
        raise ValueError(
            f"frame_start {args.frame_start} exceeds the last source position {len(frames) - 1}"
        )
    stop = min(args.frame_end + 1, len(frames))
    frames = frames[args.frame_start:stop]
    source_ids = source_ids[args.frame_start:stop]
    sharpness = [get_sharpness(image) for image in frames]

    selected_frames = []
    selected_source_ids = []
    for window_start in range(0, len(frames), args.window_size):
        window_sharpness = sharpness[window_start : window_start + args.window_size]
        local_index = int(np.argmax(window_sharpness))
        selected_index = window_start + local_index
        selected_frames.append(frames[selected_index])
        selected_source_ids.append(source_ids[selected_index])

    for output_dir in (all_frames_dir, selected_frames_dir):
        for stale_frame in output_dir.glob("*.jpg"):
            stale_frame.unlink()
    for index, image in enumerate(frames):
        _write_jpeg(all_frames_dir / f"{index:04d}.jpg", image)
    for index, image in enumerate(selected_frames):
        _write_jpeg(selected_frames_dir / f"{index:04d}.jpg", image)

    selected_ids = np.asarray(selected_source_ids, dtype=np.int64)
    np.save(image_dir / "selected_idxs.npy", selected_ids)
    print(f"Extracted {len(selected_frames)} selected frames from {len(frames)} frames")
    return selected_ids


def main(args: SelectFramesArgs) -> None:
    print("Selecting frames...")
    select_frames(args)


if __name__ == "__main__":
    main(tyro.cli(SelectFramesArgs))
