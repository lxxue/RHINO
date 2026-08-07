"""Shared helpers for the comparison-video scripts."""

from pathlib import Path

import cv2
import ffmpeg


def draw_label(
    img,
    text: str,
    *,
    org: tuple = (20, 40),
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    scale: float = 1.5,
    thickness: int = 3,
    outline: int = 0,
):
    """Draw `text` on `img` in place; `outline > 0` adds a white underlay."""
    if outline > 0:
        cv2.putText(img, text, org, font, scale, (255, 255, 255), outline)
    cv2.putText(img, text, org, font, scale, (0, 0, 0), thickness)
    return img


def encode_video(frames_dir: Path, output_path: Path, width: int, height: int, fps: int = 30):
    """Encode `frames_dir`/%04d.jpg into an H.264 mp4 at `output_path`."""
    stream = ffmpeg.input(
        str(frames_dir / "%04d.jpg"),
        format="image2",
        pix_fmt="rgb24",
        s=f"{width}x{height}",
        framerate=fps,
    )
    stream = ffmpeg.output(
        stream, str(output_path), vcodec="libx264", pix_fmt="yuv420p", r=fps
    )
    stream = ffmpeg.overwrite_output(stream)
    _, err = ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
    if err:
        print("FFmpeg stderr:\n", err.decode("utf8"))
    print(f"Video saved to: {output_path}")
