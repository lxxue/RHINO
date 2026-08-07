"""Step 5: Assemble base processed/ directory.

Reads camera reconstruction from COLMAP/hloc/MASt3R output, copies
images, masks, depth, normals, and saves intrinsic.npy + c2ws.npy.
No cameras_normalize.npz (deprecated).
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import glob
import numpy as np
from tqdm import tqdm

from rhino.preprocessing.align_utils import (
    read_intrinsics_binary,
    read_extrinsics_binary,
    load_intrinsic,
    load_extrinsics,
)


@dataclass
class PrepareBaseArgs:
    data_dir: Path
    """Path to the sequence data directory."""
    camera_pose_dir: Path
    """Path to COLMAP sparse reconstruction directory (containing cameras.bin, images.bin)."""
    resize_factor: int = 1
    """Resize factor for images/masks/depth/normals (1=full, 2=half, 4=quarter)."""
    use_text_format: bool = False
    """Use COLMAP text format (cameras.txt, images.txt) instead of binary."""


def main(args: PrepareBaseArgs) -> None:
    data_dir = args.data_dir.resolve()
    camera_pose_dir = args.camera_pose_dir.resolve()
    resize_factor = args.resize_factor
    print("Preparing base processed/ directory...")

    # Source directories
    img_dir = data_dir / "images" / "selected_frames"
    depth_dir = data_dir / "monocular_geometry" / "depth"
    normal_dir = data_dir / "monocular_geometry" / "normal"
    human_mask_dir = data_dir / "masks" / "sam2_human"
    obj_mask_dir = data_dir / "masks" / "sam2_obj"
    hand_mask_dir = data_dir / "masks" / "sam2_hand"
    all_mask_dir = data_dir / "masks" / "sam2"

    # Output directory
    save_dir = data_dir / "processed"
    save_dir.mkdir(exist_ok=True)
    for subdir in ["image", "depth", "normal", "human_mask", "obj_mask", "hand_mask", "all_mask"]:
        (save_dir / subdir).mkdir(exist_ok=True)

    # Load camera data from reconstruction
    if args.use_text_format:
        intrinsic = load_intrinsic(camera_pose_dir)
        c2ws_dict, _ = load_extrinsics(camera_pose_dir)
    else:
        intrinsic = read_intrinsics_binary(camera_pose_dir)
        c2ws_dict = read_extrinsics_binary(camera_pose_dir)

    # Apply resize factor to intrinsic
    intrinsic[0, 0] /= resize_factor
    intrinsic[1, 1] /= resize_factor
    intrinsic[0, 2] /= resize_factor
    intrinsic[1, 2] /= resize_factor

    # Gather source files
    img_paths = sorted(glob.glob(f"{img_dir}/*.[pj][np]g"))
    depth_paths = sorted(glob.glob(f"{depth_dir}/*.npy"))
    normal_paths = sorted(glob.glob(f"{normal_dir}/*.npy"))
    human_mask_paths = sorted(glob.glob(f"{human_mask_dir}/*.[pj][np]g"))
    obj_mask_paths = sorted(glob.glob(f"{obj_mask_dir}/*.[pj][np]g"))
    hand_mask_paths = sorted(glob.glob(f"{hand_mask_dir}/*.[pj][np]g"))
    all_mask_paths = sorted(glob.glob(f"{all_mask_dir}/*.[pj][np]g"))

    num_frames = len(img_paths)
    assert num_frames > 0, f"No images found in {img_dir}"
    assert num_frames == len(depth_paths), f"Image/depth count mismatch: {num_frames} vs {len(depth_paths)}"
    assert num_frames == len(normal_paths), f"Image/normal count mismatch: {num_frames} vs {len(normal_paths)}"

    c2w_list = []
    for idx, img_path in enumerate(tqdm(img_paths, desc="Copying to processed/")):
        fname = Path(img_path).stem
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        new_h, new_w = h // resize_factor, w // resize_factor

        if resize_factor > 1:
            img = cv2.resize(img, (new_w, new_h))
        cv2.imwrite(str(save_dir / "image" / f"{idx:04d}.png"), img)

        # Depth
        depth = np.load(depth_paths[idx])
        if resize_factor > 1:
            depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        np.save(save_dir / "depth" / f"{idx:04d}.npy", depth)

        # Normal
        normal = np.load(normal_paths[idx])
        if resize_factor > 1:
            normal = cv2.resize(normal, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        np.save(save_dir / "normal" / f"{idx:04d}.npy", normal)

        # Masks
        for mask_paths, mask_subdir in [
            (human_mask_paths, "human_mask"),
            (obj_mask_paths, "obj_mask"),
            (hand_mask_paths, "hand_mask"),
            (all_mask_paths, "all_mask"),
        ]:
            if idx < len(mask_paths):
                mask = cv2.imread(mask_paths[idx])
                if resize_factor > 1:
                    mask = cv2.resize(mask, (new_w, new_h))
                cv2.imwrite(str(save_dir / mask_subdir / f"{idx:04d}.png"), mask)

        # Camera pose lookup
        for ext in [".png", ".jpg"]:
            key = f"{fname}{ext}"
            if key in c2ws_dict:
                c2w_list.append(c2ws_dict[key])
                break
        else:
            # Try numeric lookup
            for ext in [".png", ".jpg"]:
                key = f"{idx:04d}{ext}"
                if key in c2ws_dict:
                    c2w_list.append(c2ws_dict[key])
                    break
            else:
                raise RuntimeError(f"No camera pose found for frame {idx} ({fname})")

    # Save camera data
    c2ws = np.array(c2w_list, dtype=np.float32)
    np.save(save_dir / "intrinsic.npy", intrinsic.astype(np.float32))
    np.save(save_dir / "c2ws.npy", c2ws)
    print(f"Done. {num_frames} frames written to {save_dir}")
    print(f"  intrinsic.npy: {intrinsic.shape}")
    print(f"  c2ws.npy: {c2ws.shape}")


if __name__ == "__main__":
    import tyro
    main(tyro.cli(PrepareBaseArgs))
