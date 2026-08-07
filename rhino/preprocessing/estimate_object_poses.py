from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pycolmap
from hloc import (
    extract_features,
    match_dense,
    match_features,
    pairs_from_exhaustive,
    pairs_from_retrieval,
    pairs_from_sequence,
    reconstruction,
)
from hloc.utils.read_write_model import read_cameras_binary

from rhino.preprocessing.path_utils import resolve_camera_pose_dir

def make_image_reader_options(camera_model: str, camera_pose_dir: Path) -> pycolmap.ImageReaderOptions:
    cameras = read_cameras_binary(camera_pose_dir / "cameras.bin")
    if len(cameras) != 1:
        raise ValueError(f"Expected a single scene camera in {camera_pose_dir}, found {len(cameras)}.")

    scene_camera = next(iter(cameras.values()))
    if scene_camera.model == "SIMPLE_PINHOLE":
        fx = fy = float(scene_camera.params[0])
        cx = float(scene_camera.params[1])
        cy = float(scene_camera.params[2])
    elif scene_camera.model == "PINHOLE":
        fx = float(scene_camera.params[0])
        fy = float(scene_camera.params[1])
        cx = float(scene_camera.params[2])
        cy = float(scene_camera.params[3])
    else:
        raise ValueError(
            f"Unsupported scene camera model {scene_camera.model}; expected SIMPLE_PINHOLE or PINHOLE."
        )

    if camera_model == "SIMPLE_PINHOLE":
        focal_length = 0.5 * (fx + fy)
        camera_params = [focal_length, cx, cy]
    elif camera_model == "PINHOLE":
        camera_params = [fx, fy, cx, cy]
    else:
        raise ValueError(f"Unsupported object camera model {camera_model}.")

    camera_params_str = ",".join(str(param) for param in camera_params)
    return pycolmap.ImageReaderOptions(
        camera_model=camera_model,
        camera_params=camera_params_str,
        single_camera=True,
    )


# mostly from nerfstudio
def run_hloc(
    data_dir,
    camera_model,
    feature_type,
    matching_method,
    matcher_type,
    num_matched,
    camera_pose_dir,
):
    hloc_dir = data_dir / "object_poses" / f"hloc-{feature_type}-{matching_method}-{matcher_type}"
    image_dir = data_dir / "images" / "objects"
    mask_dir = data_dir / "masks" / "colmap_obj"
    pairs = hloc_dir / f"pairs.txt"
    features = hloc_dir / "features.h5"
    matches = hloc_dir / "matches.h5"

    hloc_dir.mkdir(parents=True, exist_ok=True)
    if "loftr" not in matcher_type:
        feature_conf = extract_features.confs[feature_type]
        matcher_conf = match_features.confs[matcher_type]
    else:
        print("LoFTR does not need feature extraction")
        feature_conf = {}
        matcher_conf = match_dense.confs[matcher_type]
    retrieval_conf = extract_features.confs["netvlad"]

    references = [str(p.relative_to(image_dir).as_posix()) for p in image_dir.iterdir()]
    references = sorted(references)
    if matcher_type != "loftr":
        extract_features.main(
            feature_conf, image_dir, image_list=references, feature_path=features, mask_dir=mask_dir
        )
    if matching_method == "exhaustive":
        pairs_from_exhaustive.main(pairs, image_list=references)
    elif matching_method == "retrieval":
        retrieval_path = extract_features.main(retrieval_conf, image_dir, hloc_dir)
        if num_matched >= len(references):
            num_matched = len(references)
        pairs_from_retrieval.main(retrieval_path, pairs, num_matched=num_matched)
    elif matching_method == "sequential":
        pairs_from_sequence.main(pairs, image_list=references, window_size=30)
    else:
        raise NotImplementedError
    if matcher_type != "loftr":
        match_features.main(matcher_conf, pairs, features=features, matches=matches)
    else:
        match_dense.main(
            matcher_conf,
            pairs,
            image_dir=image_dir,
            export_dir=hloc_dir,
            max_kps=8192,
            overwrite=False,
            features=features,
            matches=matches,
        )

    image_options = make_image_reader_options(camera_model, camera_pose_dir)

    sfm_dir = hloc_dir
    sfm_dir.mkdir(parents=True, exist_ok=True)

    reconstruction.main(
        sfm_dir,
        image_dir,
        pairs,
        features,
        matches,
        camera_mode=pycolmap.CameraMode.SINGLE,
        image_options=image_options,
        mapper_options={
            "ba_refine_focal_length": False,
            "ba_refine_principal_point": False,
            "ba_refine_extra_params": False,
        },
    )
    return


@dataclass
class EstimateObjectPosesArgs:
    data_dir: Path
    camera_pose_dir: Optional[Path] = None
    """Path to the scene camera reconstruction used to fix object intrinsics."""
    camera_model: str = "SIMPLE_PINHOLE"
    sfm_tool: str = "hloc"
    feature_type: str = "superpoint_aachen"
    matching_method: Optional[str] = None
    matcher_type: str = "superpoint+lightglue"
    camera_use_mast3r: bool = False
    camera_use_glomap: bool = False
    camera_pair_window_size: int = 5
    record3d_num_ba_iterations: int = 3


if __name__ == "__main__":
    import tyro

    print("Estimating object poses...")

    # combos:
    #   colmap + sift + {exhaustive, vocab_tree, sequential}
    #   hloc + sift + NN-ratio
    #   hloc + superpoint_aachen + superglue
    #   hloc + superpoint_aachen + lightglue
    #   hloc + r2d2 + NN-ratio  (for synthetic data)

    args = tyro.cli(EstimateObjectPosesArgs)
    data_dir = args.data_dir.resolve()
    matching_method = args.matching_method or (
        "poses" if args.sfm_tool == "record3d" else "sequential"
    )
    camera_pose_dir = args.camera_pose_dir
    if camera_pose_dir is None:
        camera_pose_dir = resolve_camera_pose_dir(
            data_dir,
            args.sfm_tool,
            args.feature_type,
            matching_method,
            args.matcher_type,
            args.camera_use_mast3r,
            args.camera_use_glomap,
            args.camera_pair_window_size,
            args.record3d_num_ba_iterations,
        )
    else:
        camera_pose_dir = camera_pose_dir.resolve()
    sfm_tool = args.sfm_tool
    camera_model = args.camera_model
    feature_type = args.feature_type
    matcher_type = args.matcher_type
    if not camera_pose_dir.exists():
        raise FileNotFoundError(
            f"Scene camera reconstruction not found at {camera_pose_dir}. "
            "Run step 2 first or pass --camera_pose_dir explicitly."
        )

    run_hloc(
        data_dir,
        camera_model,
        feature_type,
        matching_method,
        matcher_type,
        50,
        camera_pose_dir,
    )
