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

from rhino.preprocessing.colmap_utils import run_command, undistort_images
from rhino.preprocessing.record3d_utils import (
    create_database_from_record3d,
    load_record3d_cameras,
    triangulate,
)


def run_colmap(data_dir, camera_model, matching_method):
    colmap_dir = data_dir / "localization" / f"colmap-sift-{matching_method}-NN-ratio"
    image_dir = data_dir / "images" / "selected_frames"
    mask_dir = data_dir / "masks" / "colmap"
    colmap_dir.mkdir(parents=True, exist_ok=True)
    # feature extraction
    feature_extractor_cmd = [
        "colmap",
        f"feature_extractor",
        "--database_path",
        f"{colmap_dir / 'database.db'}",
        "--image_path",
        str(image_dir),
        "--ImageReader.mask_path",
        str(mask_dir),
        f"--ImageReader.camera_model",
        f"{camera_model}",
        "--ImageReader.single_camera",
        "1",
        "--SiftExtraction.estimate_affine_shape",
        "true",
        "--SiftExtraction.domain_size_pooling",
        "true",
        "--SiftExtraction.use_gpu",
        "1",
    ]
    run_command(feature_extractor_cmd)

    # feature matching
    feature_matcher_cmd = [
        f"colmap",
        f"{matching_method}_matcher",
        f"--database_path",
        f"{colmap_dir / 'database.db'}",
        "--SiftMatching.guided_matching",
        "true",
        "--SiftMatching.use_gpu",
        "1",
    ]
    if matching_method == "sequential":
        # https://github.com/colmap/colmap/issues/636
        feature_matcher_cmd += ["--SequentialMatching.overlap", "20"]
        feature_matcher_cmd += ["--SequentialMatching.quadratic_overlap", "0"]
    run_command(feature_matcher_cmd)

    db_path = colmap_dir / "database.db"
    sparse_dir = colmap_dir / "sparse"

    # sparse reconstruction
    sparse_dir.mkdir(parents=False, exist_ok=True)
    mapper_cmd = [
        "colmap",
        "mapper",
        "--database_path",
        str(db_path),
        "--image_path",
        str(image_dir),
        "--output_path",
        str(sparse_dir),
        "--Mapper.ba_global_function_tolerance",
        "1e-6",  # from nerfstuido
    ]
    print("Sparse reconstruction started")
    run_command(mapper_cmd)
    print("Sparse reconstruction finished")

    # refine intrinsics
    bundle_adjuster_cmd = [
        "colmap",
        "bundle_adjuster",
        "--input_path",
        str(sparse_dir / "0"),
        "--output_path",
        str(sparse_dir / "0"),
        "--BundleAdjustment.refine_principal_point",
        "1",
    ]
    print("Refining intrinsics started")
    run_command(bundle_adjuster_cmd)
    print("Refining intrinsics finished")

    undistort_images(image_dir, sparse_dir / "0", sparse_dir / "dense")

    return


# mostly from nerfstudio
def run_hloc(data_dir, camera_model, feature_type, matching_method, matcher_type, num_matched):
    hloc_dir = data_dir / "localization" / f"hloc-{feature_type}-{matching_method}-{matcher_type}"
    image_dir = data_dir / "images" / "selected_frames"
    mask_dir = data_dir / "masks" / "colmap"
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
        pairs_from_sequence.main(pairs, image_list=references, window_size=20)
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

    image_options = pycolmap.ImageReaderOptions(camera_model=camera_model)

    sfm_dir = hloc_dir / "sparse" / "0"
    sfm_dir.mkdir(parents=True, exist_ok=True)
    dense_dir = hloc_dir / "sparse" / "dense"
    reconstruction.main(
        sfm_dir,
        image_dir,
        pairs,
        features,
        matches,
        camera_mode=pycolmap.CameraMode.SINGLE,
        image_options=image_options,
    )

    undistort_images(image_dir, sfm_dir, dense_dir)
    return


@dataclass
class EstimateCameraPosesArgs:
    data_dir: Path
    camera_model: str = "SIMPLE_PINHOLE"
    sfm_tool: str = "hloc"
    feature_type: str = "superpoint_aachen"
    matching_method: Optional[str] = None
    matcher_type: str = "superpoint+lightglue"
    video_resize_factor: int = 1
    record3d_num_ba_iterations: int = 3
    """Number of Record3D triangulation/bundle-adjustment rounds."""


if __name__ == "__main__":
    import tyro

    print("Estimating camera poses...")

    # combos:
    #   colmap + sift + {exhaustive, vocab_tree, sequential}
    #   hloc + sift + NN-ratio
    #   hloc + superpoint_aachen + superglue
    #   hloc + superpoint_aachen + lightglue
    #   hloc + r2d2 + NN-ratio  (for synthetic data)

    args = tyro.cli(EstimateCameraPosesArgs)
    data_dir = args.data_dir.resolve()
    sfm_tool = args.sfm_tool
    camera_model = args.camera_model
    feature_type = args.feature_type
    matching_method = args.matching_method
    if matching_method is None:
        matching_method = "poses" if sfm_tool == "record3d" else "sequential"
    matcher_type = args.matcher_type
    video_resize_factor = args.video_resize_factor
    if sfm_tool == "record3d" and matching_method != "poses":
        raise ValueError("Record3D localization requires --matching_method poses.")
    if sfm_tool == "colmap":
        if feature_type != "sift" or matcher_type != "NN-ratio":
            raise ValueError("COLMAP SfM requires feature_type='sift' and matcher_type='NN-ratio'")
        run_colmap(
            data_dir,
            camera_model,
            matching_method,
        )
    elif sfm_tool == "hloc":
        run_hloc(
            data_dir,
            camera_model,
            feature_type,
            matching_method,
            matcher_type,
            50,
        )
    elif sfm_tool == "record3d":
        c2ws, intrinsics = load_record3d_cameras(data_dir, video_resize_factor)
        create_database_from_record3d(
            data_dir, c2ws, intrinsics, feature_type, matching_method, matcher_type
        )
        sfm_dir = triangulate(
            data_dir,
            sfm_tool,
            feature_type,
            matching_method,
            matcher_type,
            num_matched=40,
            num_ba_iterations=args.record3d_num_ba_iterations,
        )
        image_dir = data_dir / "images" / "selected_frames"
        dense_dir = sfm_dir.parent / "dense"
        undistort_images(image_dir, sfm_dir, dense_dir)
    else:
        raise NotImplementedError
