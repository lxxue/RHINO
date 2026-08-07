import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import PIL.Image
import pycolmap
import torch
from hloc import pairs_from_sequence
from hloc.utils.parsers import parse_retrieval
from hloc.utils.read_write_model import read_cameras_binary

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "third_party" / "mast3r"))

from typing import List, Tuple

import imageio.v3 as iio
import kapture
import mast3r.utils.path_to_dust3r  # noqa
from dust3r.datasets.utils.transforms import ImgNorm
from dust3r.inference import inference
from dust3r_visloc.datasets.utils import get_resize_function
from kapture.converter.colmap.database import COLMAPDatabase
from kapture.converter.colmap.database_extra import (
    get_colmap_camera_ids_from_db,
    get_colmap_image_ids_from_db,
    kapture_to_colmap,
)
from mast3r.colmap.database import export_matches
from mast3r.colmap.mapping import glomap_run_mapper
from mast3r.colmap.mapping import (
    kapture_import_image_folder_or_list,
    pycolmap_run_mapper,
    remove_duplicates,
    scene_prepare_images,
)
from mast3r.model import AsymmetricMASt3R
from mast3r.colmap.database import get_im_matches_hoi
from tqdm import tqdm


# copied from dust3r_visloc.datasets.utils
# change: we use it for object masks,
# so we use the flag is_mask=True to handle the mask resizing
# use NEAREST_EXACT for mask resizing
# convert img float tensor to boolean tensor
def scene_prepare_images_hoi(
    root: str,
    maxdim: int,
    patch_size: int,
    image_paths: List[str],
    device: str,
    is_mask: bool = True,
):
    images = []
    # image loading
    for idx in tqdm(range(len(image_paths))):
        rgb_image = PIL.Image.open(os.path.join(root, image_paths[idx])).convert("RGB")

        # resize images
        W, H = rgb_image.size
        resize_func, _, to_orig = get_resize_function(maxdim, patch_size, H, W, is_mask=is_mask)
        rgb_tensor = resize_func(ImgNorm(rgb_image))
        # convert to boolean tensor
        rgb_tensor = rgb_tensor < 0

        # image dictionary
        images.append(
            {
                "img": rgb_tensor[0].to(device),
                "true_shape": np.int32([rgb_tensor.shape[1:]]),
                "to_orig": to_orig,
                "idx": idx,
                "instance": image_paths[idx],
                "orig_shape": np.int32([H, W]),
            }
        )
    return images


# copied from mast3r.colmap.mapping
# TODO: our customized version with object masks
def run_mast3r_matching_hoi(
    model: AsymmetricMASt3R,
    maxdim: int,
    patch_size: int,
    device,
    kdata: kapture.Kapture,
    root_path: str,
    image_pairs_kapture: List[Tuple[str, str]],
    colmap_db,
    dense_matching: bool,
    pixel_tol: int,
    conf_thr: float,
    skip_geometric_verification: bool,
    min_len_track: int,
    masks,
    matches_viz_dir,
):
    assert kdata.records_camera is not None
    image_paths = kdata.records_camera.data_list()
    image_path_to_idx = {image_path: idx for idx, image_path in enumerate(image_paths)}
    image_path_to_ts = {
        kdata.records_camera[ts, camid]: (ts, camid)
        for ts, camid in kdata.records_camera.key_pairs()
    }

    images = scene_prepare_images(root_path, maxdim, patch_size, image_paths)
    image_pairs = [
        (
            (image_path_to_idx[image_path1], image_path1),
            (image_path_to_idx[image_path2], image_path2),
        )
        for image_path1, image_path2 in image_pairs_kapture
    ]
    matching_pairs = remove_duplicates(images, image_pairs)
    print(len(matching_pairs), "pairs to match")

    colmap_camera_ids = get_colmap_camera_ids_from_db(colmap_db, kdata.records_camera)
    colmap_image_ids = get_colmap_image_ids_from_db(colmap_db)
    im_keypoints = {idx: {} for idx in range(len(image_paths))}

    im_matches = {}
    image_to_colmap = {}
    for image_path, idx in image_path_to_idx.items():
        _, camid = image_path_to_ts[image_path]
        colmap_camid = colmap_camera_ids[camid]
        colmap_imid = colmap_image_ids[image_path]
        image_to_colmap[idx] = {"colmap_imid": colmap_imid, "colmap_camid": colmap_camid}

    # compute 2D-2D matching from dust3r inference
    batch_size = 4
    for chunk in tqdm(range(0, len(matching_pairs), batch_size)):
        pairs_chunk = matching_pairs[chunk : chunk + batch_size]
        output = inference(pairs_chunk, model, device, batch_size=1, verbose=False)
        pred1, pred2 = output["pred1"], output["pred2"]
        for i in range(len(pairs_chunk)):
            idx1 = pairs_chunk[i][0]["idx"]
            mask1 = masks[idx1]["img"]
            pred1["conf"][i][mask1] = 0.0
            pred1["desc_conf"][i][mask1] = 0.0
            idx2 = pairs_chunk[i][1]["idx"]
            mask2 = masks[idx2]["img"]
            pred2["conf"][i][mask2] = 0.0
            pred2["desc_conf"][i][mask2] = 0.0
        # TODO handle caching
        im_images_chunk = get_im_matches_hoi(
            pred1,
            pred2,
            pairs_chunk,
            image_to_colmap,
            im_keypoints,
            conf_thr,
            not dense_matching,
            pixel_tol=pixel_tol,
            viz=False,
            viz_dir=matches_viz_dir,
            num_matches_thres=0,
        )
        im_matches.update(im_images_chunk.items())

    # filter matches, convert them and export keypoints and matches to colmap db
    colmap_image_pairs = export_matches(
        colmap_db,
        images,
        image_to_colmap,
        im_keypoints,
        im_matches,
        min_len_track,
        skip_geometric_verification,
    )
    colmap_db.commit()

    return colmap_image_pairs


def main(
    data_dir: Path,
    pair_window_size: int,
    is_object: bool,
    use_glomap_mapper: bool,
    camera_pair_window_size: int = -1,
    pose_method: str = "mast3r",
):

    assert pose_method in ["mast3r", "align3r"]
    if pair_window_size == -1:
        print("pair_window_size is -1, using manual")
        pair_window_size = "manual"

    if pair_window_size == -2:
        print("pair_window_size is -2, using cotracker")
        pair_window_size = "cotracker"

    if is_object:
        image_dir = data_dir / "images" / "objects"
        mask_dir = data_dir / "masks" / "sam2_obj"
        mast3r_dir = data_dir / "object_poses" / "mast3r"
    else:
        image_dir = data_dir / "images" / "selected_frames"
        mask_dir = data_dir / "masks" / "colmap"
        mast3r_dir = data_dir / "camera_poses" / "mast3r"

    mast3r_dir.mkdir(parents=True, exist_ok=True)

    root_path = str(image_dir)
    patch_size = 16
    maxdim = 512

    image_paths = [image_path.name for image_path in image_dir.glob("*.jpg")]
    image_paths = sorted(image_paths)
    filelist_relpath = image_paths
    if not is_object:
        height, width = iio.imread(str(image_dir / image_paths[0])).shape[:2]
        focal_length = kapture.converter.colmap.cameras.DEFAULT_FOCAL_LENGTH_FACTOR * max(
            width, height
        )
        print(f"using prior focal length {focal_length}")
    else:
        # use focal length from the scene cameras
        if pose_method == "mast3r":
            camera_dir = data_dir / "camera_poses" / "mast3r" / f"glomap_sparse_{camera_pair_window_size}" / "0"
        elif pose_method == "align3r":
            camera_dir = data_dir / "camera_poses" / "align3r"
        else:
            raise ValueError(f"Invalid pose method: {pose_method}")

        cameras = read_cameras_binary(
            camera_dir / "cameras.bin"
        )
        focal_length = cameras[1].params[0]
        print(f"Using focal length from the scene camera: {focal_length:.2f}")
    kdata = kapture_import_image_folder_or_list(
        (root_path, filelist_relpath), use_single_camera=True, focal_length=focal_length
    )

    if is_object:
        mask_paths = [mask_path.name for mask_path in mask_dir.glob("*.jpg")]
    else:
        # colmap mask fnames look like {i:04d}.jpg.png
        mask_paths = [mask_path.name for mask_path in mask_dir.glob("*.png")]
    mask_paths = sorted(mask_paths)
    mask_root_path = str(mask_dir)
    device = "cuda"
    masks = scene_prepare_images_hoi(
        mask_root_path, maxdim, patch_size, mask_paths, device, is_mask=True
    )
    image_path_to_idx = {image_path: idx for idx, image_path in enumerate(image_paths)}

    references = [str(p.relative_to(image_dir).as_posix()) for p in image_dir.iterdir()]
    references = sorted(references)
    pairs_path = mast3r_dir / f"pairs_{pair_window_size}.txt"
    if type(pair_window_size) is int:
        pairs_from_sequence.main(pairs_path, image_list=references, window_size=pair_window_size)
    pairs_dict = parse_retrieval(pairs_path)
    pairs_list = [(q, r) for q, rs in pairs_dict.items() for r in rs]

    torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
    weights_path = "naver/" + "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)

    if use_glomap_mapper:
        mapper = "glomap"
    else:
        mapper = "colmap"

    colmap_db_path = mast3r_dir / f"database_{mapper}_{pair_window_size}.db"
    # Build the database under a temporary name and rename it only after
    # matching + verification complete, so an interrupted run can never leave
    # a partial database that a rerun would trust and feed to the mapper.
    building_db_path = colmap_db_path.with_name(colmap_db_path.name + ".building")
    matches_viz_dir = mast3r_dir / f"matches_viz_{pair_window_size}"
    if not os.path.isfile(colmap_db_path):
        if os.path.isfile(building_db_path):
            os.remove(building_db_path)
        colmap_db = COLMAPDatabase.connect(building_db_path)
        try:
            kapture_to_colmap(
                kdata,
                root_path,
                tar_handler=None,
                database=colmap_db,
                keypoints_type=None,
                descriptors_type=None,
                export_two_view_geometry=False,
            )
            pixel_tol = 5  # default 5
            matches_viz_dir.mkdir(exist_ok=True)
            colmap_image_pairs = run_mast3r_matching_hoi(
                model,
                maxdim,
                16,
                device,
                kdata,
                root_path,
                pairs_list,
                colmap_db,
                False,
                pixel_tol,
                1.001,
                False,
                3,
                masks=masks,
                matches_viz_dir=matches_viz_dir,
            )
            colmap_db.close()
        except Exception as e:
            print(f"Error {e}")
            colmap_db.close()
            os.remove(building_db_path)
            exit(1)

        if len(colmap_image_pairs) == 0:
            raise Exception("no matches were kept")

        # colmap db is now full, run colmap
        print("verify_matches")

        with open(pairs_path, "w", encoding="utf-8") as f:
            for image_path1, image_path2 in colmap_image_pairs:
                f.write(f"{image_path1} {image_path2}\n")
        pycolmap.verify_matches(building_db_path, pairs_path)
        os.rename(building_db_path, colmap_db_path)

    # Free MASt3R GPU memory before the mapper: GLOMAP's CUDA bundle
    # adjustment (cuDSS Cholesky) needs the VRAM and fails with
    # cudaErrorLaunchFailure on large track counts if the ViT-L model
    # is still resident in this process.
    del model, masks
    torch.cuda.empty_cache()

    if use_glomap_mapper:
        reconstruction_path = mast3r_dir / f"glomap_sparse_{pair_window_size}"
    else:
        reconstruction_path = mast3r_dir / f"colmap_sparse_{pair_window_size}"
    if os.path.isdir(reconstruction_path):
        shutil.rmtree(reconstruction_path)
    os.makedirs(reconstruction_path, exist_ok=True)
    new_colmap_db_path = reconstruction_path / f"database_{pair_window_size}.db"
    shutil.move(colmap_db_path, new_colmap_db_path)

    if use_glomap_mapper:
        glomap_bin = "glomap"
        glomap_run_mapper(
            glomap_bin, new_colmap_db_path, reconstruction_path, root_path, fix_intrinsics=is_object
        )
    else:
        pycolmap_run_mapper(
            new_colmap_db_path, reconstruction_path, root_path, fix_intrinsics=is_object
        )

    shutil.move(pairs_path, reconstruction_path)
    if matches_viz_dir.exists():
        shutil.move(matches_viz_dir, reconstruction_path)

    ouput_recon = pycolmap.Reconstruction(os.path.join(reconstruction_path, "0"))
    print(ouput_recon.summary())

    print("Final camera parameters", ouput_recon.cameras[1])



@dataclass
class Mast3rMatchingArgs:
    data_dir: Path
    pair_window_size: int = 5
    is_object: bool = False
    use_glomap_mapper: bool = False
    camera_pair_window_size: int = -1
    pose_method: str = "mast3r"


if __name__ == "__main__":
    import tyro

    args = tyro.cli(Mast3rMatchingArgs)
    main(
        data_dir=args.data_dir,
        pair_window_size=args.pair_window_size,
        is_object=args.is_object,
        use_glomap_mapper=args.use_glomap_mapper,
        camera_pair_window_size=args.camera_pair_window_size,
        pose_method=args.pose_method,
    )
