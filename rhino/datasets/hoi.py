import glob
import multiprocessing as mp
import pickle as pkl
from pathlib import Path

import cv2
import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset

from rhino.datasets.frame_utils import build_frame_indices
from rhino.utils import sample_utils


class HOIObjDataset(Dataset):
    def __init__(self, cfg):
        root = Path(cfg.data_dir)
        self.fg_ratio = mp.Value('f', 0.2)
        self.obj_ratio = mp.Value('f', 0.7)
        self.hand_ratio = mp.Value('f', 0.0)
        # self.contact_start_frame = cfg.contact_start_frame
        # self.contact_end_frame = cfg.contact_end_frame

        # images
        img_dir = root / "image"
        img_paths_all = sorted(glob.glob(f"{img_dir}/*.png"))
        # depth
        depth_dir = root / "depth"
        depth_paths_all = sorted(glob.glob(f"{depth_dir}/*.npy"))
        # normal
        normal_dir = root / "normal"
        normal_paths_all = sorted(glob.glob(f"{normal_dir}/*.npy"))
        # different types of masks
        all_mask_dir = root / "all_mask"
        human_mask_dir = root / "human_mask"
        obj_mask_dir = root / "obj_mask"
        hand_mask_dir = root / "hand_mask"
        all_mask_paths_all = sorted(glob.glob(f"{all_mask_dir}/*.png"))
        human_mask_paths_all = sorted(glob.glob(f"{human_mask_dir}/*.png"))
        obj_mask_paths_all = sorted(glob.glob(f"{obj_mask_dir}/*.png"))
        hand_mask_paths_all = sorted(glob.glob(f"{hand_mask_dir}/*.png"))

        source_paths = {
            "image": img_paths_all,
            "depth": depth_paths_all,
            "normal": normal_paths_all,
            "all_mask": all_mask_paths_all,
            "human_mask": human_mask_paths_all,
            "obj_mask": obj_mask_paths_all,
            "hand_mask": hand_mask_paths_all,
        }
        source_count = len(img_paths_all)
        mismatched = {
            name: len(paths)
            for name, paths in source_paths.items()
            if len(paths) != source_count
        }
        if mismatched:
            counts = {name: len(paths) for name, paths in source_paths.items()}
            raise ValueError(f"Processed frame count mismatch before selection: {counts}")

        training_indices = build_frame_indices(
            cfg.start_frame,
            cfg.end_frame,
            cfg.skip_step,
            cfg.exclude_frames,
            available_count=source_count,
        )
        self.training_indices = np.asarray(training_indices, dtype=np.int64)

        def selected(paths):
            return [paths[i] for i in training_indices]

        img_paths = selected(img_paths_all)
        depth_paths = selected(depth_paths_all)
        normal_paths = selected(normal_paths_all)
        all_mask_paths = selected(all_mask_paths_all)
        human_mask_paths = selected(human_mask_paths_all)
        obj_mask_paths = selected(obj_mask_paths_all)
        hand_mask_paths = selected(hand_mask_paths_all)

        self.img_paths = img_paths
        self.depth_paths = depth_paths
        self.normal_paths = normal_paths
        self.all_mask_paths = all_mask_paths
        self.human_mask_paths = human_mask_paths
        self.obj_mask_paths = obj_mask_paths
        self.hand_mask_paths = hand_mask_paths

        self.num_imgs = len(img_paths)
        # shape = np.load(root / "mean_shape.npy")
        # # scale factor for the SMPL model to be consistent with scene scale
        # self.scale = shape[0]
        # self.shape = shape[1:]
        # self.poses = np.load(root / "poses.npy")[self.training_indices]
        # self.trans = np.load(root / "normalize_trans.npy")[self.training_indices]
        # assert self.num_imgs == self.poses.shape[0]
        # assert self.num_imgs == self.trans.shape[0]

        aligned_smplx_path = root / "aligned_smplx.pkl"
        with open(aligned_smplx_path, "rb") as f:
            smplx_params = pkl.load(f)

        for key in smplx_params:
            if key != "betas":
                values = np.asarray(smplx_params[key])
                if values.ndim == 0 or values.shape[0] != source_count:
                    first_dim = "scalar" if values.ndim == 0 else values.shape[0]
                    raise ValueError(
                        f"aligned_smplx.pkl field {key!r} has {first_dim} frames, "
                        f"expected {source_count}"
                    )
                smplx_params[key] = values[self.training_indices]
            else:
                smplx_params[key] = np.array(smplx_params[key])
        self.smplx_params = smplx_params

        # do another resize for fast rendering, typically during evaluation and test
        img_resize_factor = cfg.img_resize_factor
        if img_resize_factor not in (1, 2, 4):
            raise ValueError(
                f"img_resize_factor must be one of 1, 2, or 4, got {img_resize_factor}"
            )
        first_image = self._read_image(img_paths[0])
        img_size = first_image.shape[:2]
        new_h, new_w = img_size[0] // img_resize_factor, img_size[1] // img_resize_factor
        self.img_size = (new_h, new_w)

        self.scale = smplx_params["scale"][0].astype(np.float32)
        intrinsic = np.load(root / "intrinsic.npy")
        intrinsic[0, 0] *= 1 / img_resize_factor
        intrinsic[1, 1] *= 1 / img_resize_factor
        intrinsic[0, 2] *= 1 / img_resize_factor
        intrinsic[1, 2] *= 1 / img_resize_factor
        def load_frame_array(filename):
            array = np.load(root / filename)
            if array.ndim == 0 or array.shape[0] != source_count:
                first_dim = "scalar" if array.ndim == 0 else array.shape[0]
                raise ValueError(
                    f"{filename} has {first_dim} frames, expected {source_count}"
                )
            return array[self.training_indices]

        c2ws = load_frame_array("c2ws.npy")
        obj_pose_file = getattr(cfg, "obj_pose_file", "object_poses_centered.npy")
        self.obj_poses = load_frame_array(obj_pose_file)
        self.contact_labels = load_frame_array("contact_labels.npy")
        # contact_vertices = np.load(root / "contact_vertices.npy")
        self.contact_vertices = load_frame_array("contact_vertices_refined.npy")

        self.intrinsics_all = []
        self.pose_all = []
        for c2w in c2ws:
            self.intrinsics_all.append(torch.from_numpy(intrinsic))
            self.pose_all.append(torch.from_numpy(c2w))

        self.num_sample = cfg.num_sample
        if "sampling_strategy" in cfg:
            self.sampling_strategy = cfg.sampling_strategy
        if "bbox_sampling_ratio" in cfg:
            self.bbox_sampling_ratio = cfg.bbox_sampling_ratio

        self.lazy_loading = getattr(cfg, "lazy_loading", False)
        if not self.lazy_loading:
            self.imgs = np.zeros((self.num_imgs, new_h, new_w, 3), dtype=np.float32)
            self.depths = np.zeros((self.num_imgs, new_h, new_w), dtype=np.float32)
            self.normals = np.zeros((self.num_imgs, new_h, new_w, 3), dtype=np.float32)
            self.all_masks = np.zeros((self.num_imgs, new_h, new_w), dtype=np.float32)
            self.human_masks = np.zeros((self.num_imgs, new_h, new_w), dtype=np.float32)
            self.obj_masks = np.zeros((self.num_imgs, new_h, new_w), dtype=np.float32)
            self.hand_masks = np.zeros((self.num_imgs, new_h, new_w), dtype=np.float32)
            for i in range(self.num_imgs):
                self.imgs[i], self.depths[i], self.normals[i], self.all_masks[i], \
                    self.human_masks[i], self.obj_masks[i], self.hand_masks[i] = self._load_frame(i)

    @staticmethod
    def _read_image(path):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Failed to read image: {path}")
        return image

    def _load_frame(self, i):
        new_h, new_w = self.img_size

        img = self._read_image(self.img_paths[i])
        img = cv2.resize(img, (new_w, new_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        depth = np.load(self.depth_paths[i])
        depth = cv2.resize(depth, (new_w, new_h))

        normal = np.load(self.normal_paths[i])
        normal = cv2.resize(normal, (new_w, new_h))
        normal = normal * 2.0 - 1.0
        normal_norm = np.linalg.norm(normal, axis=-1, keepdims=True)
        normal = normal / np.maximum(normal_norm, 1e-8)

        all_mask = self._read_image(self.all_mask_paths[i])
        all_mask = cv2.resize(all_mask, (new_w, new_h))
        all_mask = all_mask[:, :, 0].astype(np.float32) / 255.0

        human_mask = self._read_image(self.human_mask_paths[i])
        human_mask = cv2.resize(human_mask, (new_w, new_h))
        human_mask = human_mask[:, :, 0].astype(np.float32) / 255.0

        obj_mask = self._read_image(self.obj_mask_paths[i])
        obj_mask = cv2.resize(obj_mask, (new_w, new_h))
        obj_mask = obj_mask[:, :, 0].astype(np.float32) / 255.0

        hand_mask = self._read_image(self.hand_mask_paths[i])
        hand_mask = cv2.resize(hand_mask, (new_w, new_h))
        hand_mask = hand_mask[:, :, 0].astype(np.float32) / 255.0

        return img, depth, normal, all_mask, human_mask, obj_mask, hand_mask

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        if self.lazy_loading:
            img, depth, normal, all_mask, human_mask, obj_mask, hand_mask = self._load_frame(idx)
        else:
            img = self.imgs[idx]
            depth = self.depths[idx]
            normal = self.normals[idx]
            all_mask = self.all_masks[idx]
            human_mask = self.human_masks[idx]
            obj_mask = self.obj_masks[idx]
            hand_mask = self.hand_masks[idx]

        img_size = img.shape[:2]

        uv = np.mgrid[: img_size[0], : img_size[1]]
        uv = np.flip(uv, axis=0).copy().transpose(1, 2, 0).astype(np.float32)

        # if idx >= self.contact_start_frame and idx <= self.contact_end_frame:
        #     contact = True
        # else:
        #     contact = False
        contact_label = self.contact_labels[idx]
        contact_vertices = self.contact_vertices[idx]

        # smpl_params = torch.zeros([86], dtype=torch.float32)
        # smpl_params[0] = torch.tensor([self.scale])
        # smpl_params[1:4] = torch.from_numpy(self.trans[idx])
        # smpl_params[4:76] = torch.from_numpy(self.poses[idx])
        # smpl_params[76:] = torch.from_numpy(self.shape)

        num_sample = self.num_sample
        smplx_params = np.concatenate(
            [
                self.smplx_params["global_orient"][idx],
                self.smplx_params["body_pose"][idx],
                self.smplx_params["left_hand_pose"][idx],
                self.smplx_params["right_hand_pose"][idx],
                self.smplx_params["transl"][idx],
                self.smplx_params["expression"][idx],
                self.smplx_params["jaw_pose"][idx],
            ]
        )
        if num_sample > 0:
            data = {
                "rgb": img,
                "uv": uv,
                "human_mask": human_mask,
                "all_mask": all_mask,
                "obj_mask": obj_mask,
                "hand_mask": hand_mask,
                "depth": depth,
                "normal": normal,
            }
            if self.sampling_strategy == "uniform":
                samples = sample_utils.uniform_sampling(data, img_size, num_sample)
            elif self.sampling_strategy == "uniform_continuous":
                samples = sample_utils.uniform_sampling_continuous(data, img_size, num_sample)
            elif self.sampling_strategy == "weighted":
                samples, _ = sample_utils.weighted_sampling(
                    data, img_size, num_sample, self.bbox_sampling_ratio
                )
            elif self.sampling_strategy == "bg_only":
                samples, _ = sample_utils.bg_only_sampling(data, img_size, num_sample)
            elif self.sampling_strategy == "bg_only_patch":
                samples, _ = sample_utils.bg_only_patch_sampling(
                    data, img_size, num_sample
                )
            elif self.sampling_strategy == "fg_only":
                samples, _ = sample_utils.fg_only_sampling(data, img_size, num_sample)
            elif self.sampling_strategy == "obj_only":
                samples, _, _ = sample_utils.obj_only_sampling(
                    data, img_size, num_sample
                )
            elif self.sampling_strategy == "obj_bg":
                samples = sample_utils.obj_bg_sampling(data, img_size, num_sample)
            elif self.sampling_strategy == "fg_bg":
                samples = sample_utils.fg_bg_sampling(data, img_size, num_sample)
            elif self.sampling_strategy == "fg_obj_bg":
                samples = sample_utils.fg_obj_bg_sampling(
                    data, img_size, num_sample, fg_ratio=self.fg_ratio.value, obj_ratio=self.obj_ratio.value
                )
            elif self.sampling_strategy == "fg_obj_hand_bg":
                samples = sample_utils.fg_obj_hand_bg_sampling(
                    data, img_size, num_sample, fg_ratio=self.fg_ratio.value, obj_ratio=self.obj_ratio.value, hand_ratio=self.hand_ratio.value
                )
            else:
                raise ValueError(f"Unknown sampling strategy {self.sampling_strategy}")

            inputs = {
                "uv": samples["uv"],
                # "smpl_params": smpl_params,
                "smplx_params": smplx_params.astype(np.float32),
                "smplx_scale": self.scale.astype(np.float32),
                "smplx_betas": self.smplx_params["betas"].astype(np.float32),
                "obj_pose": self.obj_poses[idx],
                # "index_inside": index_inside,
                # "index_outside": index_outside,
                "idx": idx,
                "contact_label": contact_label,
                "contact_vertices": contact_vertices,
            }
            inputs["intrinsics"] = self.intrinsics_all[idx]
            inputs["pose"] = self.pose_all[idx]
            images = {
                "rgb": samples["rgb"],
                "depth": samples["depth"],
                "normal": samples["normal"],
                "human_mask": samples["human_mask"],
                "all_mask": samples["all_mask"],
                "obj_mask": samples["obj_mask"],
            }
        else:
            inputs = {
                "uv": uv.reshape(-1, 2),
                # "smpl_params": smpl_params,
                "smplx_params": smplx_params.astype(np.float32),
                "smplx_scale": self.scale.astype(np.float32),
                "smplx_betas": self.smplx_params["betas"].astype(np.float32),
                "idx": idx,
                "obj_pose": self.obj_poses[idx],
            }
            inputs["intrinsics"] = self.intrinsics_all[idx]
            inputs["pose"] = self.pose_all[idx]
            depth = rearrange(depth, "h w -> (h w) 1")
            normal = rearrange(normal, "h w c -> (h w) c")
            images = {
                "rgb": img.reshape(-1, 3),
                "img_size": img_size,
                "depth": depth,
                "normal": normal,
                "human_mask": human_mask,
                "all_mask": all_mask,
                "object_mask": obj_mask,
            }
        return inputs, images


class HOIObjValDataset(Dataset):
    def __init__(self, cfg):
        self.dataset = HOIObjDataset(cfg)
        self.img_size = self.dataset.img_size
        self.total_pixels = np.prod(self.img_size)
        self.pixel_per_batch = cfg.pixel_per_batch

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        image_id = np.random.randint(0, len(self.dataset))
        inputs, targets = self.dataset[image_id]
        inputs["total_pixels"] = self.total_pixels
        return inputs, targets


class HOIObjTestDataset(Dataset):
    def __init__(self, cfg):
        self.dataset = HOIObjDataset(cfg)
        self.img_size = self.dataset.img_size
        self.total_pixels = np.prod(self.img_size)
        self.pixel_per_batch = cfg.pixel_per_batch

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        inputs, targets = self.dataset[idx]
        inputs["total_pixels"] = self.total_pixels
        return inputs, targets
