"""Step 4: Extract monocular depth and normal maps using Metric3D."""

import sys
from dataclasses import dataclass, field
from pathlib import Path

# Ensure rhino package is importable when invoked with external Python interpreter
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from rhino.preprocessing.tool_config import ToolConfig, require_tools


def inference_one_image(model, rgb_origin, img_fname, data_dir):
    np_fname = img_fname.replace(".jpg", ".npy").replace(".png", ".npy")
    input_size = (616, 1064)  # for vit model
    h, w = rgb_origin.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

    padding = [123.675, 116.28, 103.53]
    h, w = rgb.shape[:2]
    pad_h = input_size[0] - h
    pad_w = input_size[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb = cv2.copyMakeBorder(
        rgb,
        pad_h_half,
        pad_h - pad_h_half,
        pad_w_half,
        pad_w - pad_w_half,
        cv2.BORDER_CONSTANT,
        value=padding,
    )
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb = torch.div((rgb - mean), std)
    rgb = rgb[None, :, :, :].cuda()

    with torch.no_grad():
        pred_depth, confidence, output_dict = model.inference({"input": rgb})

    pred_depth = pred_depth.squeeze()
    pred_depth = pred_depth[
        pad_info[0] : pred_depth.shape[0] - pad_info[1],
        pad_info[2] : pred_depth.shape[1] - pad_info[3],
    ]

    pred_depth = torch.nn.functional.interpolate(
        pred_depth[None, None, :, :], rgb_origin.shape[:2], mode="bilinear"
    ).squeeze()

    normalized_pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min())
    normalized_pred_depth = normalized_pred_depth.cpu().numpy()
    depth_dir = data_dir / "monocular_geometry" / "depth"
    plt.imsave(depth_dir / img_fname, normalized_pred_depth)
    np.save(depth_dir / np_fname, normalized_pred_depth)

    if "prediction_normal" in output_dict:
        pred_normal = output_dict["prediction_normal"][:, :3, :, :]
        pred_normal = pred_normal.squeeze()
        pred_normal = pred_normal[
            :,
            pad_info[0] : pred_normal.shape[1] - pad_info[1],
            pad_info[2] : pred_normal.shape[2] - pad_info[3],
        ]

        pred_normal = torch.nn.functional.interpolate(
            pred_normal[None, :, :, :], rgb_origin.shape[:2], mode="bilinear"
        ).squeeze()
        pred_normal = pred_normal / torch.norm(pred_normal, dim=0, keepdim=True)

        pred_normal_vis = pred_normal.cpu().numpy().transpose((1, 2, 0))
        pred_normal_vis = (pred_normal_vis + 1) / 2
        normal_dir = data_dir / "monocular_geometry" / "normal"
        plt.imsave(normal_dir / img_fname, pred_normal_vis)
        np.save(normal_dir / np_fname, pred_normal_vis)


@dataclass
class ExtractMonocularCuesArgs:
    data_dir: Path
    """Path to the sequence data directory."""
    metric3d_model: str = "metric3d_vit_small"
    """Metric3D model name for torch.hub.load."""
    use_submodule: bool = True
    """Load Metric3D from third_party/Metric3D instead of torch.hub."""
    tools: ToolConfig = field(default_factory=ToolConfig)


def main(args: ExtractMonocularCuesArgs) -> None:
    import sys

    data_dir = args.data_dir.resolve()
    print("Extracting monocular cues with Metric3D...")

    geometry_dir = data_dir / "monocular_geometry"
    geometry_dir.mkdir(exist_ok=True)
    (geometry_dir / "depth").mkdir(exist_ok=True)
    (geometry_dir / "normal").mkdir(exist_ok=True)

    img_dir = data_dir / "images" / "selected_frames"
    img_fnames = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))

    if args.use_submodule:
        require_tools(args.tools, "metric3d")
        sys.path.insert(0, str(args.tools.metric3d_root))
        model = torch.hub.load(str(args.tools.metric3d_root), args.metric3d_model,
                               pretrain=True, source="local")
    else:
        model = torch.hub.load("yvanyin/metric3d", args.metric3d_model, pretrain=True)
    model.cuda().eval()
    for img_path in tqdm(img_fnames):
        img = cv2.imread(str(img_path))[:, :, ::-1]
        inference_one_image(model, img, img_path.name, data_dir)


if __name__ == "__main__":
    import tyro
    main(tyro.cli(ExtractMonocularCuesArgs))
