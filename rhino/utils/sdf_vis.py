"""Visualize learned SDF fields (object, human, scene) from a trained HOI checkpoint.

Produces 2D slice plots (XY, XZ, YZ) and an extracted mesh (.ply) for each field.
"""

from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch

from rhino.utils import mesh_utils
from rhino.utils.misc import get_class, suppress_warning


def config_torch():
    pl.seed_everything(42)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.benchmark = True
    suppress_warning("trimesh")


def config_hydra(cfg):
    cfg.dataset.train.data_dir = Path(cfg.data_root) / cfg.dataset.train.data_dir


def plot_slice(sdf_2d, extent, axis_labels, title, save_path, sdf_range=0.2, level_step=0.02):
    """Plot a 2D SDF slice with zero-level contour."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    sdf_clamped = np.clip(sdf_2d, -sdf_range, sdf_range)

    im = ax.imshow(
        sdf_clamped.T,
        origin="lower",
        extent=extent,
        cmap="RdBu",
        vmin=-sdf_range,
        vmax=sdf_range,
    )
    h, w = sdf_2d.shape
    xs = np.linspace(extent[0], extent[1], h)
    ys = np.linspace(extent[2], extent[3], w)
    # Additional level sets at regular intervals
    levels = np.arange(-sdf_range, sdf_range + level_step * 0.5, level_step)
    ax.contour(xs, ys, sdf_2d.T, levels=levels, colors="dimgray", linewidths=0.7, alpha=0.8)
    # Zero-level contour highlighted
    ax.contour(xs, ys, sdf_2d.T, levels=[0.0], colors="black", linewidths=2.0)

    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label="SDF value")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    # print(f"Saved {save_path}")


@torch.no_grad()
def query_sdf_batched(query_fn, points, cond, batch_size=100000):
    """Query an SDF function in batches."""
    sdfs = []
    for pnts in torch.split(points, batch_size, dim=0):
        sdfs.append(query_fn(pnts, cond))
    return torch.cat(sdfs, dim=0)


def make_cubic_grid(grid_min, grid_max):
    """Expand grid bounds to be cubic (equal extent on all axes)."""
    grid_min = np.asarray(grid_min, dtype=np.float32)
    grid_max = np.asarray(grid_max, dtype=np.float32)
    center = (grid_min + grid_max) / 2.0
    half = (grid_max - grid_min).max() * 0.5
    return center - half, center + half, center


def visualize_sdf_field(query_fn, cond, grid_min, grid_max, label, out_dir,
                        slice_prefix, mesh_name, slice_resolution=512, mesh_resolution=128,
                        sdf_range=0.2, level_step=0.02):
    """Visualize an SDF field: 2D slice plots + mesh extraction.

    Args:
        query_fn: callable(points, cond) -> sdf values
        cond: conditioning dict (or None)
        grid_min, grid_max: 3D grid bounds (np arrays)
        label: human-readable name for plot titles (e.g. "Object", "Human", "Scene")
        out_dir: Path to output directory
        slice_prefix: filename prefix for slices (e.g. "slice", "fg_slice", "bg_slice")
        mesh_name: filename for mesh (e.g. "obj_mesh.ply")
        slice_resolution: resolution of 2D slice grid
        mesh_resolution: resolution for marching cubes
        sdf_range: clamp range for colormap (symmetric around 0)
        level_step: spacing between contour level sets
    """
    grid_min = np.asarray(grid_min, dtype=np.float32)
    grid_max = np.asarray(grid_max, dtype=np.float32)
    center = (grid_min + grid_max) / 2.0

    print(f"\n[{label}] Grid bounds: min={grid_min}, max={grid_max}, center={center}")

    # --- 2D slice visualizations ---
    planes = [
        ("xy", 2, ("X", "Y")),  # fix Z at center
        ("xz", 1, ("X", "Z")),  # fix Y at center
        ("yz", 0, ("Y", "Z")),  # fix X at center
    ]

    for plane_name, fixed_axis, axis_labels in planes:
        varying = [i for i in range(3) if i != fixed_axis]
        ax0, ax1 = varying

        u = np.linspace(grid_min[ax0], grid_max[ax0], slice_resolution)
        v = np.linspace(grid_min[ax1], grid_max[ax1], slice_resolution)
        uu, vv = np.meshgrid(u, v, indexing="ij")

        points_np = np.zeros((slice_resolution * slice_resolution, 3), dtype=np.float32)
        points_np[:, ax0] = uu.ravel()
        points_np[:, ax1] = vv.ravel()
        points_np[:, fixed_axis] = center[fixed_axis]

        points = torch.from_numpy(points_np).cuda()
        sdf_vals = query_sdf_batched(query_fn, points, cond).cpu().numpy()
        sdf_2d = sdf_vals.reshape(slice_resolution, slice_resolution)

        extent = [grid_min[ax0], grid_max[ax0], grid_min[ax1], grid_max[ax1]]
        title = f"{label} SDF — {plane_name.upper()} slice (fixed {'XYZ'[fixed_axis]}={center[fixed_axis]:.3f})"
        save_path = str(out_dir / f"{slice_prefix}_{plane_name}.png")
        plot_slice(sdf_2d, extent, axis_labels, title, save_path, sdf_range, level_step)

    # --- Extract mesh ---
    if mesh_name is not None:
        print(f"[{label}] Extracting mesh via marching cubes (resolution={mesh_resolution})...")
        mesh = mesh_utils.get_surface_sliding(
            lambda x: query_fn(x, cond),
            resolution=mesh_resolution,
            grid_min=grid_min.tolist(),
            grid_max=grid_max.tolist(),
        )
        mesh_path = str(out_dir / mesh_name)
        if mesh is not None and len(mesh.vertices) > 0:
            mesh.export(mesh_path)
            print(f"[{label}] Saved mesh with {len(mesh.vertices)} vertices to {mesh_path}")
        else:
            print(f"[{label}] WARNING: Mesh extraction produced no geometry")


@hydra.main(version_base="1.12", config_path="../confs", config_name="base")
def main(cfg):
    config_torch()
    config_hydra(cfg)

    ckpt_path = cfg.ckpt_path
    if ckpt_path is None:
        # Auto-discover latest checkpoint in Hydra run dir
        ckpt_dir = Path("checkpoints")
        assert ckpt_dir.exists(), f"No checkpoints/ dir found in {Path.cwd()}"
        ckpts = sorted(ckpt_dir.glob("*.ckpt"))
        assert len(ckpts) > 0, f"No .ckpt files found in {ckpt_dir}"
        ckpt_path = str(ckpts[-1])
        print(f"Auto-discovered checkpoint: {ckpt_path}")

    # --- Load model ---
    print(f"Loading model from {ckpt_path}")
    model = get_class(cfg.model.cls)(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.on_load_checkpoint(ckpt)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model = model.cuda().eval()

    out_dir = Path("sdf_vis")
    out_dir.mkdir(exist_ok=True)

    # =====================================================================
    # Object SDF (canonical space)
    # =====================================================================
    if model.model.obj_verts_cano is not None:
        bbox_min = model.model.obj_verts_cano.min(dim=0).values.cpu().numpy()
        bbox_max = model.model.obj_verts_cano.max(dim=0).values.cpu().numpy()
        bbox_size = bbox_max - bbox_min
        half_extent = bbox_size.max() * 0.5 * 1.5  # max axis half-size + 50% padding
        center_bbox = (bbox_min + bbox_max) / 2.0
        obj_grid_min = center_bbox - half_extent
        obj_grid_max = center_bbox + half_extent
    else:
        limit = 6.0
        obj_grid_min = np.array([-limit, -limit, -limit], dtype=np.float32)
        obj_grid_max = np.array([limit, limit, limit], dtype=np.float32)

    obj_grid_min, obj_grid_max, _ = make_cubic_grid(obj_grid_min, obj_grid_max)
    visualize_sdf_field(
        query_fn=model.query_obj_sdf,
        cond=None,
        grid_min=obj_grid_min,
        grid_max=obj_grid_max,
        label="Object",
        out_dir=out_dir,
        slice_prefix="slice",
        mesh_name="obj_mesh.ply",
        mesh_resolution=128,
        sdf_range=0.2,
        level_step=0.02,
    )

    # =====================================================================
    # Human (foreground) SDF (canonical T-pose space)
    # =====================================================================
    # Zero body pose = T-pose in canonical space
    cond_fg = {"smplx": torch.zeros(1, 63, device="cuda") / np.pi}

    if model.model.mesh_v_cano is not None:
        fg_verts = model.model.mesh_v_cano[0].cpu().numpy()  # shape [V, 3]
        fg_min = fg_verts.min(axis=0)
        fg_max = fg_verts.max(axis=0)
        fg_bbox_size = fg_max - fg_min
        fg_grid_min = fg_min - fg_bbox_size * 0.1
        fg_grid_max = fg_max + fg_bbox_size * 0.1
    else:
        fg_limit = 2.0
        fg_grid_min = np.array([-fg_limit] * 3, dtype=np.float32)
        fg_grid_max = np.array([fg_limit] * 3, dtype=np.float32)

    fg_grid_min, fg_grid_max, _ = make_cubic_grid(fg_grid_min, fg_grid_max)
    visualize_sdf_field(
        query_fn=model.query_fg_sdf,
        cond=cond_fg,
        grid_min=fg_grid_min,
        grid_max=fg_grid_max,
        label="Human",
        out_dir=out_dir,
        slice_prefix="fg_slice",
        mesh_name="fg_mesh.ply",
        mesh_resolution=128,
        sdf_range=0.3,
        level_step=0.02,
    )

    # =====================================================================
    # Scene (background) SDF (world space)
    # =====================================================================
    # Extract mesh first using full bounding sphere, then zoom slices to mesh bounds
    bg_limit = float(model.model.bg_bounding_sphere)
    bg_full_min = np.array([-bg_limit] * 3, dtype=np.float32)
    bg_full_max = np.array([bg_limit] * 3, dtype=np.float32)

    print("\n[Scene] Extracting mesh via marching cubes (resolution=512)...")
    bg_mesh = mesh_utils.get_surface_sliding(
        lambda x: model.query_bg_sdf(x, None),
        resolution=512,
        grid_min=bg_full_min.tolist(),
        grid_max=bg_full_max.tolist(),
    )
    bg_mesh_path = str(out_dir / "bg_mesh.ply")
    if bg_mesh is not None and len(bg_mesh.vertices) > 0:
        bg_mesh.export(bg_mesh_path)
        print(f"[Scene] Saved mesh with {len(bg_mesh.vertices)} vertices to {bg_mesh_path}")
        # Use mesh bounds with 10% padding for slice visualization
        verts = np.array(bg_mesh.vertices)
        bg_grid_min = verts.min(axis=0).astype(np.float32)
        bg_grid_max = verts.max(axis=0).astype(np.float32)
        bg_pad = (bg_grid_max - bg_grid_min) * 0.1
        bg_grid_min -= bg_pad
        bg_grid_max += bg_pad
    else:
        print("[Scene] WARNING: Mesh extraction produced no geometry, using full bounding sphere for slices")
        bg_grid_min, bg_grid_max = bg_full_min, bg_full_max

    bg_grid_min, bg_grid_max, _ = make_cubic_grid(bg_grid_min, bg_grid_max)
    visualize_sdf_field(
        query_fn=model.query_bg_sdf,
        cond=None,
        grid_min=bg_grid_min,
        grid_max=bg_grid_max,
        label="Scene",
        out_dir=out_dir,
        slice_prefix="bg_slice",
        mesh_name=None,  # already extracted
        slice_resolution=512,
        sdf_range=2.0,
        level_step=0.2,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
