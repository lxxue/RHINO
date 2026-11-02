"""Headless (EGL / AITViewer) rendering of predicted, GT, or composite mesh sequences."""

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

from pathlib import Path
from typing import List, Optional

import numpy as np
import trimesh
import tyro

from rhino.evaluation.mesh_sequence import (
    build_frame_pairs,
    parse_gt_meshes,
    parse_predicted_meshes,
)
from tqdm import tqdm

MESH_COLOR = [255, 255, 255, 255]


def _load_renderer_dependencies():
    try:
        import moderngl
        original_create_context = moderngl.create_standalone_context

        def egl_create_standalone_context(**kwargs):
            kwargs.setdefault("backend", "egl")
            return original_create_context(**kwargs)

        moderngl.create_standalone_context = egl_create_standalone_context

        from aitviewer.headless import HeadlessRenderer
        from aitviewer.renderables.meshes import VariableTopologyMeshes
        from aitviewer.scene.camera import OpenCVCamera
        from aitviewer.scene.material import Material
    except ImportError as error:
        raise RuntimeError(
            "Headless mesh rendering requires the optional AITViewer environment; "
            "see the evaluation installation notes in README.md"
        ) from error
    return HeadlessRenderer, VariableTopologyMeshes, OpenCVCamera, Material


def render_normal_w_aitviewer(
    camera_path: Optional[Path],
    iphone_dir: Path,
    pred_mesh_dir: Path,
    gt_files: dict,
    gt_unit_divisor: float,
    render_dir: Path,
    cam_ids: list,
    render_iphone: bool,
    frame_0: int,
    frame_end: int,
    interval: int,
    offset: int,
    mode: str,
    iphone_render_dir: Optional[Path] = None,
    scale: float = 1.0,
):
    HeadlessRenderer, VariableTopologyMeshes, OpenCVCamera, Material = (
        _load_renderer_dependencies()
    )
    cols, rows = int(3004 * scale), int(4092 * scale)
    viewer = HeadlessRenderer(size=(cols, rows))
    viewer.scene.floor.enabled = False
    viewer.scene.origin.enabled = False
    viewer.shadows_enabled = False
    viewer.scene.lights = []
    viewer.backface_culling = False

    camera_infos = dict(np.load(camera_path)) if cam_ids else {}
    cameras = {}
    for cam_id in cam_ids:
        idx = np.where(camera_infos["ids"] == cam_id)[0][0]
        intrinsic = camera_infos["intrinsics"][idx].copy()
        intrinsic[:2] *= scale
        extrinsic = camera_infos["extrinsics"][idx]
        camera = OpenCVCamera(K=intrinsic, Rt=extrinsic, cols=cols, rows=rows, viewer=viewer)
        cameras[str(cam_id)] = camera
    cameras = dict(sorted(cameras.items()))

    iphone_cameras = []
    if render_iphone:
        c2ws = np.load(iphone_dir / "c2ws.npy")
        init_trans = np.load(pred_mesh_dir.parent / "init_trans.npy")
        intrinsic = np.load(iphone_dir / "intrinsic.npy")[:3, :3]
        iphone_cols, iphone_rows = 720, 960
        for i in range(frame_0, frame_end, interval):
            camera = OpenCVCamera(
                K=intrinsic,
                Rt=np.linalg.inv(init_trans @ c2ws[i])[:3, :],
                cols=iphone_cols,
                rows=iphone_rows,
                viewer=viewer,
            )
            iphone_cameras.append(camera)

    material = Material(ambient=1, diffuse=1.0, specular=0.0)
    vertices, faces, vertex_normals, vertex_colors = [], [], [], []
    pred_color = MESH_COLOR
    gt_color = MESH_COLOR
    print("loading meshes...")
    for i in tqdm(range(frame_0, frame_end, interval)):
        if mode == "pred":
            pred_mesh = trimesh.load_mesh(str(pred_mesh_dir / f"{i:04d}.ply"))
            if len(pred_mesh.vertices) == 0:
                raise ValueError(f"Empty predicted mesh: {pred_mesh_dir / f'{i:04d}.ply'}")
            pred_mesh.visual.vertex_colors = np.full(
                (len(pred_mesh.vertices), 4), pred_color, dtype=np.uint8
            )
            m = pred_mesh
        elif mode == "gt":
            gt_mesh = trimesh.load_mesh(str(gt_files[i + offset]))
            if len(gt_mesh.vertices) == 0:
                raise ValueError(f"Empty GT mesh: {gt_files[i + offset]}")
            gt_mesh.vertices = np.array(gt_mesh.vertices) / gt_unit_divisor
            gt_mesh.visual.vertex_colors = np.full(
                (len(gt_mesh.vertices), 4), gt_color, dtype=np.uint8
            )
            m = gt_mesh
        elif mode == "composite":
            pred_mesh = trimesh.load_mesh(str(pred_mesh_dir / f"{i:04d}.ply"))
            pred_mesh_vertex_colors = np.full(
                (len(pred_mesh.vertices), 4), pred_color, dtype=np.uint8
            )
            gt_mesh = trimesh.load_mesh(str(gt_files[i + offset]))
            if len(pred_mesh.vertices) == 0 or len(gt_mesh.vertices) == 0:
                raise ValueError(f"Empty mesh at predicted frame {i}, GT frame {i + offset}")
            gt_mesh.vertices = np.array(gt_mesh.vertices) / gt_unit_divisor
            gt_mesh_vertex_colors = np.full(
                (len(gt_mesh.vertices), 4), gt_color, dtype=np.uint8
            )
            m = trimesh.util.concatenate([pred_mesh, gt_mesh])
            m_vertex_colors = np.vstack([pred_mesh_vertex_colors, gt_mesh_vertex_colors])
            m.visual = trimesh.visual.ColorVisuals(vertex_colors=m_vertex_colors)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        vertices.append(m.vertices)
        faces.append(m.faces)
        vertex_normals.append(m.vertex_normals)
        vertex_colors.append(m.visual.vertex_colors / 255.0)
    meshes = VariableTopologyMeshes(vertices, faces, vertex_normals, vertex_colors=vertex_colors)
    meshes.norm_coloring = True
    meshes.material = material
    viewer.scene.add(meshes)
    n_frames = meshes.n_frames

    render_dir.mkdir(exist_ok=True, parents=True)
    if iphone_render_dir is None:
        iphone_render_dir = render_dir
    iphone_render_dir.mkdir(exist_ok=True, parents=True)
    if render_iphone:
        for idx in range(frame_0, frame_end, interval):
            stale = iphone_render_dir / f"iphone_{idx:04d}.jpg"
            if stale.exists():
                stale.unlink()
        print("render with iphone cameras")
        for i in tqdm(range(0, n_frames)):
            camera = iphone_cameras[i]
            viewer.set_temp_camera(camera)
            meshes.current_frame_id = i
            viewer.save_frame(
                str(iphone_render_dir / f"iphone_{(i * interval + frame_0):04d}.jpg")
            )

    for cam_id in cam_ids:
        for idx in range(frame_0, frame_end, interval):
            stale = render_dir / f"{cam_id}_{idx:04d}.jpg"
            if stale.exists():
                stale.unlink()
        print("Start to process camera", cam_id)
        camera = cameras[str(cam_id)]
        viewer.set_temp_camera(camera)
        for i in tqdm(range(0, n_frames)):
            meshes.current_frame_id = i
            viewer.save_frame(
                str(render_dir / f"{cam_id}_{(i * interval + frame_0):04d}.jpg")
            )


def main(
    seq_name: str,
    version_name: str,
    take_name: str,
    offset: int,
    frame_0: int,
    frame_end: int,
    cam_ids: List[str],
    output_dir: Path,
    gt_dir: Path,
    data_dir: Path,
    subdir: str = "hoi",
    mode: str = "pred",
    scale: float = 1.0,
    expected_gt_frames: Optional[int] = None,
    stage_cameras: Optional[Path] = None,
):
    """Headless mesh renderer for pred / gt / composite modes.

    ``cam_ids`` may contain ``iphone`` (the input-video camera, from the
    processed sequence) and numeric capture-stage camera ids. Stage calibrations
    are not distributed; numeric ids need ``--stage-cameras <npz>`` with keys
    ``ids``, ``intrinsics`` and ``extrinsics``.
    """
    output_dir = Path(output_dir)
    gt_dir = Path(gt_dir)
    data_dir = Path(data_dir)

    render_iphone = "iphone" in cam_ids
    stage_cam_ids = [int(c) for c in cam_ids if c != "iphone"]
    if stage_cam_ids and stage_cameras is None:
        raise ValueError(
            "Numeric cam_ids render capture-stage views and need --stage-cameras "
            "<npz>; stage calibrations are not distributed. Use cam_ids iphone."
        )

    pred_mesh_dir = (
        output_dir / seq_name / subdir / version_name / "aligned_meshes"
    )
    iphone_dir = data_dir / seq_name / "processed"

    gt = parse_gt_meshes(gt_dir / take_name, expected_gt_frames)
    gt_files = gt.files

    iphone_render_dir: Optional[Path] = None
    if mode == "pred":
        render_dir = output_dir / seq_name / subdir / version_name / "render_pred"
    elif mode == "gt":
        render_dir = output_dir / seq_name / "render_gt"
        if render_iphone:
            iphone_render_dir = (
                output_dir / seq_name / subdir / version_name / "render_gt"
            )
    elif mode == "composite":
        render_dir = output_dir / seq_name / subdir / version_name / "render_composite"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if mode in {"pred", "composite"}:
        pred_files = parse_predicted_meshes(pred_mesh_dir, "aligned predicted mesh")
        frame_pairs = build_frame_pairs(set(pred_files), set(gt_files), offset)
    else:
        hypothetical_pred_ids = set(range(max(gt_files) - offset + 1))
        frame_pairs = build_frame_pairs(hypothetical_pred_ids, set(gt_files), offset)
    frame_end = min(frame_end, len(frame_pairs))
    if frame_0 < 0 or frame_0 >= frame_end:
        raise ValueError(
            f"Invalid render range [{frame_0}, {frame_end}) for {len(frame_pairs)} overlaps"
        )

    print(f"{seq_name}: frame {frame_0} to {frame_end}, mode: {mode}, subdir: {subdir}")
    print(f"render_dir: {render_dir}")
    if iphone_render_dir is not None:
        print(f"iphone_render_dir: {iphone_render_dir}")

    render_normal_w_aitviewer(
        camera_path=Path(stage_cameras) if stage_cameras is not None else None,
        iphone_dir=iphone_dir,
        pred_mesh_dir=pred_mesh_dir,
        gt_files=gt_files,
        gt_unit_divisor=gt.unit_divisor,
        render_dir=render_dir,
        cam_ids=stage_cam_ids,
        render_iphone=render_iphone,
        frame_0=frame_0,
        frame_end=frame_end,
        interval=1,
        offset=offset,
        mode=mode,
        iphone_render_dir=iphone_render_dir,
        scale=scale,
    )


if __name__ == "__main__":
    tyro.cli(main)
