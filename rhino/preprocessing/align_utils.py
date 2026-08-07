import re

import cv2
import matplotlib
import numpy as np
import open3d as o3d
import torch
from hloc.utils.read_write_model import (
    read_cameras_binary,
    read_images_binary,
    read_points3D_binary,
)
from tqdm import tqdm

from rhino.rhino_smplx.smplx_server import SMPLXServer


# from neuman
def to_homogeneous(pts):
    if isinstance(pts, torch.Tensor):
        return torch.cat([pts, torch.ones_like(pts[..., 0:1])], axis=-1)
    elif isinstance(pts, np.ndarray):
        return np.concatenate([pts, np.ones_like(pts[..., 0:1])], axis=-1)


def read_smplx(aios_dir):
    files = sorted((aios_dir / "smplx_params").glob("*.txt"))
    aios_estimates = {
        "global_orient": [],
        "body_pose": [],
        "left_hand_pose": [],
        "right_hand_pose": [],
        "transl": [],
        "expression": [],
        "jaw_pose": [],
        "smplx_verts": [],
    }
    betas = np.loadtxt(aios_dir / "mean_shape.txt")
    # SMPLXServer uses smplx.SMPLX(num_betas=300); pad short betas (e.g. AiOS outputs 10)
    if len(betas) < 300:
        betas = np.pad(betas, (0, 300 - len(betas)), mode='constant')
    betas = torch.from_numpy(betas).float()
    smplx_fn = SMPLXServer(gender="neutral", betas=betas[None])
    for file in files:
        smplx_params = np.loadtxt(file).astype(np.float32)
        smplx_params_list = np.split(smplx_params, [3, 66, 111, 156, 159, 169], axis=0)
        global_orient, body_pose, left_hand_pose, right_hand_pose, transl, expression, jaw_pose = (
            smplx_params_list
        )
        aios_estimates["global_orient"].append(global_orient)
        aios_estimates["body_pose"].append(body_pose)
        aios_estimates["left_hand_pose"].append(left_hand_pose)
        aios_estimates["right_hand_pose"].append(right_hand_pose)
        aios_estimates["transl"].append(transl)
        aios_estimates["expression"].append(expression)
        aios_estimates["jaw_pose"].append(jaw_pose)
        smplx_verts = smplx_fn(
            scale=torch.tensor(1.0),
            betas=betas[None],
            transl=torch.tensor(transl)[None],
            global_orient=torch.tensor(global_orient)[None],
            body_pose=torch.tensor(body_pose)[None],
            left_hand_pose=torch.tensor(left_hand_pose)[None],
            right_hand_pose=torch.tensor(right_hand_pose)[None],
            expression=torch.tensor(expression)[None],
            jaw_pose=torch.tensor(jaw_pose)[None],
            leye_pose=torch.zeros((1, 3), dtype=torch.float32),
            reye_pose=torch.zeros((1, 3), dtype=torch.float32),
        )
        aios_estimates["smplx_verts"].append(smplx_verts["smplx_verts"][0].detach().numpy())

    for k, v in aios_estimates.items():
        aios_estimates[k] = np.array(v).astype(np.float32)

    aios_estimates["betas"] = betas.numpy()

    return aios_estimates


def solve_translation(p3d, p2d, mvp):
    p3d = torch.from_numpy(p3d.copy()).float()
    p2d = torch.from_numpy(p2d.copy()).float()
    mvp = torch.from_numpy(mvp.copy()).float()
    translation = torch.zeros_like(p3d[0:1, 0:3], requires_grad=True)
    optim_list = [
        {"params": translation, "lr": 1e-3},
    ]
    optim = torch.optim.Adam(optim_list)

    total_iters = 1000
    for _ in tqdm(range(total_iters), total=total_iters):
        xyzw = torch.cat([p3d[:, 0:3] + translation, torch.ones_like(p3d[:, 0:1])], axis=1)
        camera_points = torch.matmul(mvp, xyzw.T).T
        image_points = camera_points / camera_points[:, 2:3]
        image_points = image_points[:, :2]
        optim.zero_grad()
        loss = torch.nn.functional.mse_loss(image_points, p2d)
        loss.backward()
        optim.step()
    print(
        "loss",
        loss.detach().cpu().numpy(),
        "translation",
        translation.detach().cpu().numpy(),
    )
    return translation.clone().detach().cpu().numpy()


def solve_scale(joints_world, plane_model, c2w):
    cam_center = c2w[:3, 3]

    a, b, c, d = plane_model
    scales = []
    for j in joints_world:
        jx, jy, jz = j
        right = -(a * cam_center[0] + b * cam_center[1] + c * cam_center[2] + d)
        coe = a * (jx - cam_center[0]) + b * (jy - cam_center[1]) + c * (jz - cam_center[2])
        s = right / coe
        if s > 0:
            scales.append(s)
    return min(scales)


def solve_transformation(verts, plane_model, c2w):
    verts_world_ori = (to_homogeneous(verts) @ c2w.T)[:, :3]
    scale = solve_scale(verts_world_ori, plane_model, c2w)
    print("scale", scale)
    transf = np.eye(4) * scale
    transf[3, 3] = 1
    transf = transf @ c2w[:3, :].T
    verts_world = to_homogeneous(verts) @ transf
    return verts_world, scale


def transform_verts_with_scale(verts, c2w, scale):
    transf = np.eye(4) * scale
    transf[3, 3] = 1
    transf = transf @ c2w[:3, :].T
    verts_world = to_homogeneous(verts) @ transf
    return verts_world


@torch.no_grad()
def convert_smplx_to_world(
    smplx_server, raw_smplx, index, scale, T_c_w, ref_smplx_verts, new_smplxs
):
    betas = torch.tensor(raw_smplx["betas"])
    smplx_global_orient = raw_smplx["global_orient"][index]
    smplx_body_pose = raw_smplx["body_pose"][index]
    smplx_left_hand_pose = raw_smplx["left_hand_pose"][index]
    smplx_right_hand_pose = raw_smplx["right_hand_pose"][index]
    smplx_transl = raw_smplx["transl"][index]
    smplx_expression = raw_smplx["expression"][index]
    smplx_jaw_pose = raw_smplx["jaw_pose"][index]
    scale = torch.tensor(scale, dtype=torch.float32)
    R_c = cv2.Rodrigues(smplx_global_orient)[0]
    R_c_w = T_c_w[:3, :3]
    R_w = R_c_w @ R_c
    R_w = cv2.Rodrigues(R_w)[0].squeeze()
    t_w = [0.0, 0.0, 0.0]
    smplx_output = smplx_server(
        scale=scale,
        betas=betas[None],
        transl=torch.tensor(t_w)[None],
        global_orient=torch.tensor(R_w)[None],
        body_pose=torch.tensor(smplx_body_pose)[None],
        left_hand_pose=torch.tensor(smplx_left_hand_pose)[None],
        right_hand_pose=torch.tensor(smplx_right_hand_pose)[None],
        expression=torch.tensor(smplx_expression)[None],
        jaw_pose=torch.tensor(smplx_jaw_pose)[None],
        leye_pose=torch.zeros((1, 3), dtype=torch.float32),
        reye_pose=torch.zeros((1, 3), dtype=torch.float32),
    )
    new_t_w = ref_smplx_verts - smplx_output["smplx_verts"][0].cpu().numpy()
    new_t_w = new_t_w.mean(axis=0) / scale
    new_smplxs["scale"].append(scale.numpy())
    new_smplxs["global_orient"].append(R_w)
    new_smplxs["body_pose"].append(smplx_body_pose)
    new_smplxs["left_hand_pose"].append(smplx_left_hand_pose)
    new_smplxs["right_hand_pose"].append(smplx_right_hand_pose)
    new_smplxs["transl"].append(new_t_w.numpy())
    new_smplxs["expression"].append(smplx_expression)
    new_smplxs["jaw_pose"].append(smplx_jaw_pose)
    return scale, new_t_w


@torch.no_grad()
def visualize_smplx_and_scene(smplx_server, background, smplxs, sphere_radius=-1.0):
    cmap = matplotlib.cm.get_cmap("Spectral")
    smplx_pcds = []
    smplx_pcds_np = []
    betas = smplxs["betas"]
    num_smplxs = len(smplxs["scale"])
    for index in range(num_smplxs):
        scale = smplxs["scale"][index]
        smplx_global_orient = smplxs["global_orient"][index]
        smplx_body_pose = smplxs["body_pose"][index]
        smplx_left_hand_pose = smplxs["left_hand_pose"][index]
        smplx_right_hand_pose = smplxs["right_hand_pose"][index]
        smplx_transl = smplxs["transl"][index]
        smplx_expression = smplxs["expression"][index]
        smplx_jaw_pose = smplxs["jaw_pose"][index]
        smplx_output = smplx_server(
            torch.tensor(scale),
            betas=torch.tensor(betas)[None],
            transl=torch.tensor(smplx_transl)[None],
            global_orient=torch.tensor(smplx_global_orient)[None],
            body_pose=torch.tensor(smplx_body_pose)[None],
            left_hand_pose=torch.tensor(smplx_left_hand_pose)[None],
            right_hand_pose=torch.tensor(smplx_right_hand_pose)[None],
            expression=torch.tensor(smplx_expression)[None],
            jaw_pose=torch.tensor(smplx_jaw_pose)[None],
            leye_pose=torch.zeros((1, 3), dtype=torch.float32),
            reye_pose=torch.zeros((1, 3), dtype=torch.float32),
        )
        smplx_pcd = o3d.geometry.PointCloud()
        smplx_pcd.points = o3d.utility.Vector3dVector(smplx_output["smplx_verts"][0].numpy())
        smplx_pcd.paint_uniform_color(cmap(index / num_smplxs)[:3])
        smplx_pcds.append(smplx_pcd)
        smplx_pcds_np.append(smplx_output["smplx_verts"][0].numpy())
    if sphere_radius > 0.0:
        bounding_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        bounding_sphere.paint_uniform_color([0.0, 1.0, 0.0])
        background.append(bounding_sphere)

    o3d.visualization.draw_geometries(background + smplx_pcds)
    return smplx_pcds_np, smplx_pcds


def filter_scene_points(scene_mesh, quantile, center=None):
    if isinstance(scene_mesh, o3d.geometry.PointCloud):
        sample_points = scene_mesh
    else:
        sample_points = scene_mesh.sample_points_uniformly(number_of_points=100000)
    if center is None:
        bbox_max = sample_points.get_max_bound().squeeze()
        bbox_min = sample_points.get_min_bound().squeeze()
        center = (bbox_max + bbox_min) / 2.0
    dist_to_center = np.linalg.norm(sample_points.points - center, axis=1)
    dist_quantile = np.quantile(dist_to_center, quantile)
    filtered_points = np.asarray(sample_points.points)[dist_to_center < dist_quantile]
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_pcd.paint_uniform_color([0.0, 0.0, 1.0])
    return filtered_pcd


def read_intrinsics_binary(sparse_dir):
    cameras = read_cameras_binary(sparse_dir / "cameras.bin")
    assert len(cameras) == 1
    model = cameras[1].model
    params = cameras[1].params
    if model == "SIMPLE_PINHOLE":
        fl_x = params[0]
        fl_y = params[0]
        cx = params[1]
        cy = params[2]
    elif model == "PINHOLE":
        fl_x = params[0]
        fl_y = params[1]
        cx = params[2]
        cy = params[3]
    elif model == "SIMPLE_RADIAL":
        fl_x = params[0]
        fl_y = params[0]
        cx = params[1]
        cy = params[2]
        print("We skip radial distortion for SIMPLE_RADIAL")
    else:
        raise NotImplementedError
    intrinsics = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]], dtype=np.float32)
    return intrinsics


def load_intrinsic(sparse_dir):
    with open(sparse_dir / "cameras.txt", "rt") as f:
        for line in f:
            if line[0] == "#":
                continue
            els = line.split(" ")
            assert els[0] == "1"
            assert els[1] == "SIMPLE_PINHOLE" or els[1] == "PINHOLE"
            w = float(els[2])
            h = float(els[3])
            fl_x = float(els[4])
            fl_y = float(els[4])
            if els[1] == "SIMPLE_PINHOLE":
                cx = float(els[5])
                cy = float(els[6])
            elif els[1] == "PINHOLE":
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
            else:
                print("unknown camera model ", els[1])

    intrinsic = np.zeros((3, 3), dtype=np.float32)
    intrinsic[0, 0] = fl_x
    intrinsic[1, 1] = fl_y
    intrinsic[0, 2] = cx
    intrinsic[1, 2] = cy
    intrinsic[2, 2] = 1
    return intrinsic


def read_extrinsics_binary(sparse_dir):
    images = read_images_binary(sparse_dir / "images.bin")
    c2ws = {}
    bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
    for id in images:
        i = id - 1
        image = images[id]
        assert image.id == id
        assert image.name == f"{i:04d}.png" or image.name == f"{i:04d}.jpg"
        qvec = image.qvec
        tvec = image.tvec
        R = qvec2rotmat(qvec)
        t = tvec.reshape([3, 1])
        w2c = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        c2w = np.linalg.inv(w2c)
        c2ws[image.name] = c2w.astype(np.float32)
    return c2ws


def load_extrinsics(sparse_dir):
    c2ws = {}
    w2cs = {}
    bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
    with open(sparse_dir / "images.txt", "rt") as f:
        i = 0
        for line in f:
            line = line.strip()
            if line[0] == "#":
                continue
            i += 1
            if i % 2 == 1:
                elems = line.split(" ")
                assert len(elems) == 10
                fname = elems[9]
                if len(fname) == 9:
                    fname = fname[1:]
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                R = qvec2rotmat(qvec)
                t = tvec.reshape([3, 1])
                w2c = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
                c2w = np.linalg.inv(w2c)
                w2cs[fname] = w2c
                c2ws[fname] = c2w

    return c2ws, w2cs


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def read_point_cloud(points_txt_path):
    with open(points_txt_path, "r") as fid:
        line = fid.readline()
        assert line == "# 3D point list with one line of data per point:\n"
        line = fid.readline()
        assert (
            line == "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        )
        line = fid.readline()
        assert re.search("^# Number of points: \d+, mean track length: [-+]?\d*\.\d+|\d+\n$", line)
        num_points, mean_track_length = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        num_points = int(num_points)
        mean_track_length = float(mean_track_length)

        xyz = np.zeros((num_points, 3), dtype=np.float32)
        rgb = np.zeros((num_points, 3), dtype=np.float32)

        for i in tqdm(range(num_points), desc="reading point cloud"):
            elems = fid.readline().split()
            xyz[i] = list(map(float, elems[1:4]))
            rgb[i] = list(map(float, elems[4:7]))
        pcd = np.concatenate([xyz, rgb], axis=1)

    sparse_pcd = o3d.geometry.PointCloud()
    sparse_pcd.points = o3d.utility.Vector3dVector(pcd[:, :3])
    sparse_pcd.colors = o3d.utility.Vector3dVector(pcd[:, 3:6] / 255.0)
    return sparse_pcd


def read_point_cloud_binary(sparse_dir, img_id=None):
    points = read_points3D_binary(sparse_dir / "points3D.bin")
    xyz = []
    rgb = []
    for key in points:
        if img_id is None:
            xyz.append(points[key].xyz)
            rgb.append(points[key].rgb)
        else:
            if img_id in points[key].image_ids:
                xyz.append(points[key].xyz)
                rgb.append(points[key].rgb)
    xyz = np.array(xyz).astype(np.float32)
    rgb = np.array(rgb).astype(np.float32)
    sparse_pcd = o3d.t.geometry.PointCloud()
    sparse_pcd.point.positions = xyz
    sparse_pcd.point.colors = rgb / 255.0
    return sparse_pcd


def crop_point_cloud(pcd, x_min, x_max, y_min, y_max, z_min, z_max):
    roi = pcd.crop(
        o3d.geometry.AxisAlignedBoundingBox([x_min, y_min, z_min], [x_max, y_max, z_max])
    )
    o3d.visualization.draw_geometries([pcd], window_name="Original Point Cloud")
    o3d.visualization.draw_geometries([roi], window_name="Cropped Point Cloud")
    return roi
