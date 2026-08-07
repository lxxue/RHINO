import numpy as np
import torch
import trimesh
from skimage import measure

avg_pool_3d = torch.nn.AvgPool3d(2, stride=2)
upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")


@torch.no_grad()
def get_surface_sliding(sdf, resolution=256, grid_min=[-1.0, -1.0, -1.0], grid_max=[1.0, 1.0, 1.0]):
    assert resolution % 128 == 0
    resN = resolution
    cropN = 128
    level = 0
    N = resN // cropN

    # if grid_min is float:
    #     grid_min = [grid_min, grid_min, grid_min]
    # if grid_max is float:
    #     grid_max = [grid_max, grid_max, grid_max]
    xs = np.linspace(grid_min[0], grid_max[0], N + 1)
    ys = np.linspace(grid_min[1], grid_max[1], N + 1)
    zs = np.linspace(grid_min[2], grid_max[2], N + 1)

    meshes = []
    for i in range(N):
        for j in range(N):
            for k in range(N):
                x_min, x_max = xs[i], xs[i + 1]
                y_min, y_max = ys[j], ys[j + 1]
                z_min, z_max = zs[k], zs[k + 1]

                x = np.linspace(x_min, x_max, cropN)
                y = np.linspace(y_min, y_max, cropN)
                z = np.linspace(z_min, z_max, cropN)

                xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
                points = torch.tensor(
                    np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float
                ).cuda()

                def evaluate(points):
                    z = []
                    for _, pnts in enumerate(torch.split(points, 100000, dim=0)):
                        z.append(sdf(pnts))
                    z = torch.cat(z, axis=0)
                    return z

                # construct point pyramids
                points = points.reshape(cropN, cropN, cropN, 3).permute(3, 0, 1, 2)
                points_pyramid = [points]
                for _ in range(3):
                    points = avg_pool_3d(points[None])[0]
                    points_pyramid.append(points)
                points_pyramid = points_pyramid[::-1]

                # evalute pyramid with mask
                mask = None
                threshold = 2 * (x_max - x_min) / cropN * 8
                for pid, pts in enumerate(points_pyramid):
                    coarse_N = pts.shape[-1]
                    pts = pts.reshape(3, -1).permute(1, 0).contiguous()

                    if mask is None:
                        pts_sdf = evaluate(pts)
                    else:
                        mask = mask.reshape(-1)
                        pts_to_eval = pts[mask]
                        # import pdb; pdb.set_trace()
                        if pts_to_eval.shape[0] > 0:
                            pts_sdf_eval = evaluate(pts_to_eval.contiguous())
                            pts_sdf[mask] = pts_sdf_eval
                        # print("ratio", pts_to_eval.shape[0] / pts.shape[0])

                    if pid < 3:
                        # update mask
                        mask = torch.abs(pts_sdf) < threshold
                        mask = mask.reshape(coarse_N, coarse_N, coarse_N)[None, None]
                        mask = upsample(mask.float()).bool()

                        pts_sdf = pts_sdf.reshape(coarse_N, coarse_N, coarse_N)[None, None]
                        pts_sdf = upsample(pts_sdf)
                        pts_sdf = pts_sdf.reshape(-1)

                    threshold /= 2.0

                z = pts_sdf.detach().cpu().numpy()

                if not (np.min(z) > level or np.max(z) < level):
                    z = z.astype(np.float32)
                    verts, faces, normals, values = measure.marching_cubes(
                        volume=z.reshape(cropN, cropN, cropN),  # .transpose([1, 0, 2]),
                        level=level,
                        spacing=(
                            (x_max - x_min) / (cropN - 1),
                            (y_max - y_min) / (cropN - 1),
                            (z_max - z_min) / (cropN - 1),
                        ),
                    )
                    # print(np.array([x_min, y_min, z_min]))
                    # print(verts.min(), verts.max())
                    verts = verts + np.array([x_min, y_min, z_min])
                    # print(verts.min(), verts.max())

                    meshcrop = trimesh.Trimesh(verts, faces, normals)
                    # meshcrop.export(f"{i}_{j}_{k}.ply")
                    meshes.append(meshcrop)

    if len(meshes) == 0:
        print(f"WARNING: get_surface_sliding produced no valid mesh crops!")
        print(f"  grid_min={grid_min}, grid_max={grid_max}, resolution={resolution}")
        return trimesh.Trimesh()  # Return empty mesh instead of crashing

    combined = trimesh.util.concatenate(meshes)

    # Clean up degenerate faces and merge duplicate vertices from crop boundaries
    combined.merge_vertices()
    combined.remove_degenerate_faces()
    combined.remove_duplicate_faces()

    # if return_mesh:
    return combined
    # else:
    #     combined.export("{0}/surface_{1}.ply".format(path, epoch), "ply")


def simplify_mesh(mesh, target_faces=None, face_ratio=0.3):
    """Simplify mesh using quadric decimation.

    Args:
        mesh: trimesh.Trimesh object to simplify
        target_faces: target number of faces (if None, uses face_ratio)
        face_ratio: ratio of faces to keep (default 0.3 = 30%)

    Returns:
        Simplified trimesh.Trimesh object
    """
    if mesh is None or len(mesh.faces) == 0:
        return mesh
    if target_faces is None:
        target_faces = max(int(len(mesh.faces) * face_ratio), 100)
    if len(mesh.faces) <= target_faces:
        return mesh
    return mesh.simplify_quadric_decimation(target_faces)
