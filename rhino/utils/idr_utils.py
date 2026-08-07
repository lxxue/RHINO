import torch
from torch.nn import functional as F


def back_project(uv, P, C):
    """compute the direction of ray through each pixel"""
    uv_extended = torch.cat([uv, torch.ones_like(uv)], dim=-1)
    d = torch.bmm(P.inverse(), uv_extended.transpose(1, 2)).transpose(1, 2)[:, :, :3] - C
    d = F.normalize(d, dim=2)
    return d, C


def get_sphere_intersection(cam_loc, ray_directions, r=1.0):
    # Input: n_images x 4 x 4 ; n_images x n_rays x 3
    # Output: n_images * n_rays x 2 (close and far) ; n_images * n_rays
    n_imgs, n_pix, _ = ray_directions.shape

    cam_loc = cam_loc.unsqueeze(-1)
    ray_cam_dot = torch.bmm(ray_directions, cam_loc).squeeze()
    under_sqrt = ray_cam_dot**2 - (cam_loc.norm(2, 1) ** 2 - r**2)

    under_sqrt = under_sqrt.reshape(-1)
    mask_intersect = under_sqrt > 0

    sphere_intersections = torch.zeros((n_imgs * n_pix, 2), device=mask_intersect.device)
    sphere_intersections[mask_intersect] = torch.sqrt(under_sqrt[mask_intersect] + 1e-8).unsqueeze(
        -1
    ) * torch.Tensor([-1.0, 1.0], device=mask_intersect.device)
    sphere_intersections[mask_intersect] -= ray_cam_dot.reshape(-1)[mask_intersect].unsqueeze(-1)

    sphere_intersections = sphere_intersections.reshape(n_imgs, n_pix, 2)
    sphere_intersections = sphere_intersections.clamp_min(0.0)
    mask_intersect = mask_intersect.reshape(n_imgs, n_pix)

    return sphere_intersections, mask_intersect


def split_input(model_input, total_pixels, n_pixels=10000):
    """
    Split the input to fit Cuda memory for large resolution.
    Can decrease the value of n_pixels in case of cuda out of memory error.
    """

    split = []

    for i, indx in enumerate(
        torch.split(torch.arange(total_pixels, device=model_input["uv"].device), n_pixels, dim=0)
    ):
        data = model_input.copy()
        data["uv"] = torch.index_select(model_input["uv"], 1, indx)
        # data['object_mask'] = torch.index_select(model_input['object_mask'], 1, indx)
        if "bg_image" in data:
            data["bg_image"] = torch.index_select(model_input["bg_image"], 1, indx)
        split.append(data)
    return split


def merge_output(res, total_pixels, batch_size):
    """Merge the split output."""

    model_outputs = {}
    for entry in res[0]:
        if res[0][entry] is None:
            continue
        if len(res[0][entry].shape) == 1:
            model_outputs[entry] = torch.cat(
                [r[entry].reshape(batch_size, -1, 1) for r in res], 1
            ).reshape(batch_size * total_pixels)
        elif len(res[0][entry].shape) == 2 and res[0][entry].shape[0] == batch_size:
            model_outputs[entry] = torch.cat([r[entry] for r in res], 1).reshape(
                batch_size * total_pixels
            )
        else:
            model_outputs[entry] = torch.cat(
                [r[entry].reshape(batch_size, -1, r[entry].shape[-1]) for r in res], 1
            ).reshape(batch_size * total_pixels, -1)
    return model_outputs
