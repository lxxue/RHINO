import pickle as pkl
from pathlib import Path
from typing import Dict, Optional

import kaolin
import numpy as np
import torch
import torch.nn as nn
import trimesh
from einops import rearrange, repeat
from kaolin.ops.mesh import index_vertices_by_faces
from omegaconf.dictconfig import DictConfig
from torch.autograd import grad

from rhino.components.body_sampler import PointInSpace
from rhino.components.deformer import RigidDeformer
from rhino.components.density import LaplaceDensity
from rhino.components.network import ImplicitNet, RenderingNet
from rhino.components.ray_sampler import (
    ErrorBoundSamplerWithoutDeformer,
    ErrorBoundSamplerWithRigidDeformerAndBBox,
    ErrorBoundSamplerWithSMPLXDeformerAndBBox,
)
from rhino.rhino_smplx.smplx_deformer import SMPLXDeformer
from rhino.rhino_smplx.smplx_server import SMPLXServer
from rhino.utils import render_utils


class HOINetwork(nn.Module):

    def __init__(
        self,
        cfg: DictConfig,
        num_training_frames: int,
        gender: str,
        scale: torch.Tensor,
        betas: torch.Tensor,
    ):
        super().__init__()
        self.cfg = cfg
        self.num_training_frames = num_training_frames

        smplx_dir = Path(__file__).parents[1] / "checkpoints" / "smplx"
        required_smplx_assets = [
            smplx_dir / "SMPLX_NEUTRAL.npz",
            smplx_dir / "MANO_SMPLX_vertex_ids.pkl",
            smplx_dir / "SMPL-X__FLAME_vertex_ids.npy",
        ]
        missing_assets = [str(path) for path in required_smplx_assets if not path.is_file()]
        if missing_assets:
            raise FileNotFoundError(
                "Missing required SMPL-X training assets:\n  - "
                + "\n  - ".join(missing_assets)
                + "\nDownload the licensed SMPL-X model and auxiliary vertex-index files "
                "described in README.md."
            )

        self.fg_implicit_network = ImplicitNet(cfg.fg_implicit_network)
        self.fg_rendering_network = RenderingNet(cfg.fg_rendering_network)

        self.obj_implicit_network = ImplicitNet(cfg.obj_implicit_network)
        self.obj_rendering_network = RenderingNet(cfg.obj_rendering_network)

        self.bg_implicit_network = ImplicitNet(cfg.bg_implicit_network)
        self.bg_rendering_network = RenderingNet(cfg.bg_rendering_network)

        self.bg_rendering_mode = cfg.bg_rendering_network.mode
        self.obj_rendering_mode = cfg.obj_rendering_network.mode
        if "frame_encoding" in self.obj_rendering_mode:
            self.obj_dim_frame_encoding = cfg.obj_rendering_network.dim_frame_encoding
            self.obj_frame_encoder = nn.Embedding(num_training_frames, self.obj_dim_frame_encoding)
        if "frame_encoding" in self.bg_rendering_mode:
            self.bg_dim_frame_encoding = cfg.bg_rendering_network.dim_frame_encoding
            self.bg_frame_encoder = nn.Embedding(num_training_frames, self.bg_dim_frame_encoding)

        self.fg_bounding_sphere = cfg.fg_implicit_network.scene_bounding_sphere
        self.obj_bounding_sphere = cfg.obj_implicit_network.scene_bounding_sphere
        self.bg_bounding_sphere = cfg.bg_implicit_network.scene_bounding_sphere
        self.fg_density = LaplaceDensity(**cfg.fg_density)
        self.obj_density = LaplaceDensity(**cfg.obj_density)
        self.bg_density = LaplaceDensity(**cfg.bg_density)

        self.fg_ray_sampler = ErrorBoundSamplerWithSMPLXDeformerAndBBox(
            self.fg_bounding_sphere, **cfg.fg_ray_sampler
        )
        self.obj_ray_sampler = ErrorBoundSamplerWithRigidDeformerAndBBox(
            self.obj_bounding_sphere, **cfg.obj_ray_sampler
        )
        self.bg_ray_sampler = ErrorBoundSamplerWithoutDeformer(
            self.bg_bounding_sphere, **cfg.bg_ray_sampler
        )

        self.eik_sampler = PointInSpace()
        self.hand_point_sampler = PointInSpace()
        self.body_point_sampler = PointInSpace()

        self.smplx_server = SMPLXServer(gender=gender, betas=betas)
        self.fg_deformer = SMPLXDeformer(
            gender=gender, betas=betas, scale=scale, smplx_server=self.smplx_server
        )
        self.obj_deformer = RigidDeformer()

        if cfg.smplx_init:
            smplx_init_ckpt_path = cfg.smplx_init_ckpt_path
            if smplx_init_ckpt_path is None:
                # Bundled canonical-human SDF initialization
                smplx_init_ckpt_path = str(
                    Path(__file__).resolve().parents[1] / "checkpoints" / "smplx_init_last.ckpt"
                )
            smplx_init_ckpt = torch.load(smplx_init_ckpt_path, weights_only=True)
            smplx_init_state_dict = smplx_init_ckpt["state_dict"]
            new_state_dict = {}
            for key, value in smplx_init_state_dict.items():
                new_state_dict[key.removeprefix("fg_implicit_network.")] = value
            self.fg_implicit_network.load_state_dict(new_state_dict)

        self.register_buffer("mesh_v_cano", self.smplx_server.verts_c)
        self.register_buffer(
            "mesh_f_cano", torch.tensor(self.smplx_server.smplx.faces.astype(np.int64))
        )
        self.register_buffer(
            "mesh_face_vertices",
            index_vertices_by_faces(self.mesh_v_cano, self.mesh_f_cano),
        )

        # place holder for the object mesh vertices
        self.register_buffer("obj_verts_cano", None)

        # obj_pcd = trimesh.load(data_root / "object_pcd.ply")
        # obj_points = torch.from_numpy(obj_pcd.vertices).float()
        # # homogeneous coordinates
        # obj_points = torch.cat([obj_points, torch.ones_like(obj_points[:, :1])], dim=1)
        # obj_points = obj_points.transpose(0, 1)
        # self.register_buffer("obj_points", obj_points)

        self.reg_hands = True
        self.check_self_penetration = True
        hand_vertex_ids_path = smplx_dir / "MANO_SMPLX_vertex_ids.pkl"
        with open(hand_vertex_ids_path, "rb") as f:
            smplx_hand_vertex_ids_dict = pkl.load(f)
        smplx_hand_vertex_ids = torch.from_numpy(
            np.concatenate(
                [smplx_hand_vertex_ids_dict["left_hand"], smplx_hand_vertex_ids_dict["right_hand"]]
            )
        )
        smplx_face_vertex_ids = np.load(smplx_dir / "SMPL-X__FLAME_vertex_ids.npy")
        # the body vertex ids are the rest of the vertices
        smplx_body_vertex_ids = np.setdiff1d(
            np.arange(self.mesh_v_cano.shape[1]), smplx_face_vertex_ids
        )
        smplx_body_vertex_ids = np.setdiff1d(smplx_body_vertex_ids, smplx_hand_vertex_ids)
        self.reg_body = True
        self.register_buffer("smplx_body_vertex_ids", torch.from_numpy(smplx_body_vertex_ids))
        self.register_buffer("smplx_hand_vertex_ids", smplx_hand_vertex_ids)
        self.register_buffer("smplx_v_cano", self.mesh_v_cano.clone().detach())
        self.register_buffer("smplx_f_cano", self.mesh_f_cano.clone().detach())
        self.register_buffer(
            "smplx_face_vertices",
            index_vertices_by_faces(self.smplx_v_cano, self.smplx_f_cano),
        )

        smplx_trimesh = trimesh.Trimesh(
            vertices=self.smplx_v_cano[0].cpu().numpy(),
            faces=self.smplx_f_cano.cpu().numpy(),
        )
        smplx_vertex_normals = smplx_trimesh.vertex_normals
        smplx_vertex_normals = smplx_vertex_normals.astype(np.float32)
        smplx_vertices = self.smplx_v_cano[0][smplx_body_vertex_ids]
        smplx_vertex_normals = smplx_vertex_normals[smplx_body_vertex_ids]
        # remove foot vertices
        smplx_non_foot_mask = smplx_vertices[:, 1] > -1.05
        smplx_foot_mask = smplx_vertices[:, 1] <= -1.05
        smplx_body_vertices = smplx_vertices[smplx_non_foot_mask]
        smplx_body_vertex_normals = smplx_vertex_normals[smplx_non_foot_mask]
        smplx_body_interior_points = smplx_body_vertices - 0.03 * smplx_body_vertex_normals
        smplx_feet_vertices = smplx_vertices[smplx_foot_mask]
        smplx_feet_vertex_normals = smplx_vertex_normals[smplx_foot_mask]
        smplx_feet_interior_points = smplx_feet_vertices - 0.01 * smplx_feet_vertex_normals

        self.register_buffer("smplx_body_interior_points", smplx_body_interior_points[None])
        self.register_buffer("smplx_feet_interior_points", smplx_feet_interior_points[None])

        self.penetration_loss_steps = cfg.max_steps // 5

    def sdf_func_fg(self, x, cond, smplx_tfs, smplx_verts):
        x_c, outlier_mask = self.fg_deformer.forward(
            x, smplx_tfs, return_weights=False, inverse=True, smplx_verts=smplx_verts
        )
        output = self.fg_implicit_network(x_c, cond)[0]
        sdf = output[:, 0:1]
        feature = output[:, 1:]
        if not self.training:
            sdf[outlier_mask] = self.fg_bounding_sphere + 1.0

        return sdf, x_c, feature

    def sdf_func_fg_with_gradient(self, x, cond, smplx_tfs, smplx_verts):
        if not self.training:
            x.requires_grad_(True)
        x_c, outlier_mask = self.fg_deformer.forward(
            x, smplx_tfs, return_weights=False, inverse=True, smplx_verts=smplx_verts
        )

        # pnts_c = x_c.detach().clone()
        # pnts_c.requires_grad_(True)
        # pnts_d = self.fg_deformer.forward_skinning(pnts_c[None], None, smplx_tfs)[0]
        # num_dim = pnts_d.shape[-1]
        # grads = []
        # for i in range(num_dim):
        #     d_out = torch.zeros_like(pnts_d, requires_grad=False, device=pnts_d.device)
        #     d_out[:, i] = 1
        #     grad = torch.autograd.grad(
        #         outputs=pnts_d,
        #         inputs=pnts_c,
        #         grad_outputs=d_out,
        #         create_graph=create_graph,
        #         retain_graph=True if i < num_dim - 1 else retain_graph,
        #         only_inputs=True,
        #     )[0]
        #     grads.append(grad)
        # grads = torch.stack(grads, dim=-2)
        # grads_inv = grads.inverse()

        output = self.fg_implicit_network(x_c, cond)[0]
        sdf = output[:, 0:1]
        feature = output[:, 1:]

        if not self.training:
            sdf[outlier_mask] = self.fg_bounding_sphere + 1.0

        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        grad_sdf_x = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=self.training,
            retain_graph=self.training,
            only_inputs=True,
        )[0]

        # grad_sdf_x = torch.einsum("bi,bij->bj", grad_sdf_xc, grads_inv)

        return x_c, sdf, feature, grad_sdf_x

    def sdf_func_obj(self, x: torch.Tensor, obj_pose: torch.Tensor, cond: Optional[Dict] = None):
        x_c = self.obj_deformer.forward(x, obj_pose)
        output = self.obj_implicit_network(x_c, cond)[0]
        sdf = output[:, 0:1]
        feature = output[:, 1:]
        return sdf, feature, x_c

    def sdf_func_obj_with_gradient(self, x, obj_pose, cond):
        if not self.training:
            x.requires_grad_(True)
        x_c = self.obj_deformer.forward(x, obj_pose)
        output = self.obj_implicit_network(x_c, cond)[0]
        sdf = output[:, 0:1]
        feature = output[:, 1:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=self.training,
            retain_graph=self.training,
            only_inputs=True,
        )[0]
        return x_c, sdf, feature, gradients

    def sdf_func_bg(self, x, cond):
        output = self.bg_implicit_network(x, cond)[0]
        sdf = output[:, 0:1]
        feature = output[:, 1:]
        return sdf, feature

    def sdf_func_bg_with_gradient(self, x, cond):
        x.requires_grad_(True)
        output = self.bg_implicit_network(x, cond)[0]
        sdf = output[:, 0:1]
        feature = output[:, 1:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=self.training,
            retain_graph=self.training,
            only_inputs=True,
        )[0]
        return sdf, feature, gradients

    # determine if ray is in or off the surface, not points
    def check_off_in_surface_points_cano(self, x_cano, N_samples, threshold=0.05):
        distance, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(
            x_cano.unsqueeze(0).contiguous(), self.mesh_face_vertices
        )

        distance = torch.sqrt(distance + 1e-8)  # kaolin outputs squared distance
        sign = kaolin.ops.mesh.check_sign(
            self.mesh_v_cano, self.mesh_f_cano, x_cano.unsqueeze(0)
        ).float()
        sign = 1 - 2 * sign
        signed_distance = sign * distance
        batch_size = x_cano.shape[0] // N_samples
        signed_distance = signed_distance.reshape(batch_size, N_samples, 1)

        minimum = torch.min(signed_distance, 1)[0]
        index_off_surface = (minimum > threshold).squeeze(1)
        index_in_surface = (minimum <= 0.0).squeeze(1)
        return index_off_surface, index_in_surface

    @torch.enable_grad()
    def forward(self, inputs):
        intrinsics = inputs["intrinsics"]
        pose = inputs["pose"]
        obj_pose = inputs["obj_pose"]
        uv = inputs["uv"]
        device = pose.device

        """
        smplx_pose: [0:3-global_orient,
                     3:66-body_pose,
                     66:111-left_hand_pose,
                     111:156-right_hand_pose,
                     156:159-transl,
                     159:169-expression,
                     169:172-jaw_pose]

        """
        smplx_scale = inputs["smplx_scale"]
        smplx_betas = inputs["smplx_betas"]
        smplx_global_orient = inputs["smplx_global_orient"]
        smplx_body_pose = inputs["smplx_body_pose"]
        smplx_transl = inputs["smplx_transl"]
        smplx_left_hand_pose = inputs["smplx_left_hand_pose"]
        smplx_right_hand_pose = inputs["smplx_right_hand_pose"]
        smplx_expression = inputs["smplx_expression"]
        smplx_jaw_pose = inputs["smplx_jaw_pose"]
        smplx_leye_pose = inputs["smplx_leye_pose"]
        smplx_reye_pose = inputs["smplx_reye_pose"]

        smplx_outputs = self.smplx_server(
            scale=smplx_scale,
            betas=smplx_betas,
            transl=smplx_transl,
            global_orient=smplx_global_orient,
            body_pose=smplx_body_pose,
            left_hand_pose=smplx_left_hand_pose,
            right_hand_pose=smplx_right_hand_pose,
            expression=smplx_expression,
            jaw_pose=smplx_jaw_pose,
            leye_pose=smplx_leye_pose,
            reye_pose=smplx_reye_pose,
        )
        smplx_tfs = smplx_outputs["smplx_tfs"]
        smplx_verts = smplx_outputs["smplx_verts"]

        fg_implicit_cond = {"smplx": smplx_body_pose / np.pi}

        ray_dirs, cam_loc = render_utils.get_camera_params(uv, pose, intrinsics)
        batch_size, num_pixels, _ = ray_dirs.shape
        ray_dirs_tmp, _ = render_utils.get_camera_params(
            uv, torch.eye(4).to(device)[None], intrinsics
        )
        depth_scale = ray_dirs_tmp[0, :, 2:]

        ray_dirs = rearrange(ray_dirs, "b n d -> (b n) d")
        cam_loc = repeat(cam_loc, "b d -> (b n) d", n=num_pixels)

        # BN x M
        fg_z_vals = self.fg_ray_sampler.get_z_vals(
            ray_dirs, cam_loc, self, fg_implicit_cond, smplx_tfs, smplx_verts
        )
        # BN x M, BN x 1
        obj_z_vals, obj_z_eik_vals = self.obj_ray_sampler.get_z_vals(
            ray_dirs, cam_loc, self, obj_pose, self.obj_verts_cano
        )
        # BN x M, BN x 1
        bg_z_vals, bg_z_eik_vals = self.bg_ray_sampler.get_z_vals(ray_dirs, cam_loc, self)

        # BN x 3
        cam_loc = rearrange(cam_loc, "bn d -> bn 1 d")
        ray_dirs = rearrange(ray_dirs, "bn d -> bn 1 d")

        # BN x M x 3 = BN x 1 x 3 + BN x M x 1 * BN x 1 x 3
        fg_points = cam_loc + fg_z_vals[:, :, None] * ray_dirs
        num_samples = fg_points.shape[1]
        fg_points_flat = rearrange(fg_points, "bn m d -> (bn m) d")

        # BNM x 3, BNM x 1, BNM x D, BNM x 3
        fg_cano_points, fg_sdf, fg_features, fg_gradients = self.sdf_func_fg_with_gradient(
            fg_points_flat,
            fg_implicit_cond,
            smplx_tfs,
            smplx_verts,
        )

        # BN x M x 3 = BN x 1 x 3 + BN x M x 1 * BN x 1 x 3
        obj_points = cam_loc + obj_z_vals[:, :, None] * ray_dirs
        obj_points_flat = rearrange(obj_points, "bn m d -> (bn m) d")
        # BNM x 3, BNM x 1, BNM x D, BNM x 3
        obj_implicit_cond = None
        obj_cano_points, obj_sdf, obj_features, obj_gradients = self.sdf_func_obj_with_gradient(
            obj_points_flat, obj_pose, obj_implicit_cond
        )

        # BN x M x 3
        bg_points = cam_loc + bg_z_vals[:, :, None] * ray_dirs
        assert bg_points.shape == fg_points.shape
        bg_points_flat = rearrange(bg_points, "bn m d -> (bn m) d")
        bg_implicit_cond = None
        # BNM x 1, BNM x D, BNM x 3
        bg_sdf, bg_features, bg_gradients = self.sdf_func_bg_with_gradient(
            bg_points_flat, bg_implicit_cond
        )

        if self.training:
            if inputs["global_step"] > self.penetration_loss_steps:
                threshold = -0.02
                fg_pene_mask = fg_sdf[:, 0] < threshold
                obj_pene_mask = obj_sdf[:, 0] < threshold
                bg_pene_mask = bg_sdf[:, 0] < threshold
                obj_samples_fg_sdf, _, _ = self.sdf_func_fg(
                    obj_points_flat[obj_pene_mask],
                    fg_implicit_cond,
                    smplx_tfs,
                    smplx_verts,
                )
                bg_samples_fg_sdf, _, _ = self.sdf_func_fg(
                    bg_points_flat[bg_pene_mask],
                    fg_implicit_cond,
                    smplx_tfs,
                    smplx_verts,
                )
                fg_samples_obj_sdf, _, _ = self.sdf_func_obj(
                    fg_points_flat[fg_pene_mask],
                    obj_pose,
                    obj_implicit_cond,
                )
                bg_samples_obj_sdf, _, _ = self.sdf_func_obj(
                    bg_points_flat[bg_pene_mask],
                    obj_pose,
                    obj_implicit_cond,
                )
                fg_samples_bg_sdf, _ = self.sdf_func_bg(
                    fg_points_flat[fg_pene_mask], bg_implicit_cond
                )
                obj_samples_bg_sdf, _ = self.sdf_func_bg(
                    obj_points_flat[obj_pene_mask], bg_implicit_cond
                )
                # bg_samples_fg_sdf = rearrange(
                #     bg_samples_fg_sdf, "(b n m) 1 -> b n m", b=batch_size, n=num_pixels
                # )
                # fg_samples_bg_sdf = rearrange(
                #     fg_samples_bg_sdf, "(b n m) 1 -> b n m", b=batch_size, n=num_pixels
                # )
            else:
                obj_samples_fg_sdf = None
                bg_samples_fg_sdf = None
                fg_samples_obj_sdf = None
                bg_samples_obj_sdf = None
                fg_samples_bg_sdf = None
                obj_samples_bg_sdf = None
                fg_pene_mask = None
                obj_pene_mask = None
                bg_pene_mask = None

            index_off_surface, index_in_surface = self.check_off_in_surface_points_cano(
                fg_cano_points, num_samples
            )

            index_off_surface = rearrange(index_off_surface, "(b n) -> b n", b=batch_size)
            index_in_surface = rearrange(index_in_surface, "(b n) -> b n", b=batch_size)

            # sample canonical SMPLX pnts for the eikonal loss
            smplx_verts_c = self.smplx_server.verts_c  # [1, V, 3]

            # 1) Body-only surface-anchored samples with larger sigma (N)
            body_verts_c = smplx_verts_c[:, self.smplx_body_vertex_ids]  # [1, V_body, 3]
            num_body_verts = body_verts_c.shape[1]
            num_body_surface = num_pixels
            indices_body = torch.randperm(num_body_verts, device=device)[:num_body_surface]
            fg_body_anchors = body_verts_c[:, indices_body].expand(batch_size, -1, -1)
            fg_eik_body_samples = fg_body_anchors + torch.randn_like(fg_body_anchors) * 0.1

            # 2) All-vertex surface-anchored samples (face+hands included) with larger sigma (0.5N)
            num_all_verts = smplx_verts_c.shape[1]
            num_all_surface = num_pixels // 2
            indices_all = torch.randperm(num_all_verts, device=device)[:num_all_surface]
            fg_all_anchors = smplx_verts_c[:, indices_all].expand(batch_size, -1, -1)
            fg_eik_all_samples = fg_all_anchors + torch.randn_like(fg_all_anchors) * 0.1

            # 3) Uniform samples within 1.5x canonical body bbox (0.5N)
            num_uniform = num_pixels // 2
            fg_bbox_min = self.mesh_v_cano[0].min(dim=0).values  # [3]
            fg_bbox_max = self.mesh_v_cano[0].max(dim=0).values  # [3]
            fg_bbox_center = (fg_bbox_min + fg_bbox_max) * 0.5
            fg_bbox_half = (fg_bbox_max - fg_bbox_min) * 0.5 * 1.5  # 1.5x padding
            fg_eik_uniform_samples = (
                torch.rand(batch_size, num_uniform, 3, device=device)
                * (fg_bbox_half * 2)
                + (fg_bbox_center - fg_bbox_half)
            )

            # B x 2N x 3
            fg_eik_samples = torch.cat(
                [fg_eik_body_samples, fg_eik_all_samples, fg_eik_uniform_samples], dim=1
            )
            fg_eik_samples.requires_grad_()
            fg_eik_sdf = self.fg_implicit_network(fg_eik_samples, fg_implicit_cond)[..., 0:1]
            fg_grad_theta = gradient(fg_eik_samples, fg_eik_sdf)

            # Near-surface samples from ray marching (sigma=0.01)
            obj_eik_near_samples = cam_loc + obj_z_eik_vals[:, :, None] * ray_dirs
            obj_eik_near_samples = rearrange(
                obj_eik_near_samples, "(b n) 1 d -> b n d", b=batch_size
            )
            obj_eik_near_samples = self.eik_sampler.get_points(
                obj_eik_near_samples, global_ratio=0.0
            )

            if self.obj_verts_cano is not None:
                # Per-axis adaptive sigma based on object bounding box
                bbox_min = self.obj_verts_cano.min(dim=0).values
                bbox_max = self.obj_verts_cano.max(dim=0).values
                half_extents = (bbox_max - bbox_min) * 0.5  # [3]
                adaptive_sigma = half_extents * 1.0  # [3]

                # Surface-anchored samples with per-axis perturbation
                num_obj_verts = self.obj_verts_cano.shape[0]
                indices = torch.randperm(num_obj_verts, device=device)[:num_pixels]
                obj_surface_anchors = self.obj_verts_cano[indices].unsqueeze(0).expand(
                    batch_size, -1, -1
                )
                obj_eik_surface_samples = (
                    obj_surface_anchors + torch.randn_like(obj_surface_anchors) * adaptive_sigma
                )
                # B x 2N x 3 (N ray-near + N surface-anchored)
                obj_eik_samples = torch.cat(
                    [obj_eik_near_samples, obj_eik_surface_samples], dim=1
                )
            else:
                # Fallback before first mesh extraction: original uniform sampling
                obj_eik_uniform_samples = torch.empty(
                    (batch_size, num_pixels, 3), device=device
                )
                obj_eik_uniform_samples.uniform_(
                    -self.obj_bounding_sphere, self.obj_bounding_sphere
                )
                obj_eik_samples = torch.cat(
                    [obj_eik_uniform_samples, obj_eik_near_samples], dim=1
                )
            obj_eik_samples.requires_grad_()
            obj_eik_sdf = self.obj_implicit_network(obj_eik_samples, obj_implicit_cond)[..., 0:1]
            obj_grad_theta = gradient(obj_eik_samples, obj_eik_sdf)

            bg_eik_uniform_samples = torch.empty((batch_size, num_pixels, 3), device=device)
            bg_eik_uniform_samples.uniform_(-self.bg_bounding_sphere, self.bg_bounding_sphere)
            bg_eik_near_samples = cam_loc + bg_z_eik_vals[:, :, None] * ray_dirs
            bg_eik_near_samples = rearrange(bg_eik_near_samples, "(b n) 1 d -> b n d", b=batch_size)
            bg_eik_near_samples = self.eik_sampler.get_points(bg_eik_near_samples, global_ratio=0.0)
            # B x 2N x 3
            bg_eik_samples = torch.cat([bg_eik_uniform_samples, bg_eik_near_samples], dim=1)
            bg_eik_samples.requires_grad_()
            bg_eik_sdf = self.bg_implicit_network(bg_eik_samples, bg_implicit_cond)[..., 0:1]
            bg_grad_theta = gradient(bg_eik_samples, bg_eik_sdf)

            if self.reg_hands:
                num_hand_vertices = self.smplx_hand_vertex_ids.shape[0]
                sample_cano_hand_v_idx = torch.randperm(num_hand_vertices, device=device)
                sample_cano_hand_v = torch.index_select(
                    self.smplx_v_cano[:, self.smplx_hand_vertex_ids], 1, sample_cano_hand_v_idx
                )
                sample_cano_hand = self.hand_point_sampler.get_points(
                    sample_cano_hand_v, local_sigma=0.005, global_ratio=0.0
                )
                sample_cano_hand = rearrange(sample_cano_hand, "b n d -> (b n) d")

                with torch.no_grad():
                    sample_cano_hand_dist = torch.sum(
                        sample_cano_hand.detach().clone() ** 2, dim=1, keepdim=True
                    )
                    # normalize to [0, 1]
                    sample_hand_weights = (sample_cano_hand_dist - sample_cano_hand_dist.min()) / (
                        sample_cano_hand_dist.max() - sample_cano_hand_dist.min()
                    )
                    # reduce weights close to the wrist
                    sample_hand_weights = sample_hand_weights**5

                sample_hand_sdf_pred = self.fg_implicit_network(sample_cano_hand, fg_implicit_cond)[
                    0, :, 0
                ]

                gt_sign = (
                    kaolin.ops.mesh.check_sign(
                        self.smplx_v_cano, self.smplx_f_cano, sample_cano_hand[None]
                    )
                    .float()
                    .unsqueeze(-1)
                )
                gt_sign = 1 - 2 * gt_sign

                gt_df = torch.sqrt(
                    kaolin.metrics.trianglemesh.point_to_mesh_distance(
                        sample_cano_hand[None], self.smplx_face_vertices
                    )[0]
                )
                sample_hand_gt_sdf = (gt_sign[..., 0] * gt_df).squeeze()
            else:
                sample_hand_sdf_pred = None
                sample_hand_gt_sdf = None
                sample_hand_weights = None

            if self.check_self_penetration and inputs["enable_contact_loss"]:
                # we only enable this loss when optimizing pose only
                # pass smplx verts to human sdf function
                smplx_verts_fg_sdf, _, _ = self.sdf_func_fg(
                    smplx_verts[0],
                    fg_implicit_cond,
                    smplx_tfs,
                    smplx_verts,
                )
            else:
                smplx_verts_fg_sdf = None

            if self.reg_body and inputs["current_epoch"] > 10:
                # num_body_vertices = self.smplx_v_cano.shape[1]
                # sample_cano_body = self.body_point_sampler.get_points(
                #     self.smplx_v_cano, local_sigma=0.0001, global_ratio=0.0
                # )
                sample_cano_body = self.body_point_sampler.get_points(
                    self.smplx_body_interior_points, local_sigma=0.008, global_ratio=0.0
                )
                sample_cano_feet = self.body_point_sampler.get_points(
                    self.smplx_feet_interior_points, local_sigma=0.003, global_ratio=0.0
                )
                sample_cano_body = torch.cat([sample_cano_body, sample_cano_feet], dim=1)
                sample_body_sdf = self.fg_implicit_network(sample_cano_body, fg_implicit_cond)[
                    0, :, 0
                ]
            else:
                sample_body_sdf = None

            # hand-object SDF probes for the (disabled) hand_contact loss
            smplx_left_hand_obj_sdf = None
            smplx_right_hand_obj_sdf = None

            if inputs["contact_label"]:
                contact_vertices = inputs["contact_vertices"]
                smplx_contact_verts = smplx_verts[0, contact_vertices[0], :]
                smplx_contact_verts_obj_sdf, _, _ = self.sdf_func_obj(
                    smplx_contact_verts,
                    obj_pose,
                    obj_implicit_cond,
                )
            else:
                smplx_contact_verts_obj_sdf = None

        else:
            fg_grad_theta = None
            obj_grad_theta = None
            bg_grad_theta = None
            sample_hand_sdf_pred = None
            sample_hand_gt_sdf = None
            sample_hand_weights = None

        view_dirs = repeat(ray_dirs, "bn 1 d -> bn m d", m=num_samples)
        view_dirs = rearrange(view_dirs, "bn m d -> (bn m) d")

        # BNM x 3, BNM x 3
        fg_rgb_flat, fg_normals_flat = self.get_fg_rgb_value(
            fg_cano_points,
            view_dirs,
            fg_features,
            fg_gradients,
            fg_implicit_cond,
        )

        # B x D
        if "frame_encoding" in self.obj_rendering_mode:
            obj_frame_code = self.obj_frame_encoder(inputs["idx"])
        else:
            obj_frame_code = None
        obj_rgb_flat, obj_normals_flat = self.get_obj_rgb_value(
            obj_cano_points, view_dirs, obj_features, obj_gradients, obj_frame_code
        )

        # B x D
        if "frame_encoding" in self.bg_rendering_mode:
            bg_frame_code = self.bg_frame_encoder(inputs["idx"])
        else:
            bg_frame_code = None
        bg_rgb_flat, bg_normals_flat = self.get_bg_rgb_value(
            bg_points_flat, view_dirs, bg_features, bg_gradients, bg_frame_code
        )

        fg_z_vals = rearrange(fg_z_vals, "bn m -> bn m 1")
        obj_z_vals = rearrange(obj_z_vals, "bn m -> bn m 1")
        bg_z_vals = rearrange(bg_z_vals, "bn m -> bn m 1")
        fg_sdf = rearrange(fg_sdf, "(bn m) d -> bn m d", m=num_samples)
        obj_sdf = rearrange(obj_sdf, "(bn m) d -> bn m d", m=num_samples)
        bg_sdf = rearrange(bg_sdf, "(bn m) d -> bn m d", m=num_samples)
        fg_rgb = rearrange(fg_rgb_flat, "(bn m) d -> bn m d", m=num_samples)
        obj_rgb = rearrange(obj_rgb_flat, "(bn m) d -> bn m d", m=num_samples)
        bg_rgb = rearrange(bg_rgb_flat, "(bn m) d -> bn m d", m=num_samples)
        fg_normals = rearrange(fg_normals_flat, "(bn m) d -> bn m d", m=num_samples)
        obj_normals = rearrange(obj_normals_flat, "(bn m) d -> bn m d", m=num_samples)
        bg_normals = rearrange(bg_normals_flat, "(bn m) d -> bn m d", m=num_samples)

        # rendering_outputs = self.fg_bg_volume_rendering(
        #     fg_z_vals, bg_z_vals, fg_sdf, bg_sdf, fg_rgb, bg_rgb, fg_normals, bg_normals
        # )
        rendering_outputs = self.fg_obj_bg_volume_rendering(
            z_vals_list=[fg_z_vals, obj_z_vals, bg_z_vals],
            sdf_list=[fg_sdf, obj_sdf, bg_sdf],
            rgb_list=[fg_rgb, obj_rgb, bg_rgb],
            normals_list=[fg_normals, obj_normals, bg_normals],
        )

        fg_sdf = rearrange(fg_sdf, "(b n) m 1 -> b n m", b=batch_size, n=num_pixels)
        obj_sdf = rearrange(obj_sdf, "(b n) m 1 -> b n m", b=batch_size, n=num_pixels)
        bg_sdf = rearrange(bg_sdf, "(b n) m 1 -> b n m", b=batch_size, n=num_pixels)

        weights = rendering_outputs["weights"]
        acc_map = torch.sum(weights, 1)
        acc_map_fg = torch.sum(rendering_outputs["fg_weights"], 1)
        acc_map_obj = torch.sum(rendering_outputs["obj_weights"], 1)
        acc_map_bg = torch.sum(rendering_outputs["bg_weights"], 1)
        fg_acc_map = torch.sum(rendering_outputs["fg_weights_only"], 1)
        obj_acc_map = torch.sum(rendering_outputs["obj_weights_only"], 1)
        bg_acc_map = torch.sum(rendering_outputs["bg_weights_only"], 1)

        rgb = torch.sum(weights * rendering_outputs["rgb"], 1)
        fg_rgb = torch.sum(rendering_outputs["fg_weights_only"] * fg_rgb, 1)
        obj_rgb = torch.sum(rendering_outputs["obj_weights_only"] * obj_rgb, 1)
        bg_rgb = torch.sum(rendering_outputs["bg_weights_only"] * bg_rgb, 1)

        normal = torch.sum(weights * rendering_outputs["normals"], 1)
        rot = pose[0, :3, :3].permute(1, 0).contiguous()
        normal = rot @ normal.permute(1, 0)
        normal = normal.permute(1, 0).contiguous()
        # we only normalize the compositional normal as it does not have very small values
        normal = torch.nn.functional.normalize(normal, p=2, dim=-1)

        depth = torch.sum(weights * rendering_outputs["z_vals"], 1)
        depth = depth_scale * depth

        rgb = rearrange(rgb, "(b n) c -> b n c", b=batch_size)
        fg_rgb = rearrange(fg_rgb, "(b n) c -> b n c", b=batch_size)
        obj_rgb = rearrange(obj_rgb, "(b n) c -> b n c", b=batch_size)
        bg_rgb = rearrange(bg_rgb, "(b n) c -> b n c", b=batch_size)
        normal = rearrange(normal, "(b n) c -> b n c", b=batch_size)
        depth = rearrange(depth, "(b n) 1 -> b n", b=batch_size)
        acc_map = rearrange(acc_map, "(b n) 1 -> b n", b=batch_size)
        acc_map_fg = rearrange(acc_map_fg, "(b n) 1 -> b n", b=batch_size)
        acc_map_obj = rearrange(acc_map_obj, "(b n) 1 -> b n", b=batch_size)
        acc_map_bg = rearrange(acc_map_bg, "(b n) 1 -> b n", b=batch_size)
        fg_acc_map = rearrange(fg_acc_map, "(b n) 1 -> b n", b=batch_size)
        obj_acc_map = rearrange(obj_acc_map, "(b n) 1 -> b n", b=batch_size)
        bg_acc_map = rearrange(bg_acc_map, "(b n) 1 -> b n", b=batch_size)

        if self.training:
            output = {
                "rgb": rgb,
                "fg_rgb": fg_rgb,
                "obj_rgb": obj_rgb,
                "bg_rgb": bg_rgb,
                "normal": normal,
                "depth": depth,
                "index_off_surface": index_off_surface,
                "index_in_surface": index_in_surface,
                "acc_map": acc_map,
                "fg_acc_map": fg_acc_map,
                "obj_acc_map": obj_acc_map,
                "bg_acc_map": bg_acc_map,
                "acc_map_fg": acc_map_fg,
                "acc_map_obj": acc_map_obj,
                "acc_map_bg": acc_map_bg,
                "fg_sdf": fg_sdf,
                "obj_sdf": obj_sdf,
                "bg_sdf": bg_sdf,
                "obj_samples_fg_sdf": obj_samples_fg_sdf,
                "bg_samples_fg_sdf": bg_samples_fg_sdf,
                "fg_samples_obj_sdf": fg_samples_obj_sdf,
                "bg_samples_obj_sdf": bg_samples_obj_sdf,
                "fg_samples_bg_sdf": fg_samples_bg_sdf,
                "obj_samples_bg_sdf": obj_samples_bg_sdf,
                "fg_pene_mask": fg_pene_mask,
                "obj_pene_mask": obj_pene_mask,
                "bg_pene_mask": bg_pene_mask,
                "fg_grad_theta": fg_grad_theta,
                "obj_grad_theta": obj_grad_theta,
                "bg_grad_theta": bg_grad_theta,
                "hand_sdf_gt": sample_hand_gt_sdf,
                "hand_sdf_pred": sample_hand_sdf_pred,
                "hand_weights": sample_hand_weights,
                "body_sdf": sample_body_sdf,
                # "hand_obj_sdf": smplx_hand_obj_sdf,
                "left_hand_obj_sdf": smplx_left_hand_obj_sdf,
                "right_hand_obj_sdf": smplx_right_hand_obj_sdf,
                "contact_verts_obj_sdf": smplx_contact_verts_obj_sdf,
                "smplx_verts_fg_sdf": smplx_verts_fg_sdf,
                # "epoch": inputs["current_epoch"],
                "obj_pose_prev": inputs["obj_pose_prev"],
                "obj_pose_cur": inputs["obj_pose_cur"],
                "obj_pose_next": inputs["obj_pose_next"],
                "body_model_params_prev": inputs["body_model_params_prev"],
                "body_model_params_cur": inputs["body_model_params_cur"],
                "body_model_params_next": inputs["body_model_params_next"],
                "smplx_verts": smplx_verts,
                "smplx_verts_prev": inputs.get("smplx_verts_prev"),
                "smplx_verts_next": inputs.get("smplx_verts_next"),
                "obj_verts_cano": self.obj_verts_cano,
            }
        else:
            fg_normal = torch.sum(rendering_outputs["fg_weights_only"] * fg_normals, 1)
            obj_normal = torch.sum(rendering_outputs["obj_weights_only"] * obj_normals, 1)
            bg_normal = torch.sum(rendering_outputs["bg_weights_only"] * bg_normals, 1)
            fg_normal = rot @ fg_normal.permute(1, 0)
            fg_normal = fg_normal.permute(1, 0).contiguous()
            # we should not do a normalization here
            # as we might have very small values in empty regions after volume rendering
            # fg_normal = torch.nn.functional.normalize(fg_normal, p=2, dim=-1)
            obj_normal = rot @ obj_normal.permute(1, 0)
            obj_normal = obj_normal.permute(1, 0).contiguous()
            bg_normal = rot @ bg_normal.permute(1, 0)
            bg_normal = bg_normal.permute(1, 0).contiguous()
            fg_normal = rearrange(fg_normal, "(b n) m -> b n m", b=batch_size)
            obj_normal = rearrange(obj_normal, "(b n) m -> b n m", b=batch_size)
            bg_normal = rearrange(bg_normal, "(b n) m -> b n m", b=batch_size)
            # bg_normal = torch.nn.functional.normalize(bg_normal, p=2, dim=-1)

            fg_depth = torch.sum(rendering_outputs["fg_weights_only"] * fg_z_vals, 1)
            fg_depth = depth_scale * fg_depth
            obj_depth = torch.sum(rendering_outputs["obj_weights_only"] * obj_z_vals, 1)
            obj_depth = depth_scale * obj_depth
            bg_depth = torch.sum(rendering_outputs["bg_weights_only"] * bg_z_vals, 1)
            bg_depth = depth_scale * bg_depth
            fg_depth = rearrange(fg_depth, "(b n) 1 -> b n", b=batch_size)
            obj_depth = rearrange(obj_depth, "(b n) 1 -> b n", b=batch_size)
            bg_depth = rearrange(bg_depth, "(b n) 1 -> b n", b=batch_size)
            output = {
                "fg_acc_map": fg_acc_map,
                "obj_acc_map": obj_acc_map,
                "bg_acc_map": bg_acc_map,
                "rgb": rgb,
                "fg_rgb": fg_rgb,
                "obj_rgb": obj_rgb,
                "bg_rgb": bg_rgb,
                "normal": normal,
                "fg_normal": fg_normal,
                "obj_normal": obj_normal,
                "bg_normal": bg_normal,
                "depth": depth,
                "fg_depth": fg_depth,
                "obj_depth": obj_depth,
                "bg_depth": bg_depth,
            }
        return output

    def get_fg_rgb_value(
        self,
        x_c,
        view_dirs,
        feature_vectors,
        gradients,
        cond,
    ):
        normals = nn.functional.normalize(gradients, dim=-1, eps=1e-6)
        fg_rendering_output = self.fg_rendering_network(
            x_c,
            normals,
            view_dirs,
            cond["smplx"],
            feature_vectors,
        )

        rgb_vals = fg_rendering_output[:, :3]
        return rgb_vals, normals

    def get_obj_rgb_value(
        self,
        x_c,
        view_dirs,
        obj_features,
        obj_gradients,
        frame_code,
    ):
        obj_normals = nn.functional.normalize(obj_gradients, dim=-1, eps=1e-6)
        obj_rendering_output = self.obj_rendering_network(
            x_c,
            obj_normals,
            view_dirs,
            body_pose=None,
            feature_vectors=obj_features,
            frame_latent_code=frame_code,
        )

        obj_rgb_vals = obj_rendering_output[:, :3]
        return obj_rgb_vals, obj_normals

    def get_bg_rgb_value(
        self,
        x,
        view_dirs,
        bg_features,
        bg_gradients,
        frame_code,
    ):
        bg_normals = nn.functional.normalize(bg_gradients, dim=-1, eps=1e-6)
        bg_rendering_output = self.bg_rendering_network(
            x,
            bg_normals,
            view_dirs,
            body_pose=None,
            feature_vectors=bg_features,
            frame_latent_code=frame_code,
        )

        bg_rgb_vals = bg_rendering_output[:, :3]
        return bg_rgb_vals, bg_normals

    def get_weights(self, density, dists):
        # LOG SPACE
        free_energy = dists * density[:, :, 0]
        # add 0 for transperancy 1 at t_0
        shifted_free_energy = torch.cat(
            [
                torch.zeros((dists.shape[0], 1), device=dists.device),
                free_energy[:, :-1],
            ],
            dim=-1,
        )
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        # probability of everything is empty up to now
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
        weights = alpha * transmittance  # probability of the ray hits something here
        return weights[:, :, None]

    def fg_obj_bg_volume_rendering(self, z_vals_list, sdf_list, rgb_list, normals_list):
        # z_vals: BN x M x 1, sdf: BN x M x 1, rgb: BN x M x 3, normals: BN x M x 3
        fg_z_vals, obj_z_vals, bg_z_vals = z_vals_list
        fg_sdf, obj_sdf, bg_sdf = sdf_list
        fg_rgb, obj_rgb, bg_rgb = rgb_list
        fg_normals, obj_normals, bg_normals = normals_list
        num_samples = fg_z_vals.shape[1]

        # density: BN x M x 1
        fg_density = self.fg_density(fg_sdf)
        obj_density = self.obj_density(obj_sdf)
        bg_density = self.bg_density(bg_sdf)

        # BN x M x D
        z_vals = torch.cat([fg_z_vals, obj_z_vals, bg_z_vals], dim=1)
        density = torch.cat([fg_density, obj_density, bg_density], dim=1)
        rgb = torch.cat([fg_rgb, obj_rgb, bg_rgb], dim=1)
        normals = torch.cat([fg_normals, obj_normals, bg_normals], dim=1)
        z_vals, sort_idx = torch.sort(z_vals, dim=1)
        density = torch.gather(density, 1, sort_idx)
        sort_idx = repeat(sort_idx, "bn m 1 -> bn m d", d=3)
        rgb = torch.gather(rgb, 1, sort_idx)
        normals = torch.gather(normals, 1, sort_idx)

        dists = z_vals[:, 1:, 0] - z_vals[:, :-1, 0]
        last_dist = torch.full((z_vals.shape[0], 1), 1e10, device=z_vals.device)
        dists = torch.cat([dists, last_dist], -1)
        fg_dists = fg_z_vals[:, 1:, 0] - fg_z_vals[:, :-1, 0]
        # do not concate infty here
        fg_dists = torch.cat([fg_dists, fg_dists[:, -1:]], -1)
        obj_dists = obj_z_vals[:, 1:, 0] - obj_z_vals[:, :-1, 0]
        obj_dists = torch.cat([obj_dists, obj_dists[:, -1:]], -1)
        bg_dists = bg_z_vals[:, 1:, 0] - bg_z_vals[:, :-1, 0]
        bg_dists = torch.cat([bg_dists, last_dist], -1)
        weights = self.get_weights(density, dists)
        fg_weights_only = self.get_weights(fg_density, fg_dists)
        obj_weights_only = self.get_weights(obj_density, obj_dists)
        bg_weights_only = self.get_weights(bg_density, bg_dists)

        # BNxMxD -> BNxMx1
        sort_idx = sort_idx[:, :, 0:1]
        inverse_sort_idx = torch.argsort(sort_idx, dim=1)
        assert z_vals.shape[1] == num_samples * 3
        fg_weights = torch.gather(weights, 1, inverse_sort_idx[:, :num_samples, :])
        obj_weights = torch.gather(
            weights, 1, inverse_sort_idx[:, num_samples : 2 * num_samples, :]
        )
        bg_weights = torch.gather(weights, 1, inverse_sort_idx[:, 2 * num_samples :, :])

        outputs = {
            "z_vals": z_vals,
            "rgb": rgb,
            "normals": normals,
            "weights": weights,
            "fg_weights": fg_weights,
            "obj_weights": obj_weights,
            "bg_weights": bg_weights,
            "fg_weights_only": fg_weights_only,
            "obj_weights_only": obj_weights_only,
            "bg_weights_only": bg_weights_only,
        }
        return outputs

    def fg_bg_volume_rendering(
        self, fg_z_vals, bg_z_vals, fg_sdf, bg_sdf, fg_rgb, bg_rgb, fg_normals, bg_normals
    ):
        # z_vals: BN x M x 1, sdf: BN x M x 1, rgb: BN x M x 3, normals: BN x M x 3
        # density: BN x M x 1
        fg_density = self.fg_density(fg_sdf)
        bg_density = self.bg_density(bg_sdf)

        # BN x M x D
        z_vals = torch.cat([fg_z_vals, bg_z_vals], dim=1)
        density = torch.cat([fg_density, bg_density], dim=1)
        rgb = torch.cat([fg_rgb, bg_rgb], dim=1)
        normals = torch.cat([fg_normals, bg_normals], dim=1)
        z_vals, sort_idx = torch.sort(z_vals, dim=1)
        density = torch.gather(density, 1, sort_idx)
        sort_idx = repeat(sort_idx, "bn m 1 -> bn m d", d=3)
        rgb = torch.gather(rgb, 1, sort_idx)
        normals = torch.gather(normals, 1, sort_idx)

        dists = z_vals[:, 1:, 0] - z_vals[:, :-1, 0]
        last_dist = torch.full((z_vals.shape[0], 1), 1e10, device=z_vals.device)
        dists = torch.cat([dists, last_dist], -1)
        fg_dists = fg_z_vals[:, 1:, 0] - fg_z_vals[:, :-1, 0]
        # do not concate infty here
        fg_dists = torch.cat([fg_dists, fg_dists[:, -1:]], -1)
        bg_dists = bg_z_vals[:, 1:, 0] - bg_z_vals[:, :-1, 0]
        bg_dists = torch.cat([bg_dists, last_dist], -1)
        weights = self.get_weights(density, dists)
        fg_weights_only = self.get_weights(fg_density, fg_dists)
        bg_weights_only = self.get_weights(bg_density, bg_dists)

        # BNxMxD -> BNxMx1
        sort_idx = sort_idx[:, :, 0:1]
        inverse_sort_idx = torch.argsort(sort_idx, dim=1)
        num_samples = z_vals.shape[1] // 2
        assert z_vals.shape[1] == num_samples * 2
        fg_weights = torch.gather(weights, 1, inverse_sort_idx[:, :num_samples, :])
        bg_weights = torch.gather(weights, 1, inverse_sort_idx[:, num_samples:, :])

        outputs = {
            "z_vals": z_vals,
            "rgb": rgb,
            "normals": normals,
            "weights": weights,
            "fg_weights": fg_weights,
            "bg_weights": bg_weights,
            "fg_weights_only": fg_weights_only,
            "bg_weights_only": bg_weights_only,
        }
        return outputs


def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0][:, :, -3:]
    return points_grad
