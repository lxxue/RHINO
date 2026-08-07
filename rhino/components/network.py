import numpy as np
import torch
import torch.nn as nn

from rhino.components.embedder import get_embedder

# from rhino.utils.misc import nan_hook


class ImplicitNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        dims = [cfg.d_in] + list(cfg.dims) + [cfg.d_out + cfg.feature_vector_size]
        self.num_layers = len(dims)
        self.skip_in = cfg.skip_in
        self.embed_fn = None
        self.cfg = cfg
        self.sdf_bounding_sphere = cfg.scene_bounding_sphere
        self.sphere_scale = 1.0

        if cfg.multires > 0:
            embed_fn, input_ch = get_embedder(
                cfg.multires, input_dims=cfg.d_in, mode=cfg.embedder_mode
            )
            self.embed_fn = embed_fn
            dims[0] = input_ch
        self.cond = cfg.cond
        if self.cond == "smpl":
            self.cond_layer = [0]
            self.cond_dim = 69
        elif self.cond == "frame":
            self.cond_layer = [0]
            self.cond_dim = 32
        elif self.cond == "smplx":
            self.cond_layer = [0]
            self.cond_dim = 63
        elif self.cond is not None:
            raise NotImplementedError
        self.dim_pose_embed = cfg.dim_pose_embed
        if self.dim_pose_embed > 0:
            self.lin_p0 = nn.Linear(self.cond_dim, self.dim_pose_embed)
            self.cond_dim = self.dim_pose_embed

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            if self.cond is not None and l in self.cond_layer:
                lin = nn.Linear(dims[l] + self.cond_dim, out_dim)
            else:
                lin = nn.Linear(dims[l], out_dim)
            if cfg.init == "geometry":
                if l == self.num_layers - 2:
                    if not cfg.inside_outside:
                        torch.nn.init.normal_(
                            lin.weight,
                            mean=np.sqrt(np.pi) / np.sqrt(dims[l]),
                            std=0.0001,
                        )
                        torch.nn.init.constant_(lin.bias, -cfg.bias)
                    else:
                        torch.nn.init.normal_(
                            lin.weight,
                            mean=-np.sqrt(np.pi) / np.sqrt(dims[l]),
                            std=0.0001,
                        )
                        torch.nn.init.constant_(lin.bias, cfg.bias)

                elif cfg.multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif cfg.multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            if cfg.init == "zero":
                init_val = 1e-5
                if l == self.num_layers - 2:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.uniform_(lin.weight, -init_val, init_val)
            if cfg.weight_norm:
                lin = nn.utils.parametrizations.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.softplus = nn.Softplus(beta=100)

    def forward(self, input, cond):
        if input.ndim == 2:
            input = input.unsqueeze(0)

        num_batch, num_point, num_dim = input.shape

        if num_batch * num_point == 0:
            return input

        input = input.reshape(num_batch * num_point, num_dim)
        # input_embed = input_embed if self.embed_fn is None else self.embed_fn(input_embed)

        if self.cond is not None:
            num_batch, num_cond = cond[self.cond].shape

            input_cond = cond[self.cond].unsqueeze(1).expand(num_batch, num_point, num_cond)

            input_cond = input_cond.reshape(num_batch * num_point, num_cond)

            if self.dim_pose_embed > 0:
                input_cond = self.lin_p0(input_cond)

        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if self.cond is not None and l in self.cond_layer:
                x = torch.cat([x, input_cond], dim=-1)
            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.softplus(x)

        x = x.reshape(num_batch, num_point, -1)

        return x

    # def gradient(self, x, cond):
    #     x.requires_grad_(True)
    #     y = self.forward(x, cond)[0, :, :1]
    #     d_output = torch.ones_like(y, requires_grad=False, device=y.device)
    #     gradients = torch.autograd.grad(
    #         outputs=y,
    #         inputs=x,
    #         grad_outputs=d_output,
    #         create_graph=True,
    #         retain_graph=True,
    #         only_inputs=True,
    #     )[0]
    #     return gradients

    # def get_outputs(self, x, cond, no_sdf_clamping=False):
    #     x.requires_grad_(True)
    #     output = self.forward(x, cond)[0]
    #     sdf = output[:, :1]
    #     """ Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded """
    #     if self.sdf_bounding_sphere > 0.0 and not no_sdf_clamping:
    #         sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2, 1, keepdim=True))
    #         sdf = torch.minimum(sdf, sphere_sdf)

    #     feature_vectors = output[:, 1:]
    #     d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
    #     gradients = torch.autograd.grad(
    #         outputs=sdf,
    #         inputs=x,
    #         grad_outputs=d_output,
    #         create_graph=True,
    #         retain_graph=True,
    #         only_inputs=True,
    #     )[0]

    #     return sdf, feature_vectors, gradients

    # def get_sdf_vals(self, x, cond, no_sdf_clamping=False):
    #     sdf = self.forward(x, cond)[0, :, :1]
    #     """ Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded """
    #     if self.sdf_bounding_sphere > 0.0 and not no_sdf_clamping:
    #         sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2, 1, keepdim=True))
    #         sdf = torch.minimum(sdf, sphere_sdf)
    #     return sdf


class RenderingNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.mode = cfg.mode
        dims = [cfg.d_in + cfg.feature_vector_size] + list(cfg.dims) + [cfg.d_out]

        self.embedview_fn = None
        if cfg.multires_view > 0:
            embedview_fn, input_ch = get_embedder(cfg.multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += input_ch - 3

        if self.mode == "idr_frame_encoding":
            dims[0] += cfg.dim_frame_encoding
        elif self.mode == "idr_pose_no_view":
            self.dim_cond_embed = cfg.dim_cond_embed
            self.cond_dim = 69
            self.lin_pose = torch.nn.Linear(self.cond_dim, self.dim_cond_embed)
        elif self.mode == "idr_smplx_pose":
            self.dim_cond_embed = cfg.dim_cond_embed
            self.cond_dim = 63
            self.lin_pose = torch.nn.Linear(self.cond_dim, self.dim_cond_embed)
        else:
            raise NotImplementedError

        self.num_layers = len(dims)
        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if cfg.weight_norm:
                lin = nn.utils.parametrizations.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        points,
        normals,
        view_dirs,
        body_pose,
        feature_vectors,
        frame_latent_code=None,
    ):
        if self.embedview_fn is not None:
            if self.mode == "idr_frame_encoding":
                view_dirs = self.embedview_fn(view_dirs)

        if self.mode == "idr":
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == "idr_frame_encoding":
            frame_latent_code = frame_latent_code.expand(view_dirs.shape[0], -1)
            rendering_input = torch.cat(
                [points, view_dirs, normals, feature_vectors, frame_latent_code], dim=-1
            )
        elif self.mode == "idr_pose":
            num_points = points.shape[0]
            body_pose = body_pose.unsqueeze(1).expand(-1, num_points, -1).reshape(num_points, -1)
            body_pose = self.lin_pose(body_pose)
            rendering_input = torch.cat(
                [points, view_dirs, normals, body_pose, feature_vectors], dim=-1
            )
        elif self.mode == "idr_pose_no_view" or self.mode == "idr_smplx_pose":
            num_points = points.shape[0]
            body_pose = body_pose.unsqueeze(1).expand(-1, num_points, -1).reshape(num_points, -1)
            body_pose = self.lin_pose(body_pose)
            rendering_input = torch.cat([points, normals, body_pose, feature_vectors], dim=-1)
        else:
            raise NotImplementedError

        x = rendering_input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.relu(x)

        x = self.sigmoid(x)
        return x
