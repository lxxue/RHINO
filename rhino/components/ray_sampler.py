import abc
from collections.abc import Callable

import torch

from rhino.utils import render_utils


class RaySampler(metaclass=abc.ABCMeta):
    def __init__(self, near, far):
        self.near = near
        self.far = far

    @abc.abstractmethod
    def get_z_vals(self, ray_dirs, cam_loc, model):
        pass


class UniformSampler(RaySampler):
    def __init__(
        self,
        scene_bounding_sphere,
        near,
        N_samples,
        take_sphere_intersection=False,
        far=-1.0,
    ):
        super().__init__(near, 2.0 * scene_bounding_sphere if far == -1 else far)
        self.N_samples = N_samples
        self.scene_bounding_sphere = scene_bounding_sphere
        self.take_sphere_intersection = take_sphere_intersection

    @torch.no_grad()
    def get_z_vals(self, ray_dirs, cam_loc, model):
        if not self.take_sphere_intersection:
            near = ray_dirs.new_full((ray_dirs.shape[0], 1), self.near)
            far = ray_dirs.new_full((ray_dirs.shape[0], 1), self.far)
        else:
            sphere_intersections = render_utils.get_sphere_intersections(
                cam_loc, ray_dirs, r=self.scene_bounding_sphere
            )
            near = ray_dirs.new_full((ray_dirs.shape[0], 1), self.near)
            far = sphere_intersections[:, 1:]

        t_vals = torch.linspace(0.0, 1.0, steps=self.N_samples, device=ray_dirs.device)
        z_vals = near * (1.0 - t_vals) + far * t_vals

        if model.training:
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            z_vals = lower + (upper - lower) * torch.rand_like(z_vals)
        return z_vals


def compute_bbox_nf(rays_o, rays_d, bbox_min, bbox_max, _bounding_sphere):
    """Compute per-ray intersections with an axis-aligned bounding box."""
    delta = rays_d.new_tensor(1e-10)
    safe_dirs = torch.sign(rays_d) * torch.maximum(torch.abs(rays_d), delta)
    safe_dirs = torch.where(rays_d == 0, delta, safe_dirs)
    inverse_dirs = safe_dirs.reciprocal()

    if not torch.isfinite(inverse_dirs).all():
        raise FloatingPointError("Nonfinite inverse ray direction in bounding-box sampling")

    bounds_1 = (bbox_min - rays_o) * inverse_dirs
    bounds_2 = (bbox_max - rays_o) * inverse_dirs
    bounds_near = torch.minimum(bounds_1, bounds_2)
    bounds_far = torch.maximum(bounds_1, bounds_2)
    near = bounds_near.max(dim=-1).values
    far = bounds_far.min(dim=-1).values

    # A missed box contributes a zero-length interval at its far intersection.
    near = torch.where(far < near, far, near)
    return near, far


class BBoxRaySampler(RaySampler):
    def __init__(self, N_samples, near, human_bound_sphere):
        super().__init__(near, far=2 * human_bound_sphere)
        self.N_samples = N_samples
        self.human_bound_sphere = human_bound_sphere

    @torch.no_grad()
    def get_z_vals(self, ray_dir, cam_loc, bbox_min, bbox_max, is_training):
        near, far = compute_bbox_nf(
            cam_loc, ray_dir, bbox_min, bbox_max, self.human_bound_sphere
        )
        near_limit = near.new_tensor(self.near)
        far_limit = far.new_tensor(self.far)
        near = near.clamp(min=near_limit, max=far_limit)
        far = far.clamp(min=near_limit, max=far_limit)

        t_vals = torch.linspace(0.0, 1.0, steps=self.N_samples, device=near.device)
        z_vals = near[..., None] * (1.0 - t_vals) + far[..., None] * t_vals
        if is_training:
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            z_vals = lower + (upper - lower) * torch.rand_like(z_vals)
        return near, far, z_vals


class _AdaptiveErrorBoundSampler(RaySampler):
    """Shared implementation of the VolSDF adaptive error-bound sampler."""

    def __init__(
        self,
        scene_bounding_sphere,
        near,
        N_samples,
        N_samples_eval,
        N_samples_extra,
        eps,
        beta_iters,
        max_total_iters,
        add_tiny=0.0,
    ):
        super().__init__(near, 2.0 * scene_bounding_sphere)
        if N_samples < 2 or N_samples_eval < 2:
            raise ValueError("N_samples and N_samples_eval must both be at least 2")
        if N_samples_extra < 0:
            raise ValueError("N_samples_extra must be non-negative")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if beta_iters < 0 or max_total_iters < 1:
            raise ValueError("beta_iters must be non-negative and max_total_iters positive")
        if add_tiny < 0:
            raise ValueError("add_tiny must be non-negative")

        self.N_samples = N_samples
        self.N_samples_eval = N_samples_eval
        self.N_samples_extra = N_samples_extra
        self.eps = eps
        self.beta_iters = beta_iters
        self.max_total_iters = max_total_iters
        self.scene_bounding_sphere = scene_bounding_sphere
        self.add_tiny = add_tiny

    @staticmethod
    def _compute_d_star(sdf, z_vals):
        d = sdf.reshape(z_vals.shape)
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        a, b, c = dists, d[:, :-1].abs(), d[:, 1:].abs()
        first_cond = a.square() + b.square() <= c.square()
        second_cond = a.square() + c.square() <= b.square()
        d_star = torch.zeros_like(dists)
        d_star[first_cond] = b[first_cond]
        d_star[second_cond] = c[second_cond]
        semiperimeter = (a + b + c) / 2.0
        area_term = (
            semiperimeter
            * (semiperimeter - a)
            * (semiperimeter - b)
            * (semiperimeter - c)
        )
        triangle_mask = ~first_cond & ~second_cond & (b + c - a > 0)
        d_star[triangle_mask] = (
            2.0 * torch.sqrt(area_term[triangle_mask] + 1e-8)
        ) / a[triangle_mask]
        same_sign = d[:, 1:].sign() * d[:, :-1].sign() == 1
        return same_sign * d_star, dists

    @staticmethod
    def _error_bound(beta, density_fn, sdf, z_vals, dists, d_star):
        density = density_fn(sdf.reshape(z_vals.shape), beta=beta)
        shifted_free_energy = torch.cat(
            [torch.zeros_like(dists[:, :1]), dists * density[:, :-1]], dim=-1
        )
        integral_estimation = torch.cumsum(shifted_free_energy, dim=-1)
        error_per_section = torch.exp(-d_star / beta) * dists.square() / (4 * beta.square())
        error_integral = torch.cumsum(error_per_section, dim=-1)
        bound_opacity = (
            torch.clamp(torch.exp(error_integral), max=1.0e6) - 1.0
        ) * torch.exp(-integral_estimation[:, :-1])
        return bound_opacity.max(-1).values

    @staticmethod
    def _normalize_pdf(pdf):
        pdf_sum = pdf.sum(dim=-1, keepdim=True)
        valid = torch.isfinite(pdf).all(dim=-1, keepdim=True) & (pdf_sum > 0)
        normalized = pdf / torch.where(valid, pdf_sum, torch.ones_like(pdf_sum))
        uniform = torch.full_like(pdf, 1.0 / pdf.shape[-1])
        return torch.where(valid, normalized, uniform)

    @classmethod
    def _invert_cdf(cls, bins, pdf, num_samples, deterministic):
        pdf = cls._normalize_pdf(pdf)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
        if deterministic:
            u = torch.linspace(0.0, 1.0, steps=num_samples, device=bins.device)
            u = u.unsqueeze(0).expand(cdf.shape[0], -1)
        else:
            u = torch.rand((*cdf.shape[:-1], num_samples), device=bins.device)
        u = u.contiguous()

        indices = torch.searchsorted(cdf, u, right=True)
        below = (indices - 1).clamp_min(0)
        above = indices.clamp_max(cdf.shape[-1] - 1)
        gather_indices = torch.stack([below, above], dim=-1)
        matched_shape = [gather_indices.shape[0], gather_indices.shape[1], cdf.shape[-1]]
        cdf_values = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, gather_indices)
        bin_values = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, gather_indices)
        denominator = cdf_values[..., 1] - cdf_values[..., 0]
        denominator = torch.where(
            denominator < 1e-5, torch.ones_like(denominator), denominator
        )
        t = (u - cdf_values[..., 0]) / denominator
        return bin_values[..., 0] + t * (bin_values[..., 1] - bin_values[..., 0])

    def _adaptive_sample(
        self,
        z_vals,
        ray_dirs,
        cam_loc,
        model,
        get_beta: Callable,
        density_fn: Callable,
        sdf_fn: Callable,
    ):
        beta0 = get_beta().detach()
        if beta0.numel() != 1:
            raise ValueError(f"Expected scalar density beta, got shape {tuple(beta0.shape)}")
        beta0 = beta0.reshape(())

        samples = z_vals
        samples_idx = None
        sdf = None
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        log_eps = torch.log(z_vals.new_tensor(self.eps + 1.0))
        beta = torch.sqrt(dists.square().sum(-1) / (4.0 * log_eps) + 1e-8)

        for total_iters in range(1, self.max_total_iters + 1):
            points = cam_loc.unsqueeze(1) + samples.unsqueeze(2) * ray_dirs.unsqueeze(1)
            samples_sdf = sdf_fn(points.reshape(-1, 3))
            if samples_idx is None:
                sdf = samples_sdf
            else:
                if sdf is None:
                    raise RuntimeError("Missing accumulated SDF samples")
                previous_count = z_vals.shape[1] - samples.shape[1]
                sdf_merge = torch.cat(
                    [
                        sdf.reshape(-1, previous_count),
                        samples_sdf.reshape(-1, samples.shape[1]),
                    ],
                    dim=-1,
                )
                sdf = torch.gather(sdf_merge, 1, samples_idx).reshape(-1, 1)

            d_star, dists = self._compute_d_star(sdf, z_vals)
            current_error = self._error_bound(
                beta0, density_fn, sdf, z_vals, dists, d_star
            )
            beta = torch.where(current_error <= self.eps, beta0, beta)
            beta_min = torch.ones_like(beta) * beta0
            beta_max = beta
            for _ in range(self.beta_iters):
                beta_mid = (beta_min + beta_max) / 2.0
                current_error = self._error_bound(
                    beta_mid.unsqueeze(-1), density_fn, sdf, z_vals, dists, d_star
                )
                beta_max = torch.where(current_error <= self.eps, beta_mid, beta_max)
                beta_min = torch.where(current_error > self.eps, beta_mid, beta_min)
            beta = beta_max

            density = density_fn(sdf.reshape(z_vals.shape), beta=beta.unsqueeze(-1))
            terminal_dist = torch.full_like(dists[:, :1], 1e10)
            render_dists = torch.cat([dists, terminal_dist], dim=-1)
            free_energy = render_dists * density
            shifted_free_energy = torch.cat(
                [torch.zeros_like(render_dists[:, :1]), free_energy[:, :-1]], dim=-1
            )
            alpha = 1 - torch.exp(-free_energy)
            transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
            weights = alpha * transmittance

            not_converged = bool((beta.max() > beta0).item())
            refine = not_converged and total_iters < self.max_total_iters
            if refine:
                error_per_section = (
                    torch.exp(-d_star / beta.unsqueeze(-1))
                    * render_dists[:, :-1].square()
                    / (4 * beta.unsqueeze(-1).square())
                )
                error_integral = torch.cumsum(error_per_section, dim=-1)
                bound_opacity = (
                    torch.clamp(torch.exp(error_integral), max=1.0e6) - 1.0
                ) * transmittance[:, :-1]
                pdf = bound_opacity + self.add_tiny
                num_samples = self.N_samples_eval
            else:
                pdf = weights[..., :-1] + 1e-5
                num_samples = self.N_samples

            samples = self._invert_cdf(
                z_vals,
                pdf,
                num_samples,
                deterministic=refine or not model.training,
            )
            if refine:
                z_vals, samples_idx = torch.sort(torch.cat([z_vals, samples], dim=-1), dim=-1)
            else:
                break

        return samples, z_vals

    def _with_extra_samples(self, z_samples, accumulated_z, near, far, is_training):
        if self.N_samples_extra > 0:
            if is_training:
                sample_idx = torch.randperm(
                    accumulated_z.shape[1], device=accumulated_z.device
                )[: self.N_samples_extra]
            else:
                sample_idx = torch.linspace(
                    0,
                    accumulated_z.shape[1] - 1,
                    self.N_samples_extra,
                    device=accumulated_z.device,
                ).long()
            extras = torch.cat([near, far, accumulated_z[:, sample_idx]], dim=-1)
        else:
            extras = torch.cat([near, far], dim=-1)
        return torch.sort(torch.cat([z_samples, extras], dim=-1), dim=-1).values

    @staticmethod
    def _sample_eikonal(z_vals):
        indices = torch.randint(z_vals.shape[-1], (z_vals.shape[0],), device=z_vals.device)
        return torch.gather(z_vals, 1, indices.unsqueeze(-1))


class ErrorBoundSamplerWithSMPLXDeformerAndBBox(_AdaptiveErrorBoundSampler):
    def __init__(self, scene_bounding_sphere, **kwargs):
        super().__init__(scene_bounding_sphere, **kwargs)
        self.bbox_sampler = BBoxRaySampler(
            self.N_samples_eval, self.near, scene_bounding_sphere
        )

    @torch.no_grad()
    def get_z_vals(self, ray_dirs, cam_loc, model, cond, smplx_tfs, smplx_verts):
        bbox_min = smplx_verts.min(dim=1).values[0]
        bbox_max = smplx_verts.max(dim=1).values[0]
        bbox_size = bbox_max - bbox_min
        bbox_min = bbox_min - bbox_size * 0.2
        bbox_max = bbox_max + bbox_size * 0.2
        near, far, z_vals = self.bbox_sampler.get_z_vals(
            ray_dirs, cam_loc, bbox_min, bbox_max, model.training
        )
        z_samples, accumulated_z = self._adaptive_sample(
            z_vals,
            ray_dirs,
            cam_loc,
            model,
            model.fg_density.get_beta,
            model.fg_density,
            lambda points: model.sdf_func_fg(
                points, cond, smplx_tfs, smplx_verts=smplx_verts
            )[0],
        )
        return self._with_extra_samples(
            z_samples, accumulated_z, near[:, None], far[:, None], model.training
        )


class ErrorBoundSamplerWithoutDeformer(_AdaptiveErrorBoundSampler):
    def __init__(self, scene_bounding_sphere, **kwargs):
        super().__init__(scene_bounding_sphere, **kwargs)
        self.uniform_sampler = UniformSampler(
            scene_bounding_sphere,
            self.near,
            self.N_samples_eval,
            take_sphere_intersection=True,
        )

    @torch.no_grad()
    def get_z_vals(self, ray_dirs, cam_loc, model):
        initial_z = self.uniform_sampler.get_z_vals(ray_dirs, cam_loc, model)
        z_samples, accumulated_z = self._adaptive_sample(
            initial_z,
            ray_dirs,
            cam_loc,
            model,
            model.bg_density.get_beta,
            model.bg_density,
            lambda points: model.sdf_func_bg(points, None)[0],
        )
        near = ray_dirs.new_full((ray_dirs.shape[0], 1), self.near)
        far = render_utils.get_sphere_intersections(
            cam_loc, ray_dirs, r=self.scene_bounding_sphere
        )[:, 1:]
        z_vals = self._with_extra_samples(
            z_samples, accumulated_z, near, far, model.training
        )
        return z_vals, self._sample_eikonal(z_vals)


class ErrorBoundSamplerWithRigidDeformerAndBBox(_AdaptiveErrorBoundSampler):
    def __init__(self, scene_bounding_sphere, **kwargs):
        super().__init__(scene_bounding_sphere, **kwargs)
        self.uniform_sampler = UniformSampler(
            scene_bounding_sphere,
            self.near,
            self.N_samples_eval,
            take_sphere_intersection=True,
        )
        self.bbox_sampler = BBoxRaySampler(
            self.N_samples_eval, self.near, scene_bounding_sphere
        )

    @torch.no_grad()
    def get_z_vals(self, ray_dirs, cam_loc, model, obj_pose, obj_verts=None):
        if obj_verts is None:
            initial_z = self.uniform_sampler.get_z_vals(ray_dirs, cam_loc, model)
        else:
            deformed_verts = model.obj_deformer.forward(
                obj_verts.detach(), obj_pose.inverse()
            )
            bbox_min = deformed_verts.min(dim=0).values
            bbox_max = deformed_verts.max(dim=0).values
            bbox_size = bbox_max - bbox_min
            bbox_min = bbox_min - bbox_size * 0.5
            bbox_max = bbox_max + bbox_size * 0.5
            _, _, initial_z = self.bbox_sampler.get_z_vals(
                ray_dirs, cam_loc, bbox_min, bbox_max, model.training
            )

        z_samples, accumulated_z = self._adaptive_sample(
            initial_z,
            ray_dirs,
            cam_loc,
            model,
            model.obj_density.get_beta,
            model.obj_density,
            lambda points: model.sdf_func_obj(points, obj_pose)[0],
        )
        near = ray_dirs.new_full((ray_dirs.shape[0], 1), self.near)
        far = render_utils.get_sphere_intersections(
            cam_loc, ray_dirs, r=self.scene_bounding_sphere
        )[:, 1:]
        z_vals = self._with_extra_samples(
            z_samples, accumulated_z, near, far, model.training
        )
        return z_vals, self._sample_eikonal(z_vals)
