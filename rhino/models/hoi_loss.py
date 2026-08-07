
import pytorch3d
import torch
import torch.nn as nn

from rhino.components.loss import S3IM, ScaleAndShiftInvariantLoss
from smplx.lbs import batch_rodrigues


class HOILoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.eikonal_weight = cfg.eikonal_weight
        self.density_weight = cfg.density_weight
        self.normal_l1_weight = cfg.normal_l1_weight
        self.normal_cos_weight = cfg.normal_cos_weight
        self.depth_weight = cfg.depth_weight
        self.off_surface_weight = cfg.off_surface_weight
        self.in_surface_weight = cfg.in_surface_weight
        self.penetration_weight = cfg.penetration_weight
        self.hand_contact_weight = cfg.hand_contact_weight
        self.interactvlm_contact_weight = cfg.interactvlm_contact_weight
        self.self_penetration_weight = cfg.self_penetration_weight
        self.fg_mask_weight = cfg.fg_mask_weight
        self.obj_mask_weight = cfg.obj_mask_weight
        self.bg_mask_weight = cfg.bg_mask_weight
        self.body_pose_smooth_weight = cfg.body_pose_smooth_weight
        self.obj_pose_smooth_weight = cfg.obj_pose_smooth_weight
        self.body_verts_smooth_weight = cfg.body_verts_smooth_weight
        self.obj_verts_smooth_weight = cfg.obj_verts_smooth_weight
        self.s3im_weight = cfg.s3im_weight
        self.hand_weight = cfg.hand_weight
        self.body_weight = cfg.body_weight
        self.l1_loss = nn.L1Loss(reduction="mean")
        self.l2_loss = nn.MSELoss(reduction="mean")
        self.depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)
        self.s3im_loss = S3IM(
            s3im_kernel_size=4, s3im_stride=4, s3im_repeat_time=10, s3im_patch_height=32
        )
        self.max_steps = cfg.max_steps
        self.penetration_loss_steps = cfg.max_steps // 5

    def get_rgb_loss(self, rgb_values, rgb_gt):
        rgb_loss = self.l1_loss(rgb_values, rgb_gt)
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        eikonal_loss = ((grad_theta.norm(2, dim=-1) - 1) ** 2).mean()
        return eikonal_loss

    def get_depth_loss(self, depth_pred, depth_gt, mask):
        return self.depth_loss(depth_pred[:, :, None], depth_gt[:, :, None], mask[:, :, None])

    def get_normal_loss(self, normal_pred, normal_gt, mask):
        if not mask.any():
            return torch.zeros((), device=mask.device), torch.zeros((), device=mask.device)
        l1 = torch.abs(normal_pred[mask] - normal_gt[mask]).sum(dim=-1).mean()
        cos = (1.0 - torch.sum(normal_pred[mask] * normal_gt[mask], dim=-1)).mean()
        return l1, cos

    def get_self_penetration_loss(self, smplx_verts_fg_sdf):
        penetration_mask = smplx_verts_fg_sdf < 0
        if penetration_mask.any():
            penetration_loss = self.ipman_penetration_loss(smplx_verts_fg_sdf[penetration_mask])
        else:
            penetration_loss = torch.zeros((), device=smplx_verts_fg_sdf.device)
        return penetration_loss

    def get_density_loss(self, acc_map):
        binary_loss = (
            -1
            * (acc_map * (acc_map + 1e-4).log() + (1 - acc_map) * (1 - acc_map + 1e-4).log()).mean()
        )
        density_loss = 2 * binary_loss
        return density_loss

    def get_off_surface_loss(self, fg_acc_map, index_off_surface):
        off_surface_loss = self.l1_loss(
            fg_acc_map[index_off_surface],
            torch.zeros_like(fg_acc_map[index_off_surface]),
        )  # torch.sum(acc_map[index_off_surface])
        return off_surface_loss

    def get_in_surface_loss(self, fg_acc_map, index_in_surface):
        if index_in_surface.any():
            in_surface_loss = self.l1_loss(
                fg_acc_map[index_in_surface], torch.ones_like(fg_acc_map[index_in_surface])
            )
        else:
            in_surface_loss = torch.zeros((), device=fg_acc_map.device)
        return in_surface_loss

    def get_mask_loss_single(self, acc_map, mask):
        return self.l1_loss(acc_map, mask)

    def get_penetration_loss_new(
        self,
        fg_sdf,
        obj_sdf,
        bg_sdf,
        fg_pene_mask,
        obj_pene_mask,
        bg_pene_mask,
        obj_samples_fg_sdf,
        bg_samples_fg_sdf,
        fg_samples_obj_sdf,
        bg_samples_obj_sdf,
        fg_samples_bg_sdf,
        obj_samples_bg_sdf,
    ):
        threshold = -0.001
        loss = torch.zeros((), device=fg_sdf.device)
        fg_sdf_negative = fg_sdf.view(-1)[fg_pene_mask]
        obj_sdf_negative = obj_sdf.view(-1)[obj_pene_mask]
        bg_sdf_negative = bg_sdf.view(-1)[bg_pene_mask]

        fg_obj_pene_mask = fg_samples_obj_sdf < threshold
        if (fg_obj_pene_mask).any():
            loss += (
                fg_sdf_negative[fg_obj_pene_mask[:, 0]] * fg_samples_obj_sdf[fg_obj_pene_mask]
            ).sum()
        fg_bg_pene_mask = fg_samples_bg_sdf < threshold
        if (fg_bg_pene_mask).any():
            loss += (
                fg_sdf_negative[fg_bg_pene_mask[:, 0]] * fg_samples_bg_sdf[fg_bg_pene_mask]
            ).sum()

        obj_fg_pene_mask = obj_samples_fg_sdf < threshold
        if (obj_fg_pene_mask).any():
            loss += (
                obj_sdf_negative[obj_fg_pene_mask[:, 0]] * obj_samples_fg_sdf[obj_fg_pene_mask]
            ).sum()
        obj_bg_pene_mask = obj_samples_bg_sdf < threshold
        if (obj_bg_pene_mask).any():
            loss += (
                obj_sdf_negative[obj_bg_pene_mask[:, 0]] * obj_samples_bg_sdf[obj_bg_pene_mask]
            ).sum()

        bg_fg_pene_mask = bg_samples_fg_sdf < threshold
        if (bg_fg_pene_mask).any():
            loss += (
                bg_sdf_negative[bg_fg_pene_mask[:, 0]] * bg_samples_fg_sdf[bg_fg_pene_mask]
            ).sum()
        bg_obj_pene_mask = bg_samples_obj_sdf < threshold
        if (bg_obj_pene_mask).any():
            loss += (
                bg_sdf_negative[bg_obj_pene_mask[:, 0]] * bg_samples_obj_sdf[bg_obj_pene_mask]
            ).sum()

        return loss

    def get_hand_loss(self, hand_sdf_pred, hand_sdf_gt, hand_weights):
        # hand_weights is [N, 1] while the SDFs are [N]; flatten so the wrist
        # down-weighting applies per vertex instead of broadcasting to an
        # [N, N] matrix, which collapsed it to a constant scale on the mean.
        hand_weights = hand_weights.reshape(-1)
        hand_loss = self.l1_loss(hand_weights * hand_sdf_pred, hand_weights * hand_sdf_gt)
        return hand_loss

    def get_hand_contact_loss(self, hand_obj_sdf):
        # we do not need the full hand to be in contact, but at least some part of it
        # find the percentile of the hand obj sdf such that 5% of the absolute values are below it
        percentile_value = torch.quantile(torch.abs(hand_obj_sdf), 0.05)
        mask = torch.abs(hand_obj_sdf) < percentile_value
        contact_sdf = hand_obj_sdf[mask]
        contact_loss = self.ipman_contact_loss(contact_sdf)
        # the rest points should have positive sdf values
        inv_mask = torch.logical_not(mask)
        penetration_sdf = hand_obj_sdf[inv_mask]
        penetration_loss = self.ipman_penetration_loss(penetration_sdf)
        return contact_loss + penetration_loss

    def get_interactvlm_contact_loss(self, contact_verts_obj_sdf):
        mask = contact_verts_obj_sdf > 0
        contact_sdf = contact_verts_obj_sdf[mask]
        contact_loss = self.ipman_contact_loss(contact_sdf)
        # the rest points should have positive sdf values
        # allow for 2 cm penetration
        penetration_mask = contact_verts_obj_sdf < -0.002
        penetration_sdf = contact_verts_obj_sdf[penetration_mask]
        penetration_loss = self.ipman_penetration_loss(penetration_sdf)
        return contact_loss + penetration_loss

    def get_body_loss(self, body_sdf):
        mask = body_sdf > 0
        if mask.any():
            # similar to ipman penetration loss
            body_loss = (torch.tanh(body_sdf[mask] / 0.15) ** 2).mean()
        else:
            body_loss = torch.zeros((), device=body_sdf.device)
        return body_loss

    def get_body_pose_smooth_loss(self, body_pose_1, body_pose_2):
        # concat global orient and body pose; compare in matrix form as the
        # axis-angle space is not continuous
        pose_1 = torch.cat([body_pose_1["global_orient"], body_pose_1["body_pose"]], dim=-1)
        pose_2 = torch.cat([body_pose_2["global_orient"], body_pose_2["body_pose"]], dim=-1)
        pose_1_mat = batch_rodrigues(pose_1.reshape(-1, 3))
        pose_2_mat = batch_rodrigues(pose_2.reshape(-1, 3))
        rel_pose = pose_2_mat @ torch.linalg.inv(pose_1_mat)
        rel_pose_log = pytorch3d.transforms.so3_log_map(rel_pose)
        loss = torch.square(rel_pose_log).sum(dim=1).sum()
        loss += self.l2_loss(body_pose_1["transl"], body_pose_2["transl"])
        return loss

    # source: https://github.com/sha2nkt/ipman-r/blob/main/stability/ground_losses.py
    def ipman_contact_loss(self, sdf):
        a1 = 1
        a2 = 0.15
        mask = sdf > 0
        if mask.any():
            loss = a1 * torch.tanh(sdf[mask] / a2) ** 2
        else:
            loss = torch.zeros((), device=sdf.device)
        return loss.mean()

    def ipman_penetration_loss(self, sdf):
        b1 = 1
        b2 = 0.15
        mask = sdf < 0
        if mask.any():
            loss = b1 * torch.tanh(sdf[mask] / b2) ** 2
        else:
            loss = torch.zeros((), device=sdf.device)
        return loss.mean()

    def get_obj_pose_smooth_loss(self, obj_pose_1, obj_pose_2):
        # l2 directly on the 4x4 pose matrices (log-map relative-pose variants
        # were numerically unstable)
        loss = self.l2_loss(obj_pose_1, obj_pose_2)
        return loss

    def get_body_verts_smooth_loss(self, verts_cur, verts_neighbor):
        return self.l2_loss(verts_cur, verts_neighbor)

    def get_obj_verts_smooth_loss(self, obj_verts_cano, obj_pose_cur, obj_pose_neighbor):
        def to_world(verts_c, pose):
            R = pose[0, :3, :3]
            t = pose[0, :3, 3]
            return (verts_c - t) @ R  # R^T @ (x_c - t)

        verts_w_cur = to_world(obj_verts_cano, obj_pose_cur)
        verts_w_neighbor = to_world(obj_verts_cano, obj_pose_neighbor)
        return self.l2_loss(verts_w_cur, verts_w_neighbor)

    def forward(self, model_outputs, ground_truth):
        # resume-safe schedule driver (the old per-forward counter reset to 0
        # on checkpoint resume, silently re-running early-training schedules)
        step = model_outputs["global_step"]
        rgb_loss = self.get_rgb_loss(model_outputs["rgb"], ground_truth["rgb"])
        fg_eikonal_loss = self.eikonal_weight * self.get_eikonal_loss(
            model_outputs["fg_grad_theta"]
        )
        obj_eikonal_loss = 10 * self.eikonal_weight * self.get_eikonal_loss(
            model_outputs["obj_grad_theta"]
        )
        bg_eikonal_loss = self.eikonal_weight * self.get_eikonal_loss(
            model_outputs["bg_grad_theta"]
        )
        if self.density_weight > 0:
            density_loss = self.density_weight * (
                self.get_density_loss(model_outputs["fg_acc_map"])
                + self.get_density_loss(model_outputs["obj_acc_map"])
                + self.get_density_loss(model_outputs["bg_acc_map"])
            )
        else:
            density_loss = torch.zeros((), device=rgb_loss.device)
        off_surface_weight = self.off_surface_weight * (1 + 10 * step / self.max_steps)
        if off_surface_weight > 0:
            off_surface_loss = off_surface_weight * self.get_off_surface_loss(
                model_outputs["fg_acc_map"], model_outputs["index_off_surface"]
            )
        else:
            off_surface_loss = torch.zeros((), device=rgb_loss.device)
        in_surface_weight = self.in_surface_weight * (1 - step / self.max_steps)
        if in_surface_weight > 0:
            in_surface_loss = in_surface_weight * self.get_in_surface_loss(
                model_outputs["fg_acc_map"], model_outputs["index_in_surface"]
            )
        else:
            in_surface_loss = torch.zeros((), device=rgb_loss.device)

        # b x n x m
        bg_sdf = model_outputs["bg_sdf"]
        bg_sdf_mask = torch.logical_and((bg_sdf < 0.0).any(dim=2), (bg_sdf > 0.0).any(dim=2))
        obj_sdf = model_outputs["obj_sdf"]
        obj_sdf_mask = torch.logical_and((obj_sdf < 0.0).any(dim=2), (obj_sdf > 0.0).any(dim=2))
        fg_sdf = model_outputs["fg_sdf"]
        fg_sdf_mask = torch.logical_and((fg_sdf < 0.0).any(dim=2), (fg_sdf > 0.0).any(dim=2))
        sdf_mask = torch.logical_or(bg_sdf_mask, obj_sdf_mask)
        sdf_mask = torch.logical_or(fg_sdf_mask, sdf_mask)
        human_mask = ground_truth["human_mask"]
        non_human_mask = torch.logical_not(human_mask)
        if step < self.max_steps * 0.3:
            mask = sdf_mask  # include human pixels for alignment
        else:
            mask = torch.logical_and(sdf_mask, non_human_mask)
        depth_weight = self.depth_weight
        if depth_weight > 0:
            depth_loss = depth_weight * self.get_depth_loss(
                model_outputs["depth"], ground_truth["depth"], mask
            )
        else:
            depth_loss = torch.zeros((), device=rgb_loss.device)
        normal_l1_weight = self.normal_l1_weight
        normal_cos_weight = self.normal_cos_weight
        if self.normal_l1_weight > 0 or self.normal_cos_weight > 0:
            normal_l1_loss, normal_cos_loss = self.get_normal_loss(
                model_outputs["normal"], ground_truth["normal"], mask
            )
            normal_l1_loss = normal_l1_weight * normal_l1_loss
            normal_cos_loss = normal_cos_weight * normal_cos_loss
        else:
            normal_l1_loss = torch.zeros((), device=rgb_loss.device)
            normal_cos_loss = torch.zeros((), device=rgb_loss.device)
        if step > self.penetration_loss_steps and self.penetration_weight > 0:
            penetration_loss = self.penetration_weight * self.get_penetration_loss_new(
                fg_sdf,
                obj_sdf,
                bg_sdf,
                model_outputs["fg_pene_mask"],
                model_outputs["obj_pene_mask"],
                model_outputs["bg_pene_mask"],
                model_outputs["obj_samples_fg_sdf"],
                model_outputs["bg_samples_fg_sdf"],
                model_outputs["fg_samples_obj_sdf"],
                model_outputs["bg_samples_obj_sdf"],
                model_outputs["fg_samples_bg_sdf"],
                model_outputs["obj_samples_bg_sdf"],
            )
        else:
            penetration_loss = torch.zeros((), device=rgb_loss.device)

        if self.self_penetration_weight > 0 and model_outputs["smplx_verts_fg_sdf"] is not None:
            self_penetration_loss = self.get_self_penetration_loss(
                model_outputs["smplx_verts_fg_sdf"]
            )
            self_penetration_loss *= self.self_penetration_weight
        else:
            self_penetration_loss = torch.zeros((), device=rgb_loss.device)

        if self.fg_mask_weight > 0:
            fg_mask_weight = self.fg_mask_weight * (1 - step / self.max_steps)
            fg_mask_loss = fg_mask_weight * self.get_mask_loss_single(
                model_outputs["acc_map_fg"], ground_truth["human_mask"]
            )
        else:
            fg_mask_loss = torch.zeros((), device=rgb_loss.device)

        if self.obj_mask_weight > 0:
            obj_mask_weight = self.obj_mask_weight * (1 - step / self.max_steps)
            obj_mask_loss = obj_mask_weight * self.get_mask_loss_single(
                model_outputs["acc_map_obj"], ground_truth["obj_mask"]
            )
        else:
            obj_mask_loss = torch.zeros((), device=rgb_loss.device)

        if self.bg_mask_weight > 0:
            all_mask = ground_truth["all_mask"]
            bg_mask = 1 - all_mask
            bg_mask_weight = self.bg_mask_weight * (1 - step / self.max_steps)
            bg_mask_loss = bg_mask_weight * self.get_mask_loss_single(
                model_outputs["acc_map_bg"], bg_mask
            )
        else:
            bg_mask_loss = torch.zeros((), device=rgb_loss.device)

        if self.s3im_weight > 0:
            s3im_loss = self.s3im_weight * self.s3im_loss(model_outputs["rgb"], ground_truth["rgb"])
        else:
            s3im_loss = torch.zeros((), device=rgb_loss.device)

        if self.hand_weight > 0:
            hand_loss = self.hand_weight * self.get_hand_loss(
                model_outputs["hand_sdf_pred"],
                model_outputs["hand_sdf_gt"],
                model_outputs["hand_weights"],
            )
        else:
            hand_loss = torch.zeros((), device=rgb_loss.device)

        if self.body_weight > 0 and model_outputs["body_sdf"] is not None:
            body_loss = self.body_weight * self.get_body_loss(model_outputs["body_sdf"])
        else:
            body_loss = torch.zeros((), device=rgb_loss.device)

        hand_contact_loss = torch.zeros((), device=rgb_loss.device)
        if self.hand_contact_weight > 0 and model_outputs["left_hand_obj_sdf"] is not None:
            hand_contact_loss = self.get_hand_contact_loss(model_outputs["left_hand_obj_sdf"])
            hand_contact_loss += self.get_hand_contact_loss(model_outputs["right_hand_obj_sdf"])
            hand_contact_loss *= self.hand_contact_weight

        interactvlm_contact_loss = torch.zeros((), device=rgb_loss.device)
        if (
            self.interactvlm_contact_weight > 0
            and model_outputs["contact_verts_obj_sdf"] is not None
        ):
            interactvlm_contact_loss = self.get_interactvlm_contact_loss(
                model_outputs["contact_verts_obj_sdf"]
            )
            interactvlm_contact_loss *= self.interactvlm_contact_weight

        body_pose_smooth_loss = torch.zeros((), device=rgb_loss.device)
        if self.body_pose_smooth_weight > 0:
            if model_outputs["body_model_params_prev"] is not None:
                body_pose_smooth_loss += self.get_body_pose_smooth_loss(
                    model_outputs["body_model_params_prev"], model_outputs["body_model_params_cur"]
                )
            if model_outputs["body_model_params_next"] is not None:
                body_pose_smooth_loss += self.get_body_pose_smooth_loss(
                    model_outputs["body_model_params_cur"], model_outputs["body_model_params_next"]
                )
            body_pose_smooth_loss *= self.body_pose_smooth_weight

        obj_pose_smooth_loss = torch.zeros((), device=rgb_loss.device)
        if self.obj_pose_smooth_weight > 0:
            if model_outputs["obj_pose_prev"] is not None:
                obj_pose_smooth_loss += self.get_obj_pose_smooth_loss(
                    model_outputs["obj_pose_prev"], model_outputs["obj_pose_cur"]
                )
            if model_outputs["obj_pose_next"] is not None:
                obj_pose_smooth_loss += self.get_obj_pose_smooth_loss(
                    model_outputs["obj_pose_cur"], model_outputs["obj_pose_next"]
                )
            obj_pose_smooth_loss *= self.obj_pose_smooth_weight

        body_verts_smooth_loss = torch.zeros((), device=rgb_loss.device)
        if self.body_verts_smooth_weight > 0 and model_outputs["smplx_verts"] is not None:
            if model_outputs["smplx_verts_prev"] is not None:
                body_verts_smooth_loss += self.get_body_verts_smooth_loss(
                    model_outputs["smplx_verts"], model_outputs["smplx_verts_prev"]
                )
            if model_outputs["smplx_verts_next"] is not None:
                body_verts_smooth_loss += self.get_body_verts_smooth_loss(
                    model_outputs["smplx_verts"], model_outputs["smplx_verts_next"]
                )
            body_verts_smooth_loss *= self.body_verts_smooth_weight

        obj_verts_smooth_loss = torch.zeros((), device=rgb_loss.device)
        if self.obj_verts_smooth_weight > 0 and model_outputs["obj_verts_cano"] is not None:
            if model_outputs["obj_pose_prev"] is not None:
                obj_verts_smooth_loss += self.get_obj_verts_smooth_loss(
                    model_outputs["obj_verts_cano"],
                    model_outputs["obj_pose_cur"],
                    model_outputs["obj_pose_prev"],
                )
            if model_outputs["obj_pose_next"] is not None:
                obj_verts_smooth_loss += self.get_obj_verts_smooth_loss(
                    model_outputs["obj_verts_cano"],
                    model_outputs["obj_pose_cur"],
                    model_outputs["obj_pose_next"],
                )
            obj_verts_smooth_loss *= self.obj_verts_smooth_weight

        loss = (
            rgb_loss
            + hand_contact_loss
            + interactvlm_contact_loss
            + self_penetration_loss
            + hand_loss
            + body_loss
            + body_pose_smooth_loss
            + obj_pose_smooth_loss
            + body_verts_smooth_loss
            + obj_verts_smooth_loss
            + fg_eikonal_loss
            + obj_eikonal_loss
            + bg_eikonal_loss
            + fg_mask_loss
            + obj_mask_loss
            + bg_mask_loss
            + density_loss
            + off_surface_loss
            + in_surface_loss
            + depth_loss
            + normal_l1_loss
            + normal_cos_loss
            + penetration_loss
            + s3im_loss
        )
        return {
            "train/loss": loss,
            "train/rgb": rgb_loss,
            "train/hand_contact": hand_contact_loss,
            "train/interactvlm_contact": interactvlm_contact_loss,
            "train/self_penetration": self_penetration_loss,
            "train/body_pose_smooth": body_pose_smooth_loss,
            "train/obj_pose_smooth": obj_pose_smooth_loss,
            "train/body_verts_smooth": body_verts_smooth_loss,
            "train/obj_verts_smooth": obj_verts_smooth_loss,
            "train/body": body_loss,
            "train/hand": hand_loss,
            "train/obj_mask": obj_mask_loss,
            "train/fg_mask": fg_mask_loss,
            "train/bg_mask": bg_mask_loss,
            "train/s3im": s3im_loss,
            "train/depth": depth_loss,
            "train/normal_cos": normal_cos_loss,
            "train/penetration": penetration_loss,
            "train/fg_eikonal": fg_eikonal_loss,
            "train/obj_eikonal": obj_eikonal_loss,
            "train/bg_eikonal": bg_eikonal_loss,
            "train/density_reg": density_loss,
            "train/off_surface": off_surface_loss,
            "train/in_surface": in_surface_loss,
            "train/normal_l1": normal_l1_loss,
        }
