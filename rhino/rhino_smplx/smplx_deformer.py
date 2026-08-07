import torch
from pytorch3d import ops

from rhino.components.deformer import skinning
from rhino.rhino_smplx.cano_smplx_params import (
    cano_smplx_body_pose,
    cano_smplx_global_orient,
    cano_smplx_left_hand_flat,
    cano_smplx_right_hand_flat,
    cano_smplx_transl,
)
from rhino.rhino_smplx.smplx_server import SMPLXServer


class SMPLXDeformer(torch.nn.Module):

    def __init__(self, K=5, gender=None, betas=None, scale=1.0, smplx_server=None):
        super().__init__()
        self.K = K
        # Reuse a caller-provided server (HOINetwork shares one instance)
        # instead of loading a second copy of the SMPL-X model.
        self.smplx = SMPLXServer(gender=gender, betas=betas) if smplx_server is None else smplx_server
        smplx_output = self.smplx.forward(
            scale=scale,
            betas=betas,
            transl=cano_smplx_transl[None],
            body_pose=cano_smplx_body_pose[None],
            global_orient=cano_smplx_global_orient[None],
            left_hand_pose=cano_smplx_left_hand_flat[None],
            right_hand_pose=cano_smplx_right_hand_flat[None],
            jaw_pose=torch.zeros((1, 3), dtype=torch.float32),
            expression=torch.zeros((1, 10), dtype=torch.float32),
            leye_pose=torch.zeros((1, 3), dtype=torch.float32),
            reye_pose=torch.zeros((1, 3), dtype=torch.float32),
            # here we set absolute to False, so that the output is wrt thetas=thetas_canonical
            absolute=False,
        )
        self.register_buffer("smplx_verts", smplx_output["smplx_verts"])
        self.register_buffer("smplx_weights", smplx_output["smplx_weights"])

    def forward(self, x, smplx_tfs, return_weights=True, inverse=False, smplx_verts=None):
        if smplx_verts is None:
            weights, outlier_mask = self.query_skinning_weights_smplx_multi(
                x[None], smplx_verts=self.smplx_verts[0], smplx_weights=self.smplx_weights
            )
        else:
            weights, outlier_mask = self.query_skinning_weights_smplx_multi(
                x[None], smplx_verts=smplx_verts[0], smplx_weights=self.smplx_weights
            )
        if return_weights:
            return weights
        x_transformed = skinning(x.unsqueeze(0), weights, smplx_tfs, inverse=inverse)

        return x_transformed.squeeze(0), outlier_mask

    def update_cano_smplx(self, scale, betas):
        device = betas.device
        output = self.smplx.forward(
            scale=scale,
            betas=betas,
            transl=cano_smplx_transl[None].to(device),
            body_pose=cano_smplx_body_pose[None].to(device),
            global_orient=cano_smplx_global_orient[None].to(device),
            left_hand_pose=cano_smplx_left_hand_flat[None].to(device),
            right_hand_pose=cano_smplx_right_hand_flat[None].to(device),
            jaw_pose=torch.zeros((1, 3), dtype=torch.float32, device=device),
            expression=torch.zeros((1, 10), dtype=torch.float32, device=device),
            # need to get tfs_c_inv for the canonical pose
            leye_pose=torch.zeros((1, 3), dtype=torch.float32, device=device),
            reye_pose=torch.zeros((1, 3), dtype=torch.float32, device=device),
            absolute=True,
        )
        self.smplx_verts = output["smplx_verts"]
        self.smplx_weights = output["smplx_weights"]
        return

    def forward_skinning(self, xc, cond, smplx_tfs):
        weights, outlier_mask = self.query_skinning_weights_smplx_multi(
            xc, smplx_verts=self.smplx_verts[0], smplx_weights=self.smplx_weights
        )
        x_transformed = skinning(xc, weights, smplx_tfs, inverse=False)

        return x_transformed

    def query_weights(self, xc, smplx_verts):
        weights = self.forward(
            xc, None, return_weights=True, inverse=False, smplx_verts=smplx_verts
        )
        return weights

    def query_skinning_weights_smplx_multi(self, pts, smplx_verts, smplx_weights):
        distance_batch, index_batch, neighbor_points = ops.knn_points(
            pts, smplx_verts.unsqueeze(0), K=self.K, return_nn=True
        )
        distance_batch = torch.clamp(distance_batch, max=4)
        weights_conf = torch.exp(-distance_batch)
        distance_batch = torch.sqrt(distance_batch + 1e-8)
        weights_conf = weights_conf / weights_conf.sum(-1, keepdim=True)
        index_batch = index_batch[0]
        weights = smplx_weights[:, index_batch, :]
        weights = torch.sum(weights * weights_conf.unsqueeze(-1), dim=-2).detach()

        outlier_mask = (distance_batch[..., 0] > 0.1)[0]
        return weights, outlier_mask
