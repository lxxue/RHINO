from pathlib import Path

import torch
from smplx import SMPLX
from smplx.lbs import batch_rigid_transform, batch_rodrigues, blend_shapes, vertices2joints

from rhino.rhino_smplx.cano_smplx_params import (
    cano_smplx_body_pose,
    cano_smplx_global_orient,
    cano_smplx_left_hand_flat,
    cano_smplx_right_hand_flat,
    cano_smplx_transl,
)


class SMPLXServer(torch.nn.Module):
    """smplx.SMPLX wrapper that applies scale/translation and exposes the
    per-joint rigid transforms and skinning weights needed for LBS-based
    deformation (recomputed via smplx.lbs public helpers, since the upstream
    forward does not return them)."""

    def __init__(self, gender="neutral", betas=None):
        super().__init__()
        model_path = str(Path(__file__).parents[1] / "checkpoints" / "smplx")
        self.smplx = SMPLX(
            model_path=model_path,
            batch_size=1,
            gender=gender,
            use_pca=False,
            num_pca_comps=45,
            num_betas=300,
            flat_hand_mean=False,
            use_face_contour=True,
            create_leye_pose=False,
            create_reye_pose=False,
            create_jaw_pose=False,
            create_expression=False,
        )

        # canonical space with scale 1.0
        output = self.get_cano_smplx(betas=betas, update=False)
        self.register_buffer("verts_c", output["smplx_verts"])
        self.register_buffer("joints_c", output["smplx_jnts"])
        self.register_buffer("tfs_c_inv", output["smplx_tfs"].squeeze(0).inverse())

    def get_cano_smplx(self, betas, update=False):
        device = betas.device
        output = self.forward(
            scale=torch.tensor(1.0, device=device),
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
        if update:
            self.verts_c = output["smplx_verts"]
            self.joints_c = output["smplx_jnts"]
            self.tfs_c_inv = output["smplx_tfs"].squeeze(0).inverse()
        return output

    def _rigid_transforms_and_weights(self, smplx_output):
        """Per-joint rigid transforms [B, J, 4, 4] (relative to the shaped
        rest pose) and expanded skinning weights [B, V, J], recomputed with
        the same smplx.lbs primitives the upstream forward uses internally."""
        full_pose = smplx_output.full_pose
        batch_size = full_pose.shape[0]
        shape_components = torch.cat([smplx_output.betas, smplx_output.expression], dim=-1)
        shapedirs = torch.cat([self.smplx.shapedirs, self.smplx.expr_dirs], dim=-1)
        v_shaped = self.smplx.v_template + blend_shapes(shape_components, shapedirs)
        joints_shaped = vertices2joints(self.smplx.J_regressor, v_shaped)
        rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(batch_size, -1, 3, 3)
        _, rel_transforms = batch_rigid_transform(rot_mats, joints_shaped, self.smplx.parents)
        weights = self.smplx.lbs_weights.unsqueeze(dim=0).expand(batch_size, -1, -1)
        return rel_transforms, weights

    def forward(
        self,
        scale,
        betas,
        transl,
        global_orient,
        body_pose,
        left_hand_pose,
        right_hand_pose,
        expression=None,
        jaw_pose=None,
        leye_pose=None,
        reye_pose=None,
        absolute=False,
    ):

        output = {}

        smplx_output = self.smplx.forward(
            betas=betas,
            transl=torch.zeros_like(transl),
            body_pose=body_pose,
            global_orient=global_orient,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            expression=expression,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            return_verts=True,
            return_full_pose=True,
        )
        rel_transforms, weights = self._rigid_transforms_and_weights(smplx_output)

        verts = smplx_output.vertices.clone()
        output["smplx_verts"] = verts * scale + transl.unsqueeze(1) * scale
        joints = smplx_output.joints.clone()
        output["smplx_jnts"] = joints * scale + transl.unsqueeze(1) * scale
        # We apply the scale in the transformation matrices here
        tf_mats = rel_transforms.clone()[:, :, :3, :]
        tf_mats[:, :, :3, 3] = tf_mats[:, :, :3, 3] + transl.unsqueeze(1)
        tf_mats = tf_mats * scale

        last_row = torch.tensor([0, 0, 0, 1], dtype=tf_mats.dtype, device=tf_mats.device)
        last_row = last_row.view(1, 1, 1, 4).expand(tf_mats.shape[0], tf_mats.shape[1], -1, -1)
        tf_mats = torch.cat([tf_mats, last_row], dim=2)

        if not absolute:
            tf_mats = torch.einsum("bnij,njk->bnik", tf_mats, self.tfs_c_inv)

        output["smplx_tfs"] = tf_mats
        output["smplx_weights"] = weights
        return output
