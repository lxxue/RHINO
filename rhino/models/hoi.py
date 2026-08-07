import math
import pickle as pkl
from pathlib import Path

import cv2
import numpy as np
import pytorch3d
import pytorch_lightning as pl
import torch
import trimesh
from einops import rearrange
from kaolin.ops.mesh import index_vertices_by_faces
import matplotlib
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from rhino.components.body_model_params import BodyModelParams
from rhino.components.camera_pose_params import CameraPoseParams
from rhino.components.deformer import skinning
from rhino.components.loss import compute_scale_and_shift
from rhino.components.object_pose_params import ObjectPoseParams
from rhino.datasets.frame_utils import build_frame_indices
from rhino.models.training_schedule import get_training_stage
from rhino.utils import idr_utils, mesh_utils
from rhino.utils.sdf_vis import plot_slice, query_sdf_batched, make_cubic_grid


from rhino.models.hoi_network import HOINetwork
from rhino.models.hoi_loss import HOILoss

class HOI(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()

        self.training_indices = build_frame_indices(
            cfg.dataset.train.start_frame,
            cfg.dataset.train.end_frame,
            cfg.dataset.train.skip_step,
            cfg.dataset.train.exclude_frames,
        )
        num_training_frames = len(self.training_indices)

        data_root = cfg.dataset.train.data_dir
        _smplx_path = data_root / "aligned_smplx.pkl"
        with open(_smplx_path, "rb") as f:
            smplx_params = pkl.load(f)
        betas = torch.tensor(smplx_params["betas"][None], dtype=torch.float32)
        scale = torch.tensor(smplx_params["scale"][0], dtype=torch.float32).view((1, 1))

        self.num_epochs = math.ceil(cfg.model.max_steps / num_training_frames)
        self.opt_pose_freq = cfg.model.opt_pose_freq
        self.opt_pose_len = cfg.model.opt_pose_len

        self.model = HOINetwork(cfg.model, num_training_frames, "neutral", scale, betas)

        self.cfg = cfg

        self.opt_smplx = cfg.model.opt_smplx
        self.opt_camera_pose = cfg.model.opt_camera_pose
        self.opt_object_pose = cfg.model.opt_object_pose
        self.enable_contact_loss = False
        self._active_optimizer_mode = "no_pose"

        self.val_render_image = cfg.model.val_render_image
        self.test_render_image = cfg.model.test_render_image
        self.test_save_numerical_results = cfg.model.test_save_numerical_results
        self.pixel_per_batch = cfg.dataset.val.pixel_per_batch
        self.update_canonical_mesh = cfg.model.update_canonical_mesh

        if self.opt_smplx:
            self.body_model_params = BodyModelParams(num_training_frames, model_type="smplx")
            self.load_body_model_params()
            optim_params = self.body_model_params.param_names
            for param_name in optim_params:
                self.body_model_params.set_requires_grad(param_name, requires_grad=True)

        if self.opt_camera_pose:
            self.camera_pose_params = CameraPoseParams(num_training_frames)
            self.load_camera_pose_params()

        if self.opt_object_pose:
            self.object_pose_params = ObjectPoseParams(num_training_frames)
            self.load_object_pose_params()

        self.loss = HOILoss(cfg.model.loss)

        Path("rgb").mkdir(parents=False, exist_ok=True)
        Path("normal").mkdir(parents=False, exist_ok=True)
        Path("depth").mkdir(parents=False, exist_ok=True)
        Path("meshes").mkdir(parents=False, exist_ok=True)
        Path("sdf_vis").mkdir(parents=False, exist_ok=True)

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.lpips = LearnedPerceptualImagePatchSimilarity()

        # colormaps
        self.sequential_colormap = matplotlib.colormaps["inferno"]
        self.diverging_colormap = matplotlib.colormaps["seismic"]
        self.depth_colormap = matplotlib.colormaps["viridis"]

    def load_camera_pose_params(self):
        data_root = self.cfg.dataset.train.data_dir

        c2ws = np.load(data_root / "c2ws.npy")
        c2ws = c2ws[self.training_indices]
        c2ws = torch.from_numpy(c2ws).float()

        # https://github.com/facebookresearch/pytorch3d/issues/1488
        c2ws = c2ws.transpose(-2, -1)
        log_reps = pytorch3d.transforms.se3_log_map(c2ws)
        self.camera_pose_params.se3_log_reps.weight.data = log_reps
        self.camera_pose_params.se3_log_reps.weight.requires_grad = True

    def load_object_pose_params(self):
        data_root = self.cfg.dataset.train.data_dir
        obj_pose_file = getattr(self.cfg.dataset.train, "obj_pose_file", "object_poses_centered.npy")
        obj_poses = np.load(data_root / obj_pose_file)
        obj_poses = obj_poses[self.training_indices]
        obj_poses = torch.from_numpy(obj_poses).float()

        # https://github.com/facebookresearch/pytorch3d/issues/1488
        obj_poses = obj_poses.transpose(-2, -1)
        log_reps = pytorch3d.transforms.se3_log_map(obj_poses)
        self.object_pose_params.se3_log_reps.weight.data = log_reps
        self.object_pose_params.se3_log_reps.weight.requires_grad = True

    def load_body_model_params(self):
        body_model_params = {param_name: [] for param_name in self.body_model_params.param_names}
        data_root = self.cfg.dataset.train.data_dir

        _smplx_path = data_root / "aligned_smplx.pkl"
        with open(_smplx_path, "rb") as f:
            smplx_params = pkl.load(f)
        body_model_params["scale"] = torch.tensor(
            smplx_params["scale"][0], dtype=torch.float32
        ).view((1, 1))
        body_model_params["betas"] = torch.tensor(smplx_params["betas"][None], dtype=torch.float32)
        global_orient = np.array(smplx_params["global_orient"])
        body_model_params["global_orient"] = torch.tensor(
            global_orient[self.training_indices], dtype=torch.float32
        )
        body_pose = np.array(smplx_params["body_pose"])
        body_model_params["body_pose"] = torch.tensor(
            body_pose[self.training_indices], dtype=torch.float32
        )
        left_hand_pose = np.array(smplx_params["left_hand_pose"])
        body_model_params["left_hand_pose"] = torch.tensor(
            left_hand_pose[self.training_indices], dtype=torch.float32
        )
        right_hand_pose = np.array(smplx_params["right_hand_pose"])
        body_model_params["right_hand_pose"] = torch.tensor(
            right_hand_pose[self.training_indices], dtype=torch.float32
        )
        transl = np.array(smplx_params["transl"])
        body_model_params["transl"] = torch.tensor(
            transl[self.training_indices], dtype=torch.float32
        )
        expression = np.array(smplx_params["expression"])
        body_model_params["expression"] = torch.tensor(
            expression[self.training_indices], dtype=torch.float32
        )
        jaw_pose = np.array(smplx_params["jaw_pose"])
        body_model_params["jaw_pose"] = torch.tensor(
            jaw_pose[self.training_indices], dtype=torch.float32
        )
        body_model_params["leye_pose"] = torch.zeros_like(body_model_params["jaw_pose"])
        body_model_params["reye_pose"] = torch.zeros_like(body_model_params["jaw_pose"])
        for param_name in body_model_params.keys():
            self.body_model_params.init_parameters(
                param_name, body_model_params[param_name], requires_grad=False
            )

    def get_deformed_mesh(self, verts, smplx_tfs, smplx_verts):
        verts = torch.tensor(verts, device=self.device, dtype=torch.float32)
        weights = self.model.fg_deformer.query_weights(verts, smplx_verts)
        verts_deformed = skinning(verts.unsqueeze(0), weights, smplx_tfs).data.cpu().numpy()[0]
        return verts_deformed

    def on_load_checkpoint(self, checkpoint):
        # fg_deformer shares the network's SMPLXServer, so both key prefixes
        # alias the same tensors. Checkpoints from before the sharing carry a
        # second, stale copy under model.fg_deformer.smplx.* (its canonical
        # buffers were never refreshed and never read); overwrite those
        # entries with the live model.smplx_server.* values so the load
        # result does not depend on checkpoint key order.
        state_dict = checkpoint["state_dict"]
        old_prefix, live_prefix = "model.fg_deformer.smplx.", "model.smplx_server."
        for key in list(state_dict):
            if key.startswith(old_prefix):
                live_key = live_prefix + key.removeprefix(old_prefix)
                if live_key in state_dict:
                    state_dict[key] = state_dict[live_key]

        self.model.mesh_v_cano = checkpoint["state_dict"]["model.mesh_v_cano"]
        self.model.mesh_f_cano = checkpoint["state_dict"]["model.mesh_f_cano"]
        self.model.mesh_face_vertices = checkpoint["state_dict"]["model.mesh_face_vertices"]
        self.model.obj_verts_cano = checkpoint["state_dict"]["model.obj_verts_cano"]
        self._active_optimizer_mode = checkpoint.get("rhino_optimizer_mode")
        return super().on_load_checkpoint(checkpoint)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["rhino_optimizer_mode"] = self._active_optimizer_mode
        return super().on_save_checkpoint(checkpoint)

    def get_optimizer_params(self, mode):
        if mode == "normal":
            all_lr = self.cfg.model.learning_rate
            pose_lr = self.cfg.model.learning_rate * 0.1
        elif mode == "no_pose":
            all_lr = self.cfg.model.learning_rate
            pose_lr = 0
        elif mode == "pose" or mode == "pose_no_hand":
            all_lr = 0
            pose_lr = self.cfg.model.learning_rate * 0.1
        else:
            raise ValueError(f"Invalid optimizer mode: {mode}")

        params = []
        for name, module in self.model.named_children():
            if name == "fg_implicit_network":
                # reduce learning rate for the human density network to avoid large shape changes
                params.append({"params": module.parameters(), "lr": 0.1 * all_lr})
            elif name == "obj_implicit_network":
                params.append({"params": module.parameters(), "lr": 0.3 * all_lr})
            elif (
                name == "body_model_params"
                or name == "camera_pose_params"
                or name == "object_pose_params"
            ):
                assert False
            elif name == "smplx_server":
                # print("Skip SMPLX server parameters")
                pass
            elif name == "fg_deformer" or name == "obj_deformer":
                # print("Skip deformer parameters")
                pass
            else:
                # print("optimizing", name)
                params.append({"params": module.parameters(), "lr": all_lr})
        if self.opt_smplx:
            # if mode == "normal":
            #     self.body_model_params.set_requires_grad("left_hand_pose", requires_grad=True)
            #     self.body_model_params.set_requires_grad("right_hand_pose", requires_grad=True)
            # elif mode == "pose":
            #     # disable hand pose optimization
            #     self.body_model_params.set_requires_grad("left_hand_pose", requires_grad=False)
            #     self.body_model_params.set_requires_grad("right_hand_pose", requires_grad=False)
            # else:
            #     raise ValueError(f"Invalid optimizer mode: {mode}")
            # disable hand pose optimization in all stages
            if mode == "pose_no_hand" or mode == "no_pose":
                #     body_params = self.body_model_params.parameters()
                # elif mode == "pose":
                body_params = []
                for param_name, param in self.body_model_params.named_parameters():
                    if param_name == "left_hand_pose" or param_name == "right_hand_pose":
                        continue
                    else:
                        body_params.append(param)
            elif mode == "pose" or mode == "normal":
                body_params = self.body_model_params.parameters()
            else:
                raise ValueError(f"Invalid optimizer mode: {mode}")

            params.append(
                {
                    "params": body_params,
                    "lr": pose_lr,
                }
            )
        if self.opt_camera_pose:
            params.append(
                {
                    "params": self.camera_pose_params.parameters(),
                    "lr": pose_lr,
                }
            )

        if self.opt_object_pose:
            params.append(
                {
                    "params": self.object_pose_params.parameters(),
                    "lr": pose_lr,
                }
            )
        return params

    def configure_optimizers(self):
        # params = self.get_optimizer_params(mode="normal")
        # start with no pose optimization
        params = self.get_optimizer_params(mode="no_pose")
        # params = self.get_optimizer_params(mode="pose")
        optimizer = torch.optim.Adam(params)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, self.cfg.model.decay_rate ** (1 / self.cfg.model.decay_steps)
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _set_optimizer_mode(self, mode: str) -> None:
        if self._active_optimizer_mode == mode:
            return
        new_params = self.get_optimizer_params(mode=mode)
        optimizer = self.trainer.optimizers[0]
        optimizer.param_groups.clear()
        for param_group in new_params:
            optimizer.add_param_group(param_group)
        self._active_optimizer_mode = mode
        print(f"Switched to {mode} optimization.")

    def training_step(self, batch):
        inputs, targets = batch
        batch_idx = inputs["idx"]

        # disable contact loss when optimizing everything
        inputs["enable_contact_loss"] = self.enable_contact_loss
        if not self.enable_contact_loss:
            inputs["contact_label"] = False

        if self.opt_smplx:
            body_model_params = self.body_model_params(batch_idx)
            inputs["smplx_scale"] = body_model_params["scale"]
            inputs["smplx_betas"] = body_model_params["betas"]
            inputs["smplx_transl"] = body_model_params["transl"]
            inputs["smplx_body_pose"] = body_model_params["body_pose"]
            inputs["smplx_global_orient"] = body_model_params["global_orient"]
            inputs["smplx_left_hand_pose"] = body_model_params["left_hand_pose"]
            inputs["smplx_right_hand_pose"] = body_model_params["right_hand_pose"]
            inputs["smplx_expression"] = body_model_params["expression"]  # None
            inputs["smplx_leye_pose"] = body_model_params["leye_pose"]
            inputs["smplx_reye_pose"] = body_model_params["reye_pose"]
            inputs["smplx_jaw_pose"] = body_model_params["jaw_pose"]
            # self.log("smplx_scale", body_model_params["scale"], prog_bar=True, on_step=True)

            # get smplx_params of previous frame
            inputs["body_model_params_cur"] = body_model_params
            if batch_idx > 0:
                body_model_params_prev = self.body_model_params(batch_idx - 1)
                inputs["body_model_params_prev"] = body_model_params_prev
            else:
                inputs["body_model_params_prev"] = None

            if batch_idx < len(self.training_indices) - 1:
                body_model_params_next = self.body_model_params(batch_idx + 1)
                inputs["body_model_params_next"] = body_model_params_next
            else:
                inputs["body_model_params_next"] = None

            # Compute neighbor SMPLX vertices for vertex-based smooth loss
            for neighbor_key in ["body_model_params_prev", "body_model_params_next"]:
                verts_key = neighbor_key.replace("body_model_params", "smplx_verts")
                if inputs[neighbor_key] is not None:
                    with torch.no_grad():
                        nb_params = inputs[neighbor_key]
                        nb_smplx_out = self.model.smplx_server(
                            scale=nb_params["scale"],
                            betas=nb_params["betas"],
                            transl=nb_params["transl"],
                            global_orient=nb_params["global_orient"],
                            body_pose=nb_params["body_pose"],
                            left_hand_pose=nb_params["left_hand_pose"],
                            right_hand_pose=nb_params["right_hand_pose"],
                            expression=nb_params["expression"],
                            jaw_pose=nb_params["jaw_pose"],
                            leye_pose=nb_params["leye_pose"],
                            reye_pose=nb_params["reye_pose"],
                        )
                        inputs[verts_key] = nb_smplx_out["smplx_verts"].detach()
                else:
                    inputs[verts_key] = None

        else:
            inputs["smplx_scale"] = inputs["smplx_scale"]
            inputs["smplx_betas"] = inputs["smplx_betas"]
            smplx_params = inputs["smplx_params"]
            # split smplx_params
            inputs["smplx_global_orient"] = smplx_params[:, 0:3]
            inputs["smplx_body_pose"] = smplx_params[:, 3:66]
            inputs["smplx_left_hand_pose"] = smplx_params[:, 66:111]
            inputs["smplx_right_hand_pose"] = smplx_params[:, 111:156]
            inputs["smplx_transl"] = smplx_params[:, 156:159]
            inputs["smplx_expression"] = smplx_params[:, 159:169]
            inputs["smplx_jaw_pose"] = smplx_params[:, 169:173]
            inputs["smplx_leye_pose"] = torch.zeros_like(smplx_params[:, 0:3])
            inputs["smplx_reye_pose"] = torch.zeros_like(smplx_params[:, 0:3])

        if self.opt_camera_pose:
            inputs["pose"] = self.camera_pose_params(batch_idx)

        if self.opt_object_pose:
            inputs["obj_pose"] = self.object_pose_params(batch_idx)
            inputs["obj_pose_cur"] = self.object_pose_params(batch_idx)

            if batch_idx > 0:
                inputs["obj_pose_prev"] = self.object_pose_params(batch_idx - 1)
            else:
                inputs["obj_pose_prev"] = None
            if batch_idx < len(self.training_indices) - 1:
                inputs["obj_pose_next"] = self.object_pose_params(batch_idx + 1)
            else:
                inputs["obj_pose_next"] = None

        inputs["current_epoch"] = self.current_epoch
        inputs["global_step"] = self.global_step
        model_outputs = self.model(inputs)
        model_outputs["epoch"] = self.current_epoch
        model_outputs["batch_idx"] = batch_idx
        model_outputs["global_step"] = self.global_step

        loss_output = self.loss(model_outputs, targets)
        for k, v in loss_output.items():
            self.log(k, v.item(), prog_bar=True, on_step=True)

        return loss_output["train/loss"]

    def query_fg_sdf(self, x_c, cond):
        x_c = x_c.reshape(-1, 3)
        sdf = self.model.fg_implicit_network(x_c, cond)[:, :, 0].reshape(-1)
        # if x_c is far away from the origin, set sdf to large value
        sdf = torch.where(x_c.norm(dim=1) > 1.5, 10, sdf)
        return sdf

    def query_obj_sdf(self, x_c, cond):
        x_c = x_c.reshape(-1, 3)
        sdf = self.model.obj_implicit_network(x_c, cond)[:, :, 0].reshape(-1)
        return sdf

    def query_bg_sdf(self, x, cond):
        x = x.reshape(-1, 3)
        sdf = self.model.bg_implicit_network(x, cond)[:, :, 0].reshape(-1)
        return sdf

    @torch.no_grad()
    def _plot_sdf_slices(self, query_fn, cond, grid_min, grid_max, label, prefix,
                         sdf_range=0.2, level_step=0.02, resolution=256):
        """Generate XY, XZ, YZ slice plots for a single SDF field."""
        grid_min = np.asarray(grid_min, dtype=np.float32)
        grid_max = np.asarray(grid_max, dtype=np.float32)
        center = (grid_min + grid_max) / 2.0

        planes = [
            ("xy", 2, ("X", "Y")),
            ("xz", 1, ("X", "Z")),
            ("yz", 0, ("Y", "Z")),
        ]

        for plane_name, fixed_axis, axis_labels in planes:
            varying = [i for i in range(3) if i != fixed_axis]
            ax0, ax1 = varying

            u = np.linspace(grid_min[ax0], grid_max[ax0], resolution)
            v = np.linspace(grid_min[ax1], grid_max[ax1], resolution)
            uu, vv = np.meshgrid(u, v, indexing="ij")

            points_np = np.zeros((resolution * resolution, 3), dtype=np.float32)
            points_np[:, ax0] = uu.ravel()
            points_np[:, ax1] = vv.ravel()
            points_np[:, fixed_axis] = center[fixed_axis]

            points = torch.from_numpy(points_np).to(self.device)
            sdf_vals = query_sdf_batched(query_fn, points, cond).cpu().numpy()
            sdf_2d = sdf_vals.reshape(resolution, resolution)

            extent = [grid_min[ax0], grid_max[ax0], grid_min[ax1], grid_max[ax1]]
            title = f"{label} SDF — {plane_name.upper()} (fixed {'XYZ'[fixed_axis]}={center[fixed_axis]:.3f})"
            save_path = f"{prefix}_{plane_name}.png"
            plot_slice(sdf_2d, extent, axis_labels, title, save_path, sdf_range, level_step)

    @torch.no_grad()
    def _visualize_sdf_slices(self):
        """Generate 2D SDF slice plots for the object field."""
        if self.model.obj_verts_cano is not None:
            bbox_min = self.model.obj_verts_cano.min(dim=0).values.cpu().numpy()
            bbox_max = self.model.obj_verts_cano.max(dim=0).values.cpu().numpy()
            bbox_size = bbox_max - bbox_min
            half_extent = bbox_size.max() * 0.5 * 1.5
            center_bbox = (bbox_min + bbox_max) / 2.0
            obj_grid_min = center_bbox - half_extent
            obj_grid_max = center_bbox + half_extent
        else:
            limit = float(self.model.obj_bounding_sphere)
            obj_grid_min = np.array([-limit] * 3, dtype=np.float32)
            obj_grid_max = np.array([limit] * 3, dtype=np.float32)

        obj_grid_min, obj_grid_max, _ = make_cubic_grid(obj_grid_min, obj_grid_max)
        self._plot_sdf_slices(
            query_fn=self.query_obj_sdf,
            cond=None,
            grid_min=obj_grid_min,
            grid_max=obj_grid_max,
            label="Object",
            prefix=f"sdf_vis/{self.current_epoch:04d}_obj",
            sdf_range=0.2,
            level_step=0.02,
        )

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        stage = get_training_stage(
            self.current_epoch,
            self.num_epochs,
            self.cfg.model.loss.interactvlm_contact_weight,
            self.opt_pose_freq,
            self.opt_pose_len,
        )
        self._set_optimizer_mode(stage.optimizer_mode)
        self.enable_contact_loss = stage.enable_contact_loss

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.val_output_list = []
        return

    def validation_step(self, batch, *args, **kwargs):
        outputs = {}
        inputs, targets = batch
        batch_idx = inputs["idx"]

        if self.opt_smplx:
            body_model_params = self.body_model_params(batch_idx)
            inputs["smplx_scale"] = body_model_params["scale"]
            inputs["smplx_betas"] = body_model_params["betas"]
            inputs["smplx_transl"] = body_model_params["transl"]
            inputs["smplx_body_pose"] = body_model_params["body_pose"]
            inputs["smplx_global_orient"] = body_model_params["global_orient"]
            inputs["smplx_left_hand_pose"] = body_model_params["left_hand_pose"]
            inputs["smplx_right_hand_pose"] = body_model_params["right_hand_pose"]
            inputs["smplx_expression"] = body_model_params["expression"]  # None
            inputs["smplx_jaw_pose"] = body_model_params["jaw_pose"]
            inputs["smplx_leye_pose"] = body_model_params["leye_pose"]
            inputs["smplx_reye_pose"] = body_model_params["reye_pose"]
        else:
            inputs["smplx_scale"] = inputs["smplx_scale"]
            inputs["smplx_betas"] = inputs["smplx_betas"]
            smplx_params = inputs["smplx_params"]
            # split smplx_params
            inputs["smplx_global_orient"] = smplx_params[:, 0:3]
            inputs["smplx_body_pose"] = smplx_params[:, 3:66]
            inputs["smplx_left_hand_pose"] = smplx_params[:, 66:111]
            inputs["smplx_right_hand_pose"] = smplx_params[:, 111:156]
            inputs["smplx_transl"] = smplx_params[:, 156:159]
            inputs["smplx_expression"] = smplx_params[:, 159:169]
            inputs["smplx_jaw_pose"] = smplx_params[:, 169:173]
            inputs["smplx_leye_pose"] = torch.zeros_like(smplx_params[:, 0:3])
            inputs["smplx_reye_pose"] = torch.zeros_like(smplx_params[:, 0:3])

        if self.opt_camera_pose:
            inputs["pose"] = self.camera_pose_params(batch_idx)

        if self.opt_object_pose:
            inputs["obj_pose"] = self.object_pose_params(batch_idx)

        cond = {"smplx": inputs["smplx_body_pose"] / np.pi}
        mesh_min = self.model.mesh_v_cano[0].min(dim=0).values.cpu().numpy()
        mesh_max = self.model.mesh_v_cano[0].max(dim=0).values.cpu().numpy()
        bbox_size = mesh_max - mesh_min
        mesh_min = mesh_min - bbox_size * 0.1
        mesh_max = mesh_max + bbox_size * 0.1
        mesh_canonical = mesh_utils.get_surface_sliding(
            lambda x: self.query_fg_sdf(x, cond),
            resolution=128,
            grid_min=mesh_min,
            grid_max=mesh_max,
        )

        smplx_outputs = self.model.smplx_server(
            scale=inputs["smplx_scale"],
            betas=inputs["smplx_betas"],
            transl=inputs["smplx_transl"],
            body_pose=inputs["smplx_body_pose"],
            global_orient=inputs["smplx_global_orient"],
            left_hand_pose=inputs["smplx_left_hand_pose"],
            right_hand_pose=inputs["smplx_right_hand_pose"],
            expression=inputs["smplx_expression"],
            jaw_pose=inputs["smplx_jaw_pose"],
            leye_pose=inputs["smplx_leye_pose"],
            reye_pose=inputs["smplx_reye_pose"],
            absolute=False,
        )
        smplx_tfs = smplx_outputs["smplx_tfs"]
        verts_deformed = self.get_deformed_mesh(
            mesh_canonical.vertices, smplx_tfs, self.model.smplx_v_cano
        )
        mesh_deformed = trimesh.Trimesh(vertices=verts_deformed, faces=mesh_canonical.faces)

        if self.model.obj_verts_cano is not None:
            bbox_min = self.model.obj_verts_cano.min(dim=0).values.cpu().numpy()
            bbox_max = self.model.obj_verts_cano.max(dim=0).values.cpu().numpy()
            bbox_size = bbox_max - bbox_min
            grid_min = bbox_min - bbox_size * 0.2
            grid_max = bbox_max + bbox_size * 0.2
        else:
            obj_grid_limit = self.model.obj_bounding_sphere
            grid_min = [-obj_grid_limit, -obj_grid_limit, -obj_grid_limit]
            grid_max = [obj_grid_limit, obj_grid_limit, obj_grid_limit]
        mesh_obj = mesh_utils.get_surface_sliding(
            lambda x: self.query_obj_sdf(x, None),
            resolution=128,
            grid_min=grid_min,
            grid_max=grid_max,
        )
        if mesh_obj is None or len(mesh_obj.vertices) == 0:
            print(f"[WARNING] Object mesh extraction failed at epoch {self.current_epoch}")
            print(f"  grid_min={grid_min}, grid_max={grid_max}")
        obj_pose = inputs["obj_pose"]
        obj_pose_np = obj_pose.cpu().numpy()[0]
        inv_obj_pose_np = np.linalg.inv(obj_pose_np)
        mesh_obj_c = mesh_obj.copy()
        try:
            mesh_obj.apply_transform(inv_obj_pose_np)
        except Exception as e:
            print(f"Error in transforming object mesh: {e}")

        bg_grid_limit = self.model.bg_bounding_sphere
        mesh_scene = mesh_utils.get_surface_sliding(
            lambda x: self.query_bg_sdf(x, None),
            resolution=512,
            grid_min=[-bg_grid_limit, -bg_grid_limit, -bg_grid_limit],
            grid_max=[bg_grid_limit, bg_grid_limit, bg_grid_limit],
        )

        outputs["mesh_canonical"] = mesh_canonical
        outputs["mesh_deformed"] = mesh_deformed
        outputs["mesh_obj"] = mesh_obj
        outputs["mesh_obj_c"] = mesh_obj_c
        outputs["mesh_scene"] = mesh_scene
        outputs["idx"] = inputs["idx"]

        if not self.val_render_image:
            self.val_output_list.append(outputs)
            return outputs

        total_pixels = inputs["total_pixels"][0]
        split = idr_utils.split_input(
            inputs,
            total_pixels,
            n_pixels=min(
                self.pixel_per_batch,
                targets["img_size"][0] * targets["img_size"][1],
            ),
        )

        res = []
        for s in split:
            out = self.model(s)
            for k, v in out.items():
                try:
                    out[k] = v.detach()
                except AttributeError:
                    out[k] = v
            res.append(
                {
                    "rgb": out["rgb"].detach(),
                    "fg_rgb": out["fg_rgb"].detach(),
                    "obj_rgb": out["obj_rgb"].detach(),
                    "bg_rgb": out["bg_rgb"].detach(),
                    "fg_acc_map": out["fg_acc_map"].detach(),
                    "obj_acc_map": out["obj_acc_map"].detach(),
                    "normal": out["normal"].detach(),
                    "fg_normal": out["fg_normal"].detach(),
                    "obj_normal": out["obj_normal"].detach(),
                    "bg_normal": out["bg_normal"].detach(),
                    "depth": out["depth"].detach(),
                    "fg_depth": out["fg_depth"].detach(),
                    "obj_depth": out["obj_depth"].detach(),
                    "bg_depth": out["bg_depth"].detach(),
                }
            )
        batch_size = targets["rgb"].shape[0]

        model_outputs = idr_utils.merge_output(res, total_pixels, batch_size)

        outputs.update(
            {
                "fg_acc_map": model_outputs["fg_acc_map"].detach(),
                "obj_acc_map": model_outputs["obj_acc_map"].detach(),
                "rgb_pred": model_outputs["rgb"].detach(),
                "fg_rgb_pred": model_outputs["fg_rgb"].detach(),
                "obj_rgb_pred": model_outputs["obj_rgb"].detach(),
                "bg_rgb_pred": model_outputs["bg_rgb"].detach(),
                "normal_pred": model_outputs["normal"].detach(),
                "fg_normal_pred": model_outputs["fg_normal"].detach(),
                "obj_normal_pred": model_outputs["obj_normal"].detach(),
                "bg_normal_pred": model_outputs["bg_normal"].detach(),
                "depth_pred": model_outputs["depth"].detach().clone(),
                "fg_depth_pred": model_outputs["fg_depth"].detach(),
                "obj_depth_pred": model_outputs["obj_depth"].detach(),
                "bg_depth_pred": model_outputs["bg_depth"].detach(),
                **targets,
            }
        )

        self.val_output_list.append(outputs)
        return outputs

    def on_validation_epoch_end(self) -> None:
        output = self.val_output_list[0]
        idx = output["idx"].item()
        mesh_canonical = output["mesh_canonical"]
        mesh_canonical = mesh_utils.simplify_mesh(mesh_canonical)
        mesh_canonical.export(f"meshes/{self.current_epoch:04d}_{idx:02d}_human.ply")
        mesh_deformed = output["mesh_deformed"]
        mesh_deformed = mesh_utils.simplify_mesh(mesh_deformed)
        mesh_deformed.export(f"meshes/{self.current_epoch:04d}_{idx:02d}_human_posed.ply")
        mesh_obj = output["mesh_obj"]
        mesh_obj_c = output["mesh_obj_c"]
        try:
            if mesh_obj is not None and hasattr(mesh_obj, 'vertices') and len(mesh_obj.vertices) > 0:
                mesh_obj = mesh_utils.simplify_mesh(mesh_obj)
                mesh_obj.export(f"meshes/{self.current_epoch:04d}_{idx:02d}_object.ply")
            else:
                print(f"[WARNING] Skipping object mesh export - empty or invalid mesh at epoch {self.current_epoch}")
        except Exception as e:
            print(f"[ERROR] Failed to export object mesh: {e}")
        mesh_scene = output["mesh_scene"]
        mesh_scene = mesh_utils.simplify_mesh(mesh_scene)
        mesh_scene.export(f"meshes/{self.current_epoch:04d}_{idx:02d}_scene.ply")

        if self.update_canonical_mesh:
            self.model.mesh_v_cano = torch.tensor(
                mesh_canonical.vertices[None], device=self.device
            ).float()
            self.model.mesh_f_cano = torch.tensor(
                mesh_canonical.faces.astype(np.int64), device=self.device
            )
            self.model.mesh_face_vertices = index_vertices_by_faces(
                self.model.mesh_v_cano, self.model.mesh_f_cano
            )
            if mesh_obj_c is not None and hasattr(mesh_obj_c, 'vertices') and len(mesh_obj_c.vertices) > 0:
                self.model.obj_verts_cano = torch.tensor(
                    mesh_obj_c.vertices, device=self.device
                ).float()
            else:
                print(f"[WARNING] Skipping obj_verts_cano update - invalid mesh at epoch {self.current_epoch}")

        if self.model.reg_hands:
            with torch.no_grad():
                betas = self.body_model_params.betas.weight.data.clone().detach()

                self.model.smplx_server.get_cano_smplx(betas, update=True)
                self.model.smplx_v_cano = self.model.smplx_server.verts_c
                self.model.smplx_face_vertices = index_vertices_by_faces(
                    self.model.smplx_v_cano, self.model.smplx_f_cano
                )
        scale = self.body_model_params.scale.weight.data.clone().detach()
        betas = self.body_model_params.betas.weight.data.clone().detach()
        self.model.fg_deformer.update_cano_smplx(scale, betas)

        self._visualize_sdf_slices()

        if not self.val_render_image:
            self.val_output_list.clear()
            return
        img_size = output["img_size"]
        h, w = img_size
        concat_dim = 1 if h > w else 0

        depth_pred = output["depth_pred"]
        depth_pred = depth_pred.reshape(*img_size, -1)
        fg_depth_pred = output["fg_depth_pred"]
        fg_depth_pred = fg_depth_pred.reshape(*img_size, -1)
        obj_depth_pred = output["obj_depth_pred"]
        obj_depth_pred = obj_depth_pred.reshape(*img_size, -1)
        bg_depth_pred = output["bg_depth_pred"]
        bg_depth_pred = bg_depth_pred.reshape(*img_size, -1)

        rgb_pred = output["rgb_pred"]
        rgb_pred = rgb_pred.reshape(*img_size, -1)
        rgb_gt = output["rgb"]
        rgb_gt = rgb_gt.reshape(*img_size, -1)
        rgb = torch.cat([rgb_gt, rgb_pred], dim=concat_dim).cpu().numpy()
        rgb_error = torch.abs(rgb_pred - rgb_gt).mean(dim=-1)
        rgb_error *= 5
        rgb_error = self.sequential_colormap(rgb_error.cpu().numpy())[:, :, :3]
        rgb = np.concatenate([rgb, rgb_error], axis=concat_dim)
        rgb = (rgb * 255).astype(np.uint8)

        fg_rgb_pred = output["fg_rgb_pred"]
        fg_rgb_pred = fg_rgb_pred.reshape(*img_size, -1)
        fg_rgb = torch.cat([rgb_gt, fg_rgb_pred], dim=concat_dim).cpu().numpy()
        fg_rgb = (fg_rgb * 255).astype(np.uint8)
        obj_rgb_pred = output["obj_rgb_pred"]
        obj_rgb_pred = obj_rgb_pred.reshape(*img_size, -1)
        obj_rgb = torch.cat([rgb_gt, obj_rgb_pred], dim=concat_dim).cpu().numpy()
        obj_rgb = (obj_rgb * 255).astype(np.uint8)
        bg_rgb_pred = output["bg_rgb_pred"]
        bg_rgb_pred = bg_rgb_pred.reshape(*img_size, -1)
        bg_rgb = torch.cat([rgb_gt, bg_rgb_pred], dim=concat_dim).cpu().numpy()
        bg_rgb = (bg_rgb * 255).astype(np.uint8)

        fg_acc_map = output["fg_acc_map"]
        fg_acc_map = fg_acc_map.reshape(*img_size, -1)
        fg_acc_map = fg_acc_map.repeat(1, 1, 3)
        fg_acc_map = torch.cat([rgb_gt * fg_acc_map, fg_acc_map], dim=concat_dim).cpu().numpy()
        fg_acc_map = (fg_acc_map * 255).astype(np.uint8)
        obj_acc_map = output["obj_acc_map"]
        obj_acc_map = obj_acc_map.reshape(*img_size, -1)
        obj_acc_map = obj_acc_map.repeat(1, 1, 3)
        obj_acc_map = torch.cat([rgb_gt * obj_acc_map, obj_acc_map], dim=concat_dim).cpu().numpy()
        obj_acc_map = (obj_acc_map * 255).astype(np.uint8)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        rgb_pred = torch.moveaxis(rgb_pred, -1, 0)[None, ...]
        rgb_gt = torch.moveaxis(rgb_gt, -1, 0)[None, ...]
        psnr = self.psnr(rgb_pred, rgb_gt)
        ssim = self.ssim(rgb_pred, rgb_gt)
        lpips = self.lpips(rgb_pred, rgb_gt)
        self.log("val/psnr", psnr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ssim", ssim, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/lpips", lpips, on_step=False, on_epoch=True, prog_bar=True)

        normal_pred = output["normal_pred"].reshape(*img_size, -1)
        fg_normal_pred = output["fg_normal_pred"].reshape(*img_size, -1)
        obj_normal_pred = output["obj_normal_pred"].reshape(*img_size, -1)
        bg_normal_pred = output["bg_normal_pred"].reshape(*img_size, -1)
        normal_gt = output["normal"].reshape(*img_size, -1)
        normal_error = 1.0 - torch.sum(normal_pred * normal_gt, dim=-1)
        normal = torch.cat([normal_gt, normal_pred], dim=concat_dim).cpu().numpy()
        normal = (normal + 1.0) * 0.5
        normal_cos = normal_error.mean()
        # map to [0, 1]
        normal_error *= 0.5
        # for visualization
        normal_error *= 5
        normal_error = self.sequential_colormap(normal_error.cpu().numpy())[:, :, :3]
        normal = np.concatenate([normal, normal_error], axis=concat_dim)
        normal = (normal * 255).astype(np.uint8)
        fg_normal = torch.cat([normal_gt, fg_normal_pred], dim=concat_dim).cpu().numpy()
        fg_normal = (fg_normal + 1.0) * 0.5
        fg_normal = (fg_normal * 255).astype(np.uint8)
        obj_normal = torch.cat([normal_gt, obj_normal_pred], dim=concat_dim).cpu().numpy()
        obj_normal = (obj_normal + 1.0) * 0.5
        obj_normal = (obj_normal * 255).astype(np.uint8)
        bg_normal = torch.cat([normal_gt, bg_normal_pred], dim=concat_dim).cpu().numpy()
        bg_normal = (bg_normal + 1.0) * 0.5
        bg_normal = (bg_normal * 255).astype(np.uint8)

        # normal_l1 = torch.abs(normal_gt - normal_pred).sum(dim=-1).mean()
        # self.log("val/normal_l1", normal_l1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/normal_cos", normal_cos, on_step=False, on_epoch=True, prog_bar=True)

        depth_gt = output["depth"]
        depth_gt = depth_gt.reshape(*img_size, -1)
        # depth_gt = (depth_gt.cpu().numpy() * 255).astype(np.uint8)
        # depth_gt = cv2.applyColorMap(depth_gt, cv2.COLORMAP_VIRIDIS)

        # fg_rgb = torch.cat([fg_rgb_pred], dim=0).cpu().numpy()
        # fg_rgb = (fg_rgb * 255).astype(np.uint8)

        depth_gt = rearrange(depth_gt, "h w 1 -> 1 h w")
        depth_pred = rearrange(depth_pred, "h w 1 -> 1 h w")
        # Scale/shift alignment is kept even when pred and gt depths share a
        # metric scale, since the model is trained with this alignment.
        # depth_l1 = torch.abs(depth_gt - depth_pred).mean()
        # TODO: fix this scale / shift issue
        mask = torch.ones_like(depth_gt)
        scale, shift = compute_scale_and_shift(depth_pred, depth_gt, mask)
        depth_l1 = torch.abs(depth_gt - depth_pred * scale - shift).mean()
        depth_pred_aligned = depth_pred * scale + shift
        self.log("val/depth_l1", depth_l1, on_step=False, on_epoch=True, prog_bar=True)

        depth_gt = rearrange(depth_gt, "1 h w -> h w")
        depth_pred = rearrange(depth_pred, "1 h w -> h w")
        depth_pred_aligned = rearrange(depth_pred_aligned, "1 h w -> h w")
        depth = torch.cat([depth_gt, depth_pred_aligned], dim=concat_dim).cpu().numpy()
        vis_scale = 1.0 / (depth.max() - depth.min())
        vis_shift = -depth.min()
        depth = (depth + vis_shift) * vis_scale
        depth = self.depth_colormap(depth)[:, :, :3]
        depth_error = depth_pred - depth_gt
        depth_error = (depth_error - depth_error.min()) / (depth_error.max() - depth_error.min())
        depth_error = self.diverging_colormap(depth_error.cpu().numpy())[:, :, :3]
        depth = np.concatenate([depth, depth_error], axis=concat_dim)
        depth = (depth * 255).astype(np.uint8)
        fg_depth_aligned = fg_depth_pred * scale + shift
        fg_depth_aligned = rearrange(fg_depth_aligned, "h w 1 -> h w")
        fg_depth = torch.cat([depth_gt, fg_depth_aligned], dim=concat_dim).cpu().numpy()
        fg_depth = (fg_depth + vis_shift) * vis_scale
        fg_depth = self.depth_colormap(fg_depth)[:, :, :3]
        fg_depth = (fg_depth * 255).astype(np.uint8)
        obj_depth_aligned = obj_depth_pred * scale + shift
        obj_depth_aligned = rearrange(obj_depth_aligned, "h w 1 -> h w")
        obj_depth = torch.cat([depth_gt, obj_depth_aligned], dim=concat_dim).cpu().numpy()
        obj_depth = (obj_depth + vis_shift) * vis_scale
        obj_depth = self.depth_colormap(obj_depth)[:, :, :3]
        obj_depth = (obj_depth * 255).astype(np.uint8)
        bg_depth_aligned = bg_depth_pred * scale + shift
        bg_depth_aligned = rearrange(bg_depth_aligned, "h w 1 -> h w")
        bg_depth = torch.cat([depth_gt, bg_depth_aligned], dim=concat_dim).cpu().numpy()
        bg_depth = (bg_depth + vis_shift) * vis_scale
        bg_depth = self.depth_colormap(bg_depth)[:, :, :3]
        bg_depth = (bg_depth * 255).astype(np.uint8)

        cv2.imwrite(f"rgb/{self.current_epoch:04d}_{idx:02d}.png", rgb[:, :, ::-1])
        cv2.imwrite(f"rgb/{self.current_epoch:04d}_{idx:02d}_fg.png", fg_rgb[:, :, ::-1])
        cv2.imwrite(f"rgb/{self.current_epoch:04d}_{idx:02d}_obj.png", obj_rgb[:, :, ::-1])
        cv2.imwrite(f"rgb/{self.current_epoch:04d}_{idx:02d}_bg.png", bg_rgb[:, :, ::-1])
        cv2.imwrite(f"rgb/{self.current_epoch:04d}_{idx:02d}_fg_mask.png", fg_acc_map[:, :, ::-1])
        cv2.imwrite(f"rgb/{self.current_epoch:04d}_{idx:02d}_obj_mask.png", obj_acc_map[:, :, ::-1])
        cv2.imwrite(f"normal/{self.current_epoch:04d}_{idx:02d}.png", normal[:, :, ::-1])
        cv2.imwrite(f"normal/{self.current_epoch:04d}_{idx:02d}_fg.png", fg_normal[:, :, ::-1])
        cv2.imwrite(f"normal/{self.current_epoch:04d}_{idx:02d}_obj.png", obj_normal[:, :, ::-1])
        cv2.imwrite(f"normal/{self.current_epoch:04d}_{idx:02d}_bg.png", bg_normal[:, :, ::-1])
        cv2.imwrite(f"depth/{self.current_epoch:04d}_{idx:02d}.png", depth[:, :, ::-1])
        cv2.imwrite(f"depth/{self.current_epoch:04d}_{idx:02d}_fg.png", fg_depth[:, :, ::-1])
        cv2.imwrite(f"depth/{self.current_epoch:04d}_{idx:02d}_obj.png", obj_depth[:, :, ::-1])
        cv2.imwrite(f"depth/{self.current_epoch:04d}_{idx:02d}_bg.png", bg_depth[:, :, ::-1])

        self.val_output_list.clear()
        return

    def on_test_start(self):
        prefix = f"epoch_{self.current_epoch:04d}"
        Path(prefix).mkdir(parents=False, exist_ok=True)
        Path(f"{prefix}/rgb").mkdir(parents=False, exist_ok=True)
        Path(f"{prefix}/rgb_fg").mkdir(parents=False, exist_ok=True)
        Path(f"{prefix}/rgb_obj").mkdir(parents=False, exist_ok=True)
        Path(f"{prefix}/rgb_bg").mkdir(parents=False, exist_ok=True)
        Path(f"{prefix}/fg_mask").mkdir(parents=False, exist_ok=True)
        Path(f"{prefix}/obj_mask").mkdir(parents=False, exist_ok=True)
        Path(f"{prefix}/normal").mkdir(parents=False, exist_ok=True)
        Path(f"{prefix}/normal_fg").mkdir(parents=False, exist_ok=True)
        Path(f"{prefix}/normal_obj").mkdir(parents=False, exist_ok=True)
        Path(f"{prefix}/normal_bg").mkdir(parents=False, exist_ok=True)
        Path(f"{prefix}/depth").mkdir(parents=False, exist_ok=True)
        Path(f"{prefix}/depth_fg").mkdir(parents=False, exist_ok=True)
        Path(f"{prefix}/depth_obj").mkdir(parents=False, exist_ok=True)
        Path(f"{prefix}/depth_bg").mkdir(parents=False, exist_ok=True)
        Path(f"{prefix}/mesh_canonical").mkdir(parents=False, exist_ok=True)
        Path(f"{prefix}/mesh_deformed").mkdir(parents=False, exist_ok=True)
        Path(f"{prefix}/mesh_obj").mkdir(parents=False, exist_ok=True)
        self.model.fg_deformer.K = 3
        print("Reset deformer K to 3 to avoid artifacts")
        assert self.model.obj_verts_cano is not None
        grid_min = self.model.obj_verts_cano.min(dim=0).values.cpu().numpy()
        grid_max = self.model.obj_verts_cano.max(dim=0).values.cpu().numpy()
        bbox_size = grid_max - grid_min
        grid_min = grid_min - bbox_size * 0.2
        grid_max = grid_max + bbox_size * 0.2
        mesh_obj = mesh_utils.get_surface_sliding(
            lambda x: self.query_obj_sdf(x, None),
            resolution=384,
            grid_min=grid_min,
            grid_max=grid_max,
        )
        mesh_obj.export(f"{prefix}/obj.ply")
        bg_grid_limit = self.model.bg_bounding_sphere
        mesh_scene = mesh_utils.get_surface_sliding(
            lambda x: self.query_bg_sdf(x, None),
            resolution=512,
            grid_min=[-bg_grid_limit, -bg_grid_limit, -bg_grid_limit],
            grid_max=[bg_grid_limit, bg_grid_limit, bg_grid_limit],
        )
        mesh_scene.export(f"{prefix}/scene.ply")
        self.mesh_obj = mesh_obj
        return

    def test_step(self, batch, *args, **kwargs):
        inputs, targets = batch
        idx = inputs["idx"]
        idx_int = idx.item()
        total_pixels = inputs["total_pixels"]
        prefix = f"epoch_{self.current_epoch:04d}"

        if self.opt_smplx:
            body_model_params = self.body_model_params(idx)
            inputs["smplx_scale"] = body_model_params["scale"]
            inputs["smplx_betas"] = body_model_params["betas"]
            inputs["smplx_transl"] = body_model_params["transl"]
            inputs["smplx_body_pose"] = body_model_params["body_pose"]
            inputs["smplx_global_orient"] = body_model_params["global_orient"]
            inputs["smplx_left_hand_pose"] = body_model_params["left_hand_pose"]
            inputs["smplx_right_hand_pose"] = body_model_params["right_hand_pose"]
            inputs["smplx_expression"] = body_model_params["expression"]  # None
            inputs["smplx_jaw_pose"] = body_model_params["jaw_pose"]
            inputs["smplx_leye_pose"] = body_model_params["leye_pose"]
            inputs["smplx_reye_pose"] = body_model_params["reye_pose"]
        else:
            inputs["smplx_scale"] = inputs["smplx_scale"]
            inputs["smplx_betas"] = inputs["smplx_betas"]
            smplx_params = inputs["smplx_params"]
            # split smplx_params
            inputs["smplx_global_orient"] = smplx_params[:, 0:3]
            inputs["smplx_body_pose"] = smplx_params[:, 3:66]
            inputs["smplx_left_hand_pose"] = smplx_params[:, 66:111]
            inputs["smplx_right_hand_pose"] = smplx_params[:, 111:156]
            inputs["smplx_transl"] = smplx_params[:, 156:159]
            inputs["smplx_expression"] = smplx_params[:, 159:169]
            inputs["smplx_jaw_pose"] = smplx_params[:, 169:173]
            inputs["smplx_leye_pose"] = torch.zeros_like(smplx_params[:, 0:3])
            inputs["smplx_reye_pose"] = torch.zeros_like(smplx_params[:, 0:3])

        if self.opt_camera_pose:
            inputs["pose"] = self.camera_pose_params(idx)

        if self.opt_object_pose:
            inputs["obj_pose"] = self.object_pose_params(idx)

        cond = {"smplx": inputs["smplx_body_pose"] / np.pi}
        mesh_min = self.model.mesh_v_cano[0].min(dim=0).values.cpu().numpy()
        mesh_max = self.model.mesh_v_cano[0].max(dim=0).values.cpu().numpy()
        bbox_size = mesh_max - mesh_min
        mesh_min = mesh_min - bbox_size * 0.1
        mesh_max = mesh_max + bbox_size * 0.1
        mesh_canonical = mesh_utils.get_surface_sliding(
            lambda x: self.query_fg_sdf(x, cond),
            resolution=384,
            grid_min=mesh_min,
            grid_max=mesh_max,
        )
        mesh_canonical = mesh_utils.simplify_mesh(mesh_canonical)
        mesh_canonical.export(f"{prefix}/mesh_canonical/{idx_int:04d}.ply")

        smplx_outputs = self.model.smplx_server(
            scale=inputs["smplx_scale"],
            betas=inputs["smplx_betas"],
            transl=inputs["smplx_transl"],
            body_pose=inputs["smplx_body_pose"],
            global_orient=inputs["smplx_global_orient"],
            left_hand_pose=inputs["smplx_left_hand_pose"],
            right_hand_pose=inputs["smplx_right_hand_pose"],
            expression=inputs["smplx_expression"],
            jaw_pose=inputs["smplx_jaw_pose"],
            leye_pose=inputs["smplx_leye_pose"],
            reye_pose=inputs["smplx_reye_pose"],
            absolute=False,
        )
        smplx_tfs = smplx_outputs["smplx_tfs"]
        verts_deformed = self.get_deformed_mesh(
            mesh_canonical.vertices, smplx_tfs, self.model.smplx_v_cano
        )
        mesh_deformed = trimesh.Trimesh(vertices=verts_deformed, faces=mesh_canonical.faces)
        mesh_deformed = mesh_utils.simplify_mesh(mesh_deformed)
        mesh_deformed.export(f"{prefix}/mesh_deformed/{idx_int:04d}.ply")

        mesh_obj = self.mesh_obj.copy()
        obj_pose = inputs["obj_pose"]
        obj_pose_np = obj_pose.cpu().numpy()[0]
        inv_obj_pose_np = np.linalg.inv(obj_pose_np)
        mesh_obj.apply_transform(inv_obj_pose_np)
        mesh_obj = mesh_utils.simplify_mesh(mesh_obj)
        mesh_obj.export(f"{prefix}/mesh_obj/{idx_int:04d}.ply")

        if not self.test_render_image:
            return {}

        pixel_per_batch = self.pixel_per_batch
        num_splits = torch.div(
            (total_pixels + pixel_per_batch - 1), pixel_per_batch, rounding_mode="floor"
        )
        results = []
        for i in range(num_splits):
            indices = list(range(i * pixel_per_batch, min((i + 1) * pixel_per_batch, total_pixels)))
            batch_inputs = {
                "uv": inputs["uv"][:, indices],
                "pose": inputs["pose"],
                "intrinsics": inputs["intrinsics"],
                "obj_pose": inputs["obj_pose"],
                "idx": idx,
            }
            for k in inputs.keys():
                if k not in batch_inputs and "smplx" in k:
                    batch_inputs[k] = inputs[k]

            batch_targets = {
                "rgb_gt": targets["rgb"][:, indices].detach(),
                "normal_gt": targets["normal"][:, indices].detach(),
                "depth_gt": targets["depth"][:, indices].detach(),
                "img_size": targets["img_size"],
            }

            with torch.no_grad():
                model_outputs = self.model(batch_inputs)
            results.append(
                {
                    "fg_acc_map": model_outputs["fg_acc_map"].detach(),
                    "obj_acc_map": model_outputs["obj_acc_map"].detach(),
                    "rgb_pred": model_outputs["rgb"].detach(),
                    "fg_rgb_pred": model_outputs["fg_rgb"].detach(),
                    "obj_rgb_pred": model_outputs["obj_rgb"].detach(),
                    "bg_rgb_pred": model_outputs["bg_rgb"].detach(),
                    "normal_pred": model_outputs["normal"].detach(),
                    "fg_normal_pred": model_outputs["fg_normal"].detach(),
                    "obj_normal_pred": model_outputs["obj_normal"].detach(),
                    "bg_normal_pred": model_outputs["bg_normal"].detach(),
                    "depth_pred": model_outputs["depth"].detach(),
                    "fg_depth_pred": model_outputs["fg_depth"].detach(),
                    "obj_depth_pred": model_outputs["obj_depth"].detach(),
                    "bg_depth_pred": model_outputs["bg_depth"].detach(),
                    **batch_targets,
                }
            )

        img_size = results[0]["img_size"]
        rgb_pred = torch.cat([result["rgb_pred"][0] for result in results], dim=0)
        rgb_pred = rgb_pred.reshape(*img_size, -1)
        fg_rgb_pred = torch.cat([result["fg_rgb_pred"][0] for result in results], dim=0)
        fg_rgb_pred = fg_rgb_pred.reshape(*img_size, -1)
        obj_rgb_pred = torch.cat([result["obj_rgb_pred"][0] for result in results], dim=0)
        obj_rgb_pred = obj_rgb_pred.reshape(*img_size, -1)
        bg_rgb_pred = torch.cat([result["bg_rgb_pred"][0] for result in results], dim=0)
        bg_rgb_pred = bg_rgb_pred.reshape(*img_size, -1)
        normal_pred = torch.cat([result["normal_pred"][0] for result in results], dim=0)
        normal_pred = normal_pred.reshape(*img_size, -1)
        fg_normal_pred = torch.cat([result["fg_normal_pred"][0] for result in results], dim=0)
        fg_normal_pred = fg_normal_pred.reshape(*img_size, -1)
        obj_normal_pred = torch.cat([result["obj_normal_pred"][0] for result in results], dim=0)
        obj_normal_pred = obj_normal_pred.reshape(*img_size, -1)
        bg_normal_pred = torch.cat([result["bg_normal_pred"][0] for result in results], dim=0)
        bg_normal_pred = bg_normal_pred.reshape(*img_size, -1)
        depth_pred = torch.cat([result["depth_pred"][0] for result in results], dim=0)
        depth_pred = depth_pred.reshape(*img_size, -1)
        fg_depth_pred = torch.cat([result["fg_depth_pred"][0] for result in results], dim=0)
        fg_depth_pred = fg_depth_pred.reshape(*img_size, -1)
        obj_depth_pred = torch.cat([result["obj_depth_pred"][0] for result in results], dim=0)
        obj_depth_pred = obj_depth_pred.reshape(*img_size, -1)
        bg_depth_pred = torch.cat([result["bg_depth_pred"][0] for result in results], dim=0)
        bg_depth_pred = bg_depth_pred.reshape(*img_size, -1)
        fg_mask = torch.cat([result["fg_acc_map"][0] for result in results], dim=0)
        fg_mask = fg_mask.reshape(*img_size, -1)
        fg_mask = fg_mask.repeat(1, 1, 3)
        obj_mask = torch.cat([result["obj_acc_map"][0] for result in results], dim=0)
        obj_mask = obj_mask.reshape(*img_size, -1)
        obj_mask = obj_mask.repeat(1, 1, 3)

        h, w = img_size
        concat_dim = 1 if h > w else 0

        # RGB part
        rgb_gt = torch.cat([result["rgb_gt"] for result in results], dim=1).squeeze(0)
        rgb_gt = rgb_gt.reshape(*img_size, -1)
        rgb_error = torch.abs(rgb_pred - rgb_gt).mean(dim=-1)
        rgb_error = self.sequential_colormap(rgb_error.cpu().numpy())[:, :, :3]
        # scale it up for better visualization
        rgb_error *= 5
        rgb = torch.cat([rgb_gt, rgb_pred], dim=concat_dim).cpu().numpy()
        rgb = np.concatenate([rgb, rgb_error], axis=concat_dim)
        fg_rgb = torch.cat([rgb_gt, fg_rgb_pred], dim=concat_dim).cpu().numpy()
        obj_rgb = torch.cat([rgb_gt, obj_rgb_pred], dim=concat_dim).cpu().numpy()
        bg_rgb = torch.cat([rgb_gt, bg_rgb_pred], dim=concat_dim).cpu().numpy()
        fg_mask = torch.cat([rgb_gt * fg_mask, fg_mask], dim=concat_dim).cpu().numpy()
        obj_mask = torch.cat([rgb_gt * obj_mask, obj_mask], dim=concat_dim).cpu().numpy()

        # Normal part
        normal_gt = torch.cat([result["normal_gt"] for result in results], dim=1).squeeze(0)
        normal_gt = normal_gt.reshape(*img_size, -1)
        normal_error = 1 - torch.sum(normal_pred * normal_gt, dim=-1)
        # map from [0, 2] to [0, 1] to align with normal space
        normal_error *= 0.5
        # for visualization
        normal_error *= 5
        normal_error = self.sequential_colormap(normal_error.cpu().numpy())[:, :, :3]
        normal = torch.cat([normal_gt, normal_pred], dim=concat_dim).cpu().numpy()
        normal = (normal + 1.0) * 0.5
        normal = np.concatenate([normal, normal_error], axis=concat_dim)
        fg_normal = torch.cat([normal_gt, fg_normal_pred], dim=concat_dim).cpu().numpy()
        obj_normal = torch.cat([normal_gt, obj_normal_pred], dim=concat_dim).cpu().numpy()
        bg_normal = torch.cat([normal_gt, bg_normal_pred], dim=concat_dim).cpu().numpy()

        # Depth part
        # We use monocular prediction as GT for real sequences
        depth_gt = torch.cat([result["depth_gt"] for result in results], dim=1).squeeze(0)
        depth_gt = depth_gt.reshape(*img_size, -1)
        depth_gt = rearrange(depth_gt, "h w 1 -> 1 h w")
        depth_pred = rearrange(depth_pred, "h w 1 -> 1 h w")
        # Now we only consider the case in synthetic setting
        # where we have metric ground truth and cameras are aligned with gt cameras
        # we still need this scale and shift since it is trained in this way
        # TODO: change to per-image scale and shift and check values learned from training
        mask = torch.ones_like(depth_gt)
        # TODO: robusify this least square fit
        scale, shift = compute_scale_and_shift(depth_pred, depth_gt, mask)
        depth_l1 = torch.abs(depth_gt - depth_pred * scale - shift).mean()
        depth_pred_aligned = depth_pred * scale + shift
        # depth_l1 = torch.abs(depth_gt - depth_pred).mean()
        fg_depth_pred_aligned = fg_depth_pred * scale + shift
        obj_depth_pred_aligned = obj_depth_pred * scale + shift
        bg_depth_pred_aligned = bg_depth_pred * scale + shift
        depth_gt = rearrange(depth_gt, "1 h w -> h w 1")
        depth_pred_aligned = rearrange(depth_pred_aligned, "1 h w -> h w 1")
        depth_pred = rearrange(depth_pred, "1 h w -> h w 1")
        depth = torch.cat([depth_gt, depth_pred_aligned], dim=concat_dim)
        vis_scale = 1 / (depth.max() - depth.min())
        vis_shift = depth.min()
        depth = (depth - vis_shift) * vis_scale
        depth = depth.cpu().numpy()
        depth_error = depth_pred - depth_gt
        depth_error = rearrange(depth_error, "h w 1-> h w")
        depth_error = (depth_error - depth_error.min()) / (depth_error.max() - depth_error.min())
        depth_error = self.diverging_colormap(depth_error.cpu().numpy())[:, :, :3]
        # depth = (depth - depth.min()) / (depth.max() - depth.min())
        depth = rearrange(depth, "h w 1 -> h w")
        depth = self.depth_colormap(depth)[:, :, :3]
        # depth = (depth * 255).astype(np.uint8)
        # depth = cv2.applyColorMap(depth, cv2.COLORMAP_VIRIDIS)
        depth = np.concatenate([depth, depth_error], axis=concat_dim)
        fg_depth = torch.cat([depth_gt, fg_depth_pred_aligned], dim=concat_dim)
        fg_depth = (fg_depth - vis_shift) * vis_scale
        fg_depth = fg_depth.cpu().numpy()
        fg_depth = rearrange(fg_depth, "h w 1 -> h w")
        fg_depth = self.depth_colormap(fg_depth)[:, :, :3]
        obj_depth = torch.cat([depth_gt, obj_depth_pred_aligned], dim=concat_dim)
        obj_depth = (obj_depth - vis_shift) * vis_scale
        obj_depth = obj_depth.cpu().numpy()
        obj_depth = rearrange(obj_depth, "h w 1 -> h w")
        obj_depth = self.depth_colormap(obj_depth)[:, :, :3]
        bg_depth = torch.cat([depth_gt, bg_depth_pred_aligned], dim=concat_dim)
        bg_depth = (bg_depth - vis_shift) * vis_scale
        bg_depth = bg_depth.cpu().numpy()
        bg_depth = rearrange(bg_depth, "h w 1 -> h w")
        bg_depth = self.depth_colormap(bg_depth)[:, :, :3]

        # usually we don't need to save these numerical results
        if self.test_save_numerical_results:
            np.save(f"{prefix}/normal/{idx_int:04d}.npy", normal_pred.cpu().numpy())
            np.save(f"{prefix}/fg_normal/{idx_int:04d}_fg.npy", fg_normal_pred.cpu().numpy())
            np.save(f"{prefix}/obj_normal/{idx_int:04d}_obj.npy", obj_normal_pred.cpu().numpy())
            np.save(f"{prefix}/bg_normal/{idx_int:04d}_bg.npy", bg_normal_pred.cpu().numpy())
            np.save(f"{prefix}/depth/{idx_int:04d}.npy", depth_pred.cpu().numpy())
            np.save(f"{prefix}/fg_depth/{idx_int:04d}_fg.npy", fg_depth_pred.cpu().numpy())
            np.save(f"{prefix}/obj_depth/{idx_int:04d}_obj.npy", obj_depth_pred.cpu().numpy())
            np.save(f"{prefix}/bg_depth/{idx_int:04d}_bg.npy", bg_depth_pred.cpu().numpy())

        rgb = (rgb * 255).astype(np.uint8)
        fg_rgb = (fg_rgb * 255).astype(np.uint8)
        obj_rgb = (obj_rgb * 255).astype(np.uint8)
        bg_rgb = (bg_rgb * 255).astype(np.uint8)
        # normal = (normal + 1) / 2
        normal = (normal * 255).astype(np.uint8)
        fg_normal = (fg_normal + 1) / 2
        fg_normal = (fg_normal * 255).astype(np.uint8)
        obj_normal = (obj_normal + 1) / 2
        obj_normal = (obj_normal * 255).astype(np.uint8)
        bg_normal = (bg_normal + 1) / 2
        bg_normal = (bg_normal * 255).astype(np.uint8)
        depth = (depth * 255).astype(np.uint8)
        fg_depth = (fg_depth * 255).astype(np.uint8)
        obj_depth = (obj_depth * 255).astype(np.uint8)
        bg_depth = (bg_depth * 255).astype(np.uint8)
        fg_mask = (fg_mask * 255).astype(np.uint8)
        obj_mask = (obj_mask * 255).astype(np.uint8)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        rgb_pred = torch.moveaxis(rgb_pred, -1, 0)[None, ...]
        rgb_gt = torch.moveaxis(rgb_gt, -1, 0)[None, ...]
        psnr = self.psnr(rgb_pred, rgb_gt)
        ssim = self.ssim(rgb_pred, rgb_gt)
        lpips = self.lpips(rgb_pred, rgb_gt)

        # normal_l1 = torch.abs(normal_gt - normal_pred).sum(dim=-1).mean()
        normal_cos = torch.sum(normal_pred * normal_gt, dim=-1).mean()
        depth_l1 = torch.abs(depth_gt - depth_pred_aligned).mean()

        cv2.imwrite(f"{prefix}/rgb/{idx_int:04d}.png", rgb[:, :, ::-1])
        cv2.imwrite(f"{prefix}/rgb_fg/{idx_int:04d}_fg.png", fg_rgb[:, :, ::-1])
        cv2.imwrite(f"{prefix}/rgb_obj/{idx_int:04d}_obj.png", obj_rgb[:, :, ::-1])
        cv2.imwrite(f"{prefix}/rgb_bg/{idx_int:04d}_bg.png", bg_rgb[:, :, ::-1])
        cv2.imwrite(f"{prefix}/fg_mask/{idx_int:04d}_mask.png", fg_mask[:, :, ::-1])
        cv2.imwrite(f"{prefix}/obj_mask/{idx_int:04d}_mask.png", obj_mask[:, :, ::-1])
        cv2.imwrite(f"{prefix}/normal/{idx_int:04d}.png", normal[:, :, ::-1])
        cv2.imwrite(f"{prefix}/normal_fg/{idx_int:04d}_fg.png", fg_normal[:, :, ::-1])
        cv2.imwrite(f"{prefix}/normal_obj/{idx_int:04d}_obj.png", obj_normal[:, :, ::-1])
        cv2.imwrite(f"{prefix}/normal_bg/{idx_int:04d}_bg.png", bg_normal[:, :, ::-1])
        cv2.imwrite(f"{prefix}/depth/{idx_int:04d}.png", depth[:, :, ::-1])
        cv2.imwrite(f"{prefix}/depth_fg/{idx_int:04d}_fg.png", fg_depth[:, :, ::-1])
        cv2.imwrite(f"{prefix}/depth_obj/{idx_int:04d}_obj.png", obj_depth[:, :, ::-1])
        cv2.imwrite(f"{prefix}/depth_bg/{idx_int:04d}_bg.png", bg_depth[:, :, ::-1])

        return {
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips,
            # "normal_l1": normal_l1,
            "normal_cos": normal_cos,
            "depth_l1": depth_l1,
        }
