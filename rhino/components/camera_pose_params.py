# https://github.com/JanaldoChen/Anim-NeRF
import torch.nn as nn
import pytorch3d


class CameraPoseParams(nn.Module):
    def __init__(self, num_frames):
        super().__init__()
        self.num_frames = num_frames
        self.se3_log_reps = nn.Embedding(num_frames, 6)
        # self.se3_log_reps = nn.Embedding(num_frames, 16)
        # self.axis_angles = nn.Embedding(num_frames, 3)
        # self.trans = nn.Embedding(num_frames, 3)
        # self.param.weight.data.fill_(0)
        # self.param.weight.requires_grad = False

    # def init_parameters(self, param_name, data, requires_grad=False):
    #     getattr(self, param_name).weight.data = data[..., : self.params_dim[param_name]]
    #     getattr(self, param_name).weight.requires_grad = requires_grad

    # def set_requires_grad(self, param_name, requires_grad=True):
    #     getattr(self, param_name).weight.requires_grad = requires_grad

    def forward(self, frame_ids):
        log_rep = self.se3_log_reps(frame_ids)
        # if frame_ids.item() == 0:
        #     print("log_rep", log_rep)
        # axis_angle = self.axis_angles(frame_ids)
        # trans = self.trans(frame_ids)
        # rot_mat = pytorch3d.transforms.axis_angle_to_matrix(axis_angle)
        # se3 exp returns transposed matrix: https://github.com/facebookresearch/pytorch3d/issues/1488
        pose = pytorch3d.transforms.se3_exp_map(log_rep).transpose(-2, -1)
        # pose = log_rep.view(-1, 4, 4)
        return pose
