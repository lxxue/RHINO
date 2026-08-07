# https://github.com/JanaldoChen/Anim-NeRF
import torch.nn as nn
import pytorch3d


class ObjectPoseParams(nn.Module):
    def __init__(self, num_frames):
        super().__init__()
        print("Learnable object pose parameters")
        self.num_frames = num_frames
        self.se3_log_reps = nn.Embedding(num_frames, 6)

    def forward(self, frame_ids):
        log_rep = self.se3_log_reps(frame_ids)
        # se3 exp returns transposed matrix: https://github.com/facebookresearch/pytorch3d/issues/1488
        pose = pytorch3d.transforms.se3_exp_map(log_rep).transpose(-2, -1)
        return pose
