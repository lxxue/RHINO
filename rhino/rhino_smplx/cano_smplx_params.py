import math
from pathlib import Path

import numpy as np
import torch as th

# SMPL-X related
cano_smplx_pose = np.zeros(69, dtype=np.float32)
cano_smplx_pose[3 + 3 * 1 + 2] = math.radians(30)  # 7.5 degrees of each leg to the middle
cano_smplx_pose[3 + 3 * 2 + 2] = math.radians(-30)  # 7.5 degrees of each leg to the middle
cano_smplx_pose[63] = math.radians(-90)  # palm facing forward
cano_smplx_pose[66] = math.radians(-90)  # palm facing forward
cano_smplx_pose = th.from_numpy(cano_smplx_pose)
cano_smplx_transl = cano_smplx_pose[:3]
cano_smplx_global_orient = cano_smplx_pose[3:6]
cano_smplx_body_pose = cano_smplx_pose[6:69]

pose_mean_hand = np.load(Path(__file__).parents[1] / "checkpoints" / "pose_mean_hand.npy").astype(
    np.float32
)
cano_smplx_left_hand_flat = th.from_numpy(
    -pose_mean_hand[:45]
)  # minus - to compensate the flat mean hand to get flat hand
cano_smplx_right_hand_flat = th.from_numpy(
    -pose_mean_hand[45:]
)  # minus - to compensate the flat mean hand to get flat hand
