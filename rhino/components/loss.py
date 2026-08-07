import torch
from torch import nn
from torch.autograd import Variable
from math import exp
import torch.nn.functional as F


r"""Implements Stochastic Structural SIMilarity(S3IM) algorithm.
It is proposed in the ICCV2023 paper  
`S3IM: Stochastic Structural SIMilarity and Its Unreasonable Effectiveness for Neural Fields`.
Arguments:
    s3im_kernel_size (int): kernel size in ssim's convolution(default: 4)
    s3im_stride (int): stride in ssim's convolution(default: 4)
    s3im_repeat_time (int): repeat time in re-shuffle virtual patch(default: 10)
    s3im_patch_height (height): height of virtual patch(default: 64)
"""


class S3IM(torch.nn.Module):
    def __init__(
        self,
        s3im_kernel_size=4,
        s3im_stride=4,
        s3im_repeat_time=10,
        s3im_patch_height=64,
        size_average=True,
    ):
        super(S3IM, self).__init__()
        self.s3im_kernel_size = s3im_kernel_size
        self.s3im_stride = s3im_stride
        self.s3im_repeat_time = s3im_repeat_time
        self.s3im_patch_height = s3im_patch_height
        self.size_average = size_average
        self.channel = 1
        self.s3im_kernel = self.create_kernel(s3im_kernel_size, self.channel)

    def gaussian(self, s3im_kernel_size, sigma):
        gauss = torch.Tensor(
            [
                exp(-((x - s3im_kernel_size // 2) ** 2) / float(2 * sigma**2))
                for x in range(s3im_kernel_size)
            ]
        )
        return gauss / gauss.sum()

    def create_kernel(self, s3im_kernel_size, channel):
        _1D_window = self.gaussian(s3im_kernel_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        s3im_kernel = Variable(
            _2D_window.expand(channel, 1, s3im_kernel_size, s3im_kernel_size).contiguous()
        )
        return s3im_kernel

    def _ssim(
        self,
        img1,
        img2,
        s3im_kernel,
        s3im_kernel_size,
        channel,
        size_average=True,
        s3im_stride=None,
    ):
        mu1 = F.conv2d(
            img1,
            s3im_kernel,
            padding=(s3im_kernel_size - 1) // 2,
            groups=channel,
            stride=s3im_stride,
        )
        mu2 = F.conv2d(
            img2,
            s3im_kernel,
            padding=(s3im_kernel_size - 1) // 2,
            groups=channel,
            stride=s3im_stride,
        )

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(
                img1 * img1,
                s3im_kernel,
                padding=(s3im_kernel_size - 1) // 2,
                groups=channel,
                stride=s3im_stride,
            )
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(
                img2 * img2,
                s3im_kernel,
                padding=(s3im_kernel_size - 1) // 2,
                groups=channel,
                stride=s3im_stride,
            )
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(
                img1 * img2,
                s3im_kernel,
                padding=(s3im_kernel_size - 1) // 2,
                groups=channel,
                stride=s3im_stride,
            )
            - mu1_mu2
        )

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def ssim_loss(self, img1, img2):
        """
        img1, img2: torch.Tensor([b,c,h,w])
        """
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.s3im_kernel.data.type() == img1.data.type():
            s3im_kernel = self.s3im_kernel
        else:
            s3im_kernel = self.create_kernel(self.s3im_kernel_size, channel)

            if img1.is_cuda:
                s3im_kernel = s3im_kernel.cuda(img1.get_device())
            s3im_kernel = s3im_kernel.type_as(img1)

            self.s3im_kernel = s3im_kernel
            self.channel = channel

        return self._ssim(
            img1,
            img2,
            s3im_kernel,
            self.s3im_kernel_size,
            channel,
            self.size_average,
            s3im_stride=self.s3im_stride,
        )

    def forward(self, src_vec, tar_vec):
        src_vec = src_vec[0]
        tar_vec = tar_vec[0]
        loss = 0.0
        index_list = []
        for i in range(self.s3im_repeat_time):
            if i == 0:
                tmp_index = torch.arange(len(tar_vec))
                index_list.append(tmp_index)
            else:
                ran_idx = torch.randperm(len(tar_vec))
                index_list.append(ran_idx)
        res_index = torch.cat(index_list)
        tar_all = tar_vec[res_index]
        src_all = src_vec[res_index]
        tar_patch = tar_all.permute(1, 0).reshape(1, 3, self.s3im_patch_height, -1)
        src_patch = src_all.permute(1, 0).reshape(1, 3, self.s3im_patch_height, -1)
        loss = 1 - self.ssim_loss(src_patch, tar_patch)
        return loss


# source: neuralsim
# the gradient might also flow from depth loss to the computed scale and shift
# and then to the prediction
def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01

    if det.numel() > 1:
        valid = det.nonzero()
        x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
        x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]
    elif det.item() != 0:
        x_0 = (a_11 * b_0 - a_01 * b_1) / det
        x_1 = (-a_01 * b_0 + a_00 * b_1) / det

    # print(x_0, x_1)

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return torch.zeros((), device=M.device)
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):
    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):
    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction="batch-based"):
        super().__init__()

        if reduction == "batch-based":
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction="batch-based"):
        super().__init__()

        if reduction == "batch-based":
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(
                prediction[:, ::step, ::step],
                target[:, ::step, ::step],
                mask[:, ::step, ::step],
                reduction=self.__reduction,
            )

        return total


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction="batch-based"):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):
        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)
