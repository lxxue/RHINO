import logging

import torch
from pytorch_lightning.callbacks import TQDMProgressBar
from torch.autograd import grad


def get_class(kls):
    parts = kls.split(".")
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def suppress_warning(logger_name):
    logging.getLogger(logger_name).setLevel(logging.ERROR)


def nan_hook(self, inp, output):
    if not isinstance(output, tuple):
        outputs = [output]
    else:
        outputs = output

    for i, out in enumerate(outputs):
        nan_mask = torch.isnan(out)
        if nan_mask.any():
            # print("In", self.__class__.__name__)
            raise RuntimeError(f"Found NaN in {self.__class__.__name__}")
            # raise RuntimeError(
            #     f"Found NAN in output {i} at indices: ",
            #     nan_mask.nonzero(),
            #     "where:",
            #     out[nan_mask.nonzero()[:, 0].unique(sorted=True)],
            # )


class TQDMProgressBarNoVNum(TQDMProgressBar):
    def get_metrics(self, trainer, pl_module):
        metrics = super().get_metrics(trainer, pl_module)
        metrics.pop("v_num", None)
        for name in metrics:
            metrics[name] = float(f"{metrics[name]:.3f}")
        return metrics


def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0][:, :, -3:]
    return points_grad
