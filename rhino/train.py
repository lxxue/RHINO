import glob
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning import callbacks, loggers
from pytorch_lightning import Callback
from pytorch_lightning.plugins.environments import SLURMEnvironment

from rhino.datasets import create_dataloaders
from rhino.utils.misc import TQDMProgressBarNoVNum, get_class, suppress_warning


# https://github.com/Lightning-AI/lightning/issues/6389#issuecomment-1377759948
class DisabledSLURMEnvironment(SLURMEnvironment):
    @staticmethod
    def detect() -> bool:
        return False

    @staticmethod
    def _validate_srun_used() -> None:
        return

    @staticmethod
    def _validate_srun_variables() -> None:
        return


def config_torch():
    pl.seed_everything(42)
    # torch.set_default_tensor_type(torch.FloatTensor)
    # torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float32)
    # # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True
    # # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.benchmark = True
    # torch.autograd.set_detect_anomaly(True)
    suppress_warning("trimesh")


def config_hydra(cfg):
    cfg.dataset.train.data_dir = Path(cfg.data_root) / cfg.dataset.train.data_dir
    # these two values are modified accordingly
    # cfg.dataset.val.data_dir = cfg.data_root + cfg.dataset.val.data_dir
    # cfg.dataset.test.data_dir = cfg.data_root + cfg.dataset.test.data_dir
    return


def resolve_checkpoint_path(explicit_path, checkpoint_dir=Path("checkpoints")):
    """Resolve an explicit or latest checkpoint with an actionable failure."""
    if explicit_path is not None:
        return explicit_path
    checkpoint_files = sorted(checkpoint_dir.glob("*.ckpt"))
    if not checkpoint_files:
        raise FileNotFoundError(
            f"No checkpoint was provided and no *.ckpt files exist in {checkpoint_dir}"
        )
    return str(checkpoint_files[-1])


@hydra.main(version_base="1.2", config_path="confs", config_name="base")
def main(cfg):
    if cfg.version is None:
        raise ValueError("cfg.version is not set; pass version=<run_name> on the command line")
    config_torch()
    config_hydra(cfg)

    OmegaConf.save(cfg, "config.yaml")

    train_dloader, val_dloader, test_dloader = create_dataloaders(cfg.dataset)
    model = get_class(cfg.model.cls)(cfg)

    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch:04d}-{step:06d}",
        auto_insert_metric_name=False,
        # save_on_train_epoch_end=True,
        every_n_epochs=1,
        save_top_k=1,
    )
    progress_bar_callback = TQDMProgressBarNoVNum(refresh_rate=10)
    dataset_callback = DatasetSampleRatioCallback(change_epoch=10)
    if cfg.wandb:
        name = f"{cfg.dataset.train.subject}-{cfg.model.name}-{cfg.version}"
        logger = loggers.WandbLogger(project="rhino", name=name, version=name)
    else:
        logger = loggers.TensorBoardLogger(save_dir="tblogger/", name="", version="")
    # 2 iters/s so validation ~ every 30 minutes
    val_freq = max(1, cfg.model.val_steps // len(train_dloader))
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        callbacks=[checkpoint_callback, progress_bar_callback, dataset_callback],
        max_steps=cfg.model.max_steps,
        check_val_every_n_epoch=val_freq,
        logger=logger,
        log_every_n_steps=10,
        # gradient_clip_val=0.01,
        inference_mode=False,
        plugins=[DisabledSLURMEnvironment(auto_requeue=False)],
        num_sanity_val_steps=0,
    )

    if cfg.model.is_continue or cfg.test_only:
        ckpt_path = resolve_checkpoint_path(cfg.ckpt_path)
    else:
        ckpt_path = None

    if cfg.test_only:
        trainer.test(model=model, dataloaders=test_dloader, ckpt_path=ckpt_path)
    else:
        trainer.fit(
            model=model,
            train_dataloaders=train_dloader,
            val_dataloaders=val_dloader,
            ckpt_path=ckpt_path,
        )
        ckpt_files = sorted(glob.glob("checkpoints/*.ckpt"))
        if ckpt_files:
            ckpt_path = ckpt_files[-1]
            trainer.test(dataloaders=test_dloader, ckpt_path=ckpt_path)
        else:
            print("No checkpoint saved during training (max_steps too short?). Skipping test.")

    return


class DatasetSampleRatioCallback(Callback):
    def __init__(self, change_epoch: int):
        super().__init__()
        self.change_epoch = change_epoch
        self.changed = False

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.current_epoch == self.change_epoch and not self.changed:
            try:
                dataset = trainer.train_dataloader.dataset
                if hasattr(dataset, "fg_ratio") and hasattr(dataset, "obj_ratio"):
                    print(f"Epoch {trainer.current_epoch}: change sampling ratio.")
                    dataset.fg_ratio.value = 0.5
                    dataset.obj_ratio.value = 0.3
                    dataset.hand_ratio.value = 0.1
                    print(f"fg_ratio: {dataset.fg_ratio.value}, obj_ratio: {dataset.obj_ratio.value}")
                    self.changed = True
                else:
                    print("Warning: Dataset does not have 'fg_ratio' attribute.")
            except Exception as e:
                print(f"Error accessing/modifying dataset attribute: {e}")


if __name__ == "__main__":
    main()
