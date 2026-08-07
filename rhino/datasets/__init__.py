from torch.utils.data import DataLoader

from rhino.datasets.hoi import HOIObjDataset, HOIObjTestDataset, HOIObjValDataset


def find_dataset_classes(name):
    mapping = {
        "hoi_obj": (HOIObjDataset, HOIObjValDataset, HOIObjTestDataset),
    }
    cls = mapping.get(name, None)
    if cls is None:
        raise ValueError(f"Fail to find dataset {name}")
    return cls


def create_dataloaders(cfg):
    train_cls, val_cls, test_cls = find_dataset_classes(cfg.name)
    train_dloader = create_dataloader(cfg.train, train_cls)
    val_dloader = create_dataloader(cfg.val, val_cls)
    test_dloader = create_dataloader(cfg.test, test_cls)
    return train_dloader, val_dloader, test_dloader


def create_dataloader(cfg, dset_cls):
    if cfg.batch_size != 1:
        raise ValueError(
            f"RHINO requires batch_size=1, got {cfg.batch_size} for {dset_cls.__name__}. "
            "Temporal neighbors, contact labels, and frame-conditioned embeddings are "
            "defined for one sequence frame per batch."
        )
    dset = dset_cls(cfg)
    dloader = DataLoader(
        dset,
        batch_size=cfg.batch_size,
        drop_last=cfg.drop_last,
        shuffle=cfg.shuffle,
        num_workers=cfg.worker,
        pin_memory=True,
        persistent_workers=cfg.worker > 0,
    )
    return dloader
