import os
import numpy as np

import torch
import torch.utils.data
import torch.distributed
import torch.multiprocessing as mp

from omegaconf import OmegaConf

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from libs.datasets import get_dataset
from model import get_model


def cam_worker(rank, world_size, cfg, split):
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    dataset_split = {
        'train': cfg.dataset_cls.split.train,
        'val': cfg.dataset_cls.split.val,
        'test': cfg.dataset_cls.split.test
    }
    dataset = get_dataset(cfg.dataset_cls.name)(
        root=cfg.dataset_cls.root,
        data_dir=cfg.exp.data_dir,
        split=dataset_split[split],
        mean_rgb=(cfg.image.mean.R, cfg.image.mean.G, cfg.image.mean.B),
        std_rgb=(cfg.image.std.R, cfg.image.std.G, cfg.image.std.B),
        target_size=cfg.image.size.test,
        augmentation=True,
        fg_path=cfg.dataset_cls.fg_path,
    )

    sampler = DistributedSampler(dataset, shuffle=False, rank=rank, num_replicas=world_size)
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.solver.batch_size.test,
        num_workers=cfg.dataloader.num_workers,
        shuffle=False,  # no use
        pin_memory=False,
        drop_last=False,
        sampler=sampler,
        prefetch_factor=4
    )

    model = get_model(
        backbone=cfg.model.backbone,
        num_classes=cfg.dataset_cls.n_classes,
        pretrained=cfg.model.pretrained,
        init_momentum=cfg.model.init_momentum,
    )
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    model.eval()
    for i,(image_id, image, label) in enumerate(loader):
        with torch.no_grad():
            raise NotImplementedError()
