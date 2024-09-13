import sys
import os
import time
import numpy as np

import torch
import torch.utils.data
import torch.distributed
import torch.multiprocessing as mp

from omegaconf import OmegaConf

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from libs.datasets import get_dataset
from model import get_model
import libs.utils as my_utils
from model.loss_functions import AdaptiveSpatialBCELoss, MaskConsistencyLoss, ForegroundRatioLoss


def train_worker(rank, world_size, cfg):
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    if rank == 0:
        log_dir = os.path.join(cfg.exp.data_dir, 'logs', cfg.exp.id)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

    train_dataset = get_dataset(cfg.dataset_wsss.name)(
        root=cfg.dataset_wsss.root,
        data_dir=cfg.exp.data_dir,
        split=cfg.dataset_wsss.split.train,
        mean_rgb=(cfg.image.mean.R, cfg.image.mean.G, cfg.image.mean.B),
        std_rgb=(cfg.image.std.R, cfg.image.std.G, cfg.image.std.B),
        target_size=cfg.image.size.train,
        augmentation=True,
        fg_path=cfg.dataset_wsss.fg_path,
    )

    train_sampler = DistributedSampler(
        train_dataset,
        shuffle=True,
        rank=rank,
        num_replicas=world_size
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.solver.batch_size.train,
        num_workers=cfg.dataloader.num_workers,
        # shuffle=True,  # Since we use the sampler, we should set 'shuffle' in sampler
        pin_memory=True,  # CHANGE: to True
        persistent_workers=True,  # CHANGE: add
        drop_last=True,
        sampler=train_sampler,
        prefetch_factor=4
    )

    model = get_model(
        backbone=cfg.model.backbone,
        num_classes=cfg.dataset_wsss.n_classes,
        pretrained=cfg.model.pretrained,
        init_momentum=cfg.model.init_momentum,
    )
    param_groups = model.get_param_groups()
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = getattr(my_utils, cfg.solver.optimizer)(
        params=[
            {
                "params": param_groups[0],
                "lr": cfg.solver.lr,
                "weight_decay": cfg.solver.weight_decay,
            },
            {
                "params": param_groups[1],
                "lr": cfg.solver.lr,
                "weight_decay": cfg.solver.weight_decay,
            },
            {
                "params": param_groups[2],
                "lr": cfg.solver.lr * 10,
                "weight_decay": cfg.solver.weight_decay,
            },
        ],
        lr=cfg.solver.lr,
        weight_decay=cfg.solver.weight_decay,
        betas=tuple(cfg.solver.adam_betas),
        warmup_iter=cfg.solver.iter_warmup,
        max_iter=cfg.solver.iter_max,
        warmup_ratio=cfg.solver.lr_warmup,
        power=cfg.solver.poly_power
    )

    spatial_loss_function = AdaptiveSpatialBCELoss()
    mask_loss_function = MaskConsistencyLoss()
    fg_loss_function = ForegroundRatioLoss()

    epoch = 0
    running_spatial_loss = 0.0
    running_mask_loss = 0.0
    running_fg_loss = 0.0
    running_loss = 0.0

    train_sampler.set_epoch(epoch)
    train_loader_iter = iter(train_loader)

    model.train()
    time_start_train = time.perf_counter()
    for n_iter in range(cfg.solver.iter_max):
        optimizer.zero_grad()

        try:
            image_id, image, label, mask, fg = next(train_loader_iter)
        except:
            epoch += 1
            train_sampler.set_epoch(epoch)
            train_loader_iter = iter(train_loader)
            image_id, image, label, mask, fg = next(train_loader_iter)

        image, label, mask, fg = image.to(rank), label.to(rank), mask.to(rank), fg.to(rank)
        cams = model(image)
        spatial_loss = spatial_loss_function(cams, label)
        mask_loss = mask_loss_function(cams, mask)
        fg_loss = fg_loss_function(cams, label, fg)
        loss = 1.0 * spatial_loss + 1.0 * mask_loss + 1.0 * fg_loss

        loss.backward()
        optimizer.step()

        running_spatial_loss += spatial_loss.item()
        running_mask_loss += mask_loss.item()
        running_fg_loss += fg_loss.item()
        running_loss += loss.item()

        # debug
        # print(f'Process {rank} has finished iter {n_iter}')
        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(name)

        # 打印进度
        if rank == 0 and (n_iter + 1) % cfg.logger.iter_print == 0:
            print(f'iter {n_iter + 1} of {cfg.solver.iter_max} has finished \
                in {time.perf_counter()-time_start_train} seconds')
            sys.stdout.flush()

        # 记录损失
        if rank == 0 and (n_iter + 1) % cfg.logger.iter_log == 0:
            writer.add_scalars('training_losses',
                               {'spatial_loss': running_spatial_loss / cfg.logger.iter_log,
                                'mask_loss': running_mask_loss / cfg.logger.iter_log,
                                'fg_loss': running_fg_loss / cfg.logger.iter_log,
                                'total_loss': running_loss / cfg.logger.iter_log},
                               n_iter)
            running_spatial_loss = 0.0
            running_mask_loss = 0.0
            running_fg_loss = 0.0
            running_loss = 0.0

        if rank == 0 and (n_iter + 1) % cfg.solver.iter_eval == 0:
            # 记录模型参数
            for name, param in model.named_parameters():
                writer.add_histogram(name, param, n_iter)

            # 保存模型权重
            checkpoint_dir = os.path.join(cfg.exp.data_dir, 'results', cfg.exp.id)
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f'model_checkpoint_{n_iter + 1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")


def op_train(config_path):
    cfg = OmegaConf.load(config_path)

    mp.spawn(train_worker, args=(cfg.solver.world_size, cfg), nprocs=cfg.solver.world_size, join=True)