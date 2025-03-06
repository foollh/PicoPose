import os
import sys
import importlib
import argparse
import logging
import numpy as np
from torch import optim
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'provider'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'model'))

from utils.lr_scheduler import WarmupCosineLR
from utils.lite import Lite
from utils.loss_utils  import Loss
from utils.logging import MyPrintingCallback
from utils.torch_utils import set_seed

def get_parser():
    parser = argparse.ArgumentParser(
        description="Pose Estimation")

    parser.add_argument("--model",
                        type=str,
                        default="picopose",
                        help="name of training model")
    parser.add_argument("--config",
                        type=str,
                        default="config/base.yaml",
                        help="path to config file")
    parser.add_argument("--version_id",
                        type=int,
                        default=0,
                        help="experiment id")
    parser.add_argument("--ckpt_path",
                        type=str,
                        default='none',
                        help="iter num. of checkpoint")
    args_cfg = parser.parse_args()

    return args_cfg

def run_train(cfg):
    # seed
    set_seed(cfg.trainer.rd_seed)

    # train dataloader
    max_iters, num_epoch, batch_size = cfg.lr_scheduler.max_iters, cfg.trainer.training_epoch, cfg.train_dataloader.bs
    iters_per_epoch = int((np.floor(max_iters / num_epoch))*len(cfg.trainer.devices))

    train_dataset = importlib.import_module(cfg.train_dataset.name)
    train_dataset = train_dataset.Dataset(cfg.train_dataset, iters_per_epoch*batch_size)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train_dataloader.bs,
        num_workers=cfg.train_dataloader.num_workers,
        shuffle=cfg.train_dataloader.shuffle,
        sampler=None,
        drop_last=cfg.train_dataloader.drop_last,
        pin_memory=cfg.train_dataloader.pin_memory,
    )
    dataloaders = {
        "train": train_dataloader,
    }

    # network
    MODEL = importlib.import_module(cfg.model_name)
    model = MODEL.Net(cfg.model)

    # optimizer
    if cfg.optimizer.type == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=cfg.optimizer.lr, betas=cfg.optimizer.betas, eps=cfg.optimizer.eps, weight_decay=cfg.optimizer.weight_decay)
    elif cfg.optimizer.type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.optimizer.lr, betas=cfg.optimizer.betas, eps=cfg.optimizer.eps, weight_decay=cfg.optimizer.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=cfg.optimizer.lr, betas=cfg.optimizer.betas, eps=cfg.optimizer.eps, weight_decay=cfg.optimizer.weight_decay)

    # learning rate
    lr_scheduler = WarmupCosineLR(optimizer, max_iters=max_iters, warmup_factor=cfg.lr_scheduler.warmup_factor, warmup_iters=cfg.lr_scheduler.warmup_iters)
    lr_scheduler_config = {
        "scheduler": lr_scheduler,
        "interval": "step",
        "frequency": 1,
    }
    
    # logger
    log_every_n_steps = cfg.trainer.iters_to_print
    logger = [
        loggers.TensorBoardLogger(save_dir='log', name=cfg.model_name, version=cfg.version_id),
    ]
    # callbacks
    callbacks = [
        MyPrintingCallback(save_dir='log', name=cfg.model_name, version=cfg.version_id, log_every_n_steps=log_every_n_steps),
        ModelCheckpoint(save_top_k=-1)
    ]
    
    if cfg.ckpt_path == 'none':
        ckpt_path = None
    else:
        ckpt_path = cfg.ckpt_path
    # Trainer 
    trainer = pl.Trainer(
        logger,
        callbacks=callbacks,
        strategy=cfg.trainer.strategy, 
        accelerator=cfg.trainer.accelerator, 
        devices=cfg.trainer.devices, 
        log_every_n_steps=log_every_n_steps, 
        max_epochs=num_epoch, 
        enable_progress_bar=True,
    )

    # starting training
    trainer.fit(
        Lite(
            network=model,
            loss=Loss(),
            optimizer=optimizer,
            lr_scheduler_config=lr_scheduler_config,
            dataloaders=dataloaders,
        ),
        ckpt_path=ckpt_path
    )
    logging.info(f"---" * 20)


if __name__ == "__main__":
    args = get_parser()
    cfg = OmegaConf.load(args.config)
    cfg.model_name = args.model
    cfg.version_id = args.version_id
    cfg.ckpt_path = args.ckpt_path
    run_train(cfg)
