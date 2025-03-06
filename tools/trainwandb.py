import os
import torch
import torch.nn as nn
from pathlib import Path
import argparse
import datetime
import glob
import wandb
from tensorboardX import SummaryWriter
import numpy as np

from pcdet.config import cfg, cfg_from_yaml_file, log_config_to_file
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from train_utils import train_utils
from train_utils.optimization import build_optimizer, build_scheduler

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    
    # Set paths
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'tools' and 'xxxx.yaml'
    
    return args, cfg


def main():
    args, cfg = parse_config()
    
    if args.fix_random_seed:
        common_utils.set_random_seed(666)

    # Create output directory path
    root_dir = Path(cfg.ROOT_DIR)
    if not hasattr(cfg, 'EXP_GROUP_PATH'):
        cfg.EXP_GROUP_PATH = Path(args.cfg_file).parent.name
    if not hasattr(cfg, 'TAG'):
        cfg.TAG = Path(args.cfg_file).stem
    
    # Create output directories
    output_dir = root_dir / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=0)
    log_config_to_file(cfg, logger=logger)

    # Initialize wandb
    wandb.init(
        project="OpenPCDet",
        name=f"{cfg.TAG}_{args.extra_tag}",
        config={
            "config_file": args.cfg_file,
            "batch_size": cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU,
            "epochs": cfg.OPTIMIZATION.NUM_EPOCHS,
            "learning_rate": cfg.OPTIMIZATION.LR,
            "model": cfg.MODEL.NAME,
            "username": "sangeethnrs",  # Adding username
            "start_time": "2025-02-11 20:14:20"  # Adding start time
        }
    )

    # Log to file
    logger.info('**********************Start training %s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, args.extra_tag))

    # Create tensorboard writer
    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard'))

    # Build dataloader
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU,
        dist=False,  # Set to False for single GPU
        workers=args.workers,
        logger=logger,
        training=True
    )

    # Build model
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    model.cuda()

    # Build optimizer
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # Load checkpoint if specified
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=False, logger=logger)

    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=False, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            it, start_epoch = model.load_params_with_optimizer(
                ckpt_list[-1], to_cpu=False, optimizer=optimizer, logger=logger
            )
            last_epoch = start_epoch + 1

    model.train()

    # Build learning rate scheduler
    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=cfg.OPTIMIZATION.NUM_EPOCHS,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    try:
        train_utils.train_model(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            model_func=model_fn_decorator(),
            lr_scheduler=lr_scheduler,
            optim_cfg=cfg.OPTIMIZATION,
            start_epoch=start_epoch,
            total_epochs=cfg.OPTIMIZATION.NUM_EPOCHS,
            start_iter=it,
            rank=0,  # Set to 0 for single GPU
            tb_log=tb_log,
            ckpt_save_dir=ckpt_dir,
            train_sampler=train_sampler,
            lr_warmup_scheduler=lr_warmup_scheduler,
            ckpt_save_interval=1,
            max_ckpt_save_num=50,
            logger=logger
        )

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        wandb.finish()
        raise

    logger.info('**********************End training %s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, args.extra_tag))
    wandb.finish()

if __name__ == '__main__':
    main()
