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
from typing import Dict, Optional, Tuple

from pcdet.config import cfg, cfg_from_yaml_file, log_config_to_file
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from train_utils import train_utils
from train_utils.optimization import build_optimizer, build_scheduler
from pcdet.datasets.dataset_config import DatasetConfig


class UnifiedTrainer:
    """Unified trainer class for handling training process"""
    
    def __init__(self, args: argparse.Namespace, cfg: Dict):
        self.args = args
        self.cfg = cfg
        self.start_epoch = 0
        self.it = 0
        self.rank = 0  # For single GPU training
        
        self._setup_environment()
        self._initialize_logging()
        self._build_components()

    def _setup_environment(self):
        """Setup training environment"""
        if self.args.fix_random_seed:
            common_utils.set_random_seed(666)

        # Create output directories
        self.root_dir = Path(self.cfg.ROOT_DIR)
        self.output_dir = self.root_dir / 'output' / self.cfg.EXP_GROUP_PATH / self.cfg.TAG / self.args.extra_tag
        self.ckpt_dir = self.output_dir / 'ckpt'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_logging(self):
        """Initialize logging and monitoring tools"""
        # Create logger
        log_file = self.output_dir / f'log_train_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.txt'
        self.logger = common_utils.create_logger(log_file, rank=self.rank)
        log_config_to_file(self.cfg, logger=self.logger)

        # Initialize wandb
        self._init_wandb()

        # Create tensorboard writer
        self.tb_log = SummaryWriter(log_dir=str(self.output_dir / 'tensorboard'))
        
        self.logger.info(f'**********************Start training {self.cfg.EXP_GROUP_PATH}({self.args.extra_tag})**********************')

    def _init_wandb(self):
        """Initialize Weights & Biases logging"""
        current_time = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        wandb.init(
            project="OpenPCDet",
            name=f"{self.cfg.TAG}_{self.args.extra_tag}",
            config={
                "config_file": self.args.cfg_file,
                "batch_size": self.cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU,
                "epochs": self.cfg.OPTIMIZATION.NUM_EPOCHS,
                "learning_rate": self.cfg.OPTIMIZATION.LR,
                "model": self.cfg.MODEL.NAME,
                "username": "sangeethnrs",
                "start_time": current_time,
                "dataset": self.cfg.DATA_CONFIG.DATASET,
                "point_cloud_range": self.cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
                "voxel_size": self.cfg.DATA_CONFIG.get('VOXEL_SIZE', None)
            }
        )

    def _build_components(self):
        """Build all necessary components for training"""
        self._build_dataloader()
        self._build_model()
        self._build_optimizer()
        self._load_checkpoint()
        self._build_scheduler()

    def _build_dataloader(self):
        """Build data loader"""
        dataset_cfg = DatasetConfig.from_dict(self.cfg.DATA_CONFIG)
        self.train_set, self.train_loader, self.train_sampler = build_dataloader(
            dataset_cfg=dataset_cfg,
            class_names=self.cfg.CLASS_NAMES,
            batch_size=self.cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU,
            dist=False,
            workers=self.args.workers,
            logger=self.logger,
            training=True
        )

    def _build_model(self):
        """Build network model"""
        self.model = build_network(
            model_cfg=self.cfg.MODEL,
            num_class=len(self.cfg.CLASS_NAMES),
            dataset=self.train_set
        )
        self.model.cuda()

    def _build_optimizer(self):
        """Build optimizer"""
        self.optimizer = build_optimizer(self.model, self.cfg.OPTIMIZATION)

    def _load_checkpoint(self):
        """Load checkpoint if specified"""
        if self.args.pretrained_model is not None:
            self.model.load_params_from_file(
                filename=self.args.pretrained_model,
                to_cpu=False,
                logger=self.logger
            )

        if self.args.ckpt is not None:
            self._load_ckpt(self.args.ckpt)
        else:
            self._load_latest_ckpt()

    def _load_ckpt(self, ckpt_path: str):
        """Load specific checkpoint"""
        self.it, self.start_epoch = self.model.load_params_with_optimizer(
            ckpt_path,
            to_cpu=False,
            optimizer=self.optimizer,
            logger=self.logger
        )
        self.last_epoch = self.start_epoch + 1

    def _load_latest_ckpt(self):
        """Load latest checkpoint from checkpoint directory"""
        ckpt_list = glob.glob(str(self.ckpt_dir / '*checkpoint_epoch_*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            self._load_ckpt(ckpt_list[-1])
        else:
            self.last_epoch = -1

    def _build_scheduler(self):
        """Build learning rate scheduler"""
        self.lr_scheduler, self.lr_warmup_scheduler = build_scheduler(
            self.optimizer,
            total_iters_each_epoch=len(self.train_loader),
            total_epochs=self.cfg.OPTIMIZATION.NUM_EPOCHS,
            last_epoch=self.last_epoch,
            optim_cfg=self.cfg.OPTIMIZATION
        )

    def train(self):
        """Main training loop"""
        try:
            train_utils.train_model(
                model=self.model,
                optimizer=self.optimizer,
                train_loader=self.train_loader,
                model_func=model_fn_decorator(),
                lr_scheduler=self.lr_scheduler,
                optim_cfg=self.cfg.OPTIMIZATION,
                start_epoch=self.start_epoch,
                total_epochs=self.cfg.OPTIMIZATION.NUM_EPOCHS,
                start_iter=self.it,
                rank=self.rank,
                tb_log=self.tb_log,
                ckpt_save_dir=self.ckpt_dir,
                train_sampler=self.train_sampler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                ckpt_save_interval=1,
                max_ckpt_save_num=50,
                logger=self.logger
            )
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            wandb.finish()
            raise
        finally:
            self.logger.info(
                f'**********************End training {self.cfg.EXP_GROUP_PATH}({self.args.extra_tag})**********************\n\n\n'
            )
            wandb.finish()


def parse_config() -> Tuple[argparse.Namespace, Dict]:
    """Parse command line arguments and configuration file"""
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
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])
    
    return args, cfg


def main():
    """Main entry point"""
    args, cfg = parse_config()
    trainer = UnifiedTrainer(args, cfg)
    trainer.train()


if __name__ == '__main__':
    main()
