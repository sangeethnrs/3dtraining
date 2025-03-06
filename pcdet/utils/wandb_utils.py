import wandb
from pcdet.utils import common_utils

class WandbLogger:
    def __init__(self, cfg):
        """
        Initialize WandB logger for OpenPCDet
        Args:
            cfg: OpenPCDet config object
        """
        self.cfg = cfg
        
        # Initialize wandb
        wandb.init(
            project="openpcdet",  # Change this to your project name
            name=cfg.TAG if hasattr(cfg, 'TAG') else None,
            config=common_utils.cfg_to_dict(cfg),  # Log config as parameters
            dir=cfg.ROOT_DIR
        )
        
    def log_metrics(self, metrics_dict, step=None):
        """
        Log training/validation metrics
        Args:
            metrics_dict: Dictionary containing metrics
            step: Current training step
        """
        wandb.log(metrics_dict, step=step)
    
    def log_model(self, model_info):
        """
        Log model architecture and parameters
        Args:
            model_info: Dictionary containing model information
        """
        wandb.watch(model_info['model'])
    
    def log_images(self, image_dict):
        """
        Log detection images with bounding boxes
        Args:
            image_dict: Dictionary containing images and metadata
        """
        wandb.log(image_dict)
    
    def close(self):
        """
        Close wandb logging
        """
        wandb.finish()
