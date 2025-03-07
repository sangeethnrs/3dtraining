import torch.nn as nn
from typing import Dict

class UnifiedModel(nn.Module):
    """Base class for all 3D detection models"""
    
    def __init__(self, model_cfg: Dict):
        super().__init__()
        self.model_cfg = model_cfg
        
    def forward(self, batch_dict):
        """Forward pass"""
        raise NotImplementedError
        
    def post_process(self, batch_dict):
        """Post processing"""
        raise NotImplementedError
        
    @staticmethod
    def build_losses(loss_cfg):
        """Build loss functions"""
        raise NotImplementedError
