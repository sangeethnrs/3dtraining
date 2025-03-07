from typing import Dict, List, Optional
import numpy as np

class UnifiedAugmentor:
    """Unified augmentation pipeline for all datasets"""
    
    def __init__(self, aug_config: Dict):
        self.aug_config = aug_config
        self.augmentors = self._build_augmentors()
        
    def _build_augmentors(self):
        """Build augmentation pipeline from config"""
        augmentors = []
        if self.aug_config.get('random_flip', False):
            augmentors.append(self.random_flip)
        if self.aug_config.get('random_rotation', False):
            augmentors.append(self.random_rotation)
        # Add more augmentations
        return augmentors
        
    def augment(self, data_dict: Dict) -> Dict:
        """Apply augmentation pipeline"""
        for augmentor in self.augmentors:
            data_dict = augmentor(data_dict)
        return data_dict
        
    def random_flip(self, data_dict: Dict) -> Dict:
        """Random flip augmentation"""
        if np.random.rand() < 0.5:
            points = data_dict['points']
            points[:, 0] = -points[:, 0]
            if 'gt_boxes' in data_dict:
                data_dict['gt_boxes'][:, 0] = -data_dict['gt_boxes'][:, 0]
                data_dict['gt_boxes'][:, 6] = -data_dict['gt_boxes'][:, 6]
        return data_dict
        
    # Implement more augmentation methods
