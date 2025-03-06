import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict

from .dataset_factory import DatasetFactory
from .processor.data_processor import DataProcessor

class UnifiedDataLoader:
    """
    Unified data loader that handles multiple datasets.
    """
    def __init__(
        self,
        dataset_cfg,
        class_names,
        batch_size: int,
        dist: bool = False,
        training: bool = True,
        workers: int = 4,
        seed: Optional[int] = None
    ):
        self.dataset_cfg = dataset_cfg
        self.batch_size = batch_size
        self.training = training
        self.workers = workers

        if seed is not None:
            torch.manual_seed(seed)

        # Create dataset instance
        self.dataset = DatasetFactory.create_dataset(
            dataset_name=dataset_cfg.DATASET,
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=training
        )

        # Initialize data processor
        self.data_processor = DataProcessor(
            processor_configs=dataset_cfg.DATA_PROCESSOR,
            point_cloud_range=dataset_cfg.POINT_CLOUD_RANGE,
            training=training
        )

        # Create PyTorch DataLoader
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=True,
            shuffle=training and not dist,
            collate_fn=self.collate_batch,
            drop_last=False
        )
        self.iter = iter(self.dataloader)

    def collate_batch(self, batch_list):
        """
        Collate batch of data dictionaries.
        
        Args:
            batch_list: List of data dictionaries
            
        Returns:
            Collated batch dictionary
        """
        data_dict = {}
        for key in batch_list[0].keys():
            if key in ['points', 'voxels', 'num_points', 'coordinates']:
                data_dict[key] = []
            elif key in ['gt_boxes']:
                max_gt = max([len(b[key]) for b in batch_list])
                batch_gt_boxes = []
                for batch_dict in batch_list:
                    batch_boxes = batch_dict[key]
                    if len(batch_boxes) < max_gt:
                        pad_boxes = torch.zeros((max_gt - len(batch_boxes), batch_boxes.shape[-1]))
                        batch_boxes = torch.cat([batch_boxes, pad_boxes], dim=0)
                    batch_gt_boxes.append(batch_boxes)
                data_dict[key] = torch.stack(batch_gt_boxes, dim=0)
            else:
                data_dict[key] = [b[key] for b in batch_list]

        if 'points' in data_dict:
            data_dict['points'] = torch.cat(data_dict['points'], dim=0)
        if 'voxels' in data_dict:
            data_dict['voxels'] = torch.cat(data_dict['voxels'], dim=0)
        
        return data_dict

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        self.iter = iter(self.dataloader)
        return self

    def __next__(self):
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.dataloader)
            raise StopIteration
