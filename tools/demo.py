import argparse
import glob
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.models.feature_extractors import get_feature_extractor


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict



def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--output_dir', type=str, default='demo_output',
                        help='specify the directory to save prediction results')
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg


def get_class_probabilities(pred_dict, num_classes):
    """
    Extract class probabilities from model predictions.
    """
    if 'cls_preds' in pred_dict:
        # Get raw logits and apply softmax to get probabilities
        cls_logits = pred_dict['cls_preds']  # Shape: (N, num_classes)
        cls_probs = F.softmax(cls_logits, dim=-1)  # Convert to probabilities
        return cls_probs
    elif 'pred_scores' in pred_dict:
        # If only pred_scores available, create probability distribution
        scores = pred_dict['pred_scores']
        labels = pred_dict['pred_labels']
        num_preds = len(scores)
        
        # Initialize probability tensor
        cls_probs = torch.zeros((num_preds, num_classes))
        
        # Set the predicted class probabilities to the prediction scores
        for i, (score, label) in enumerate(zip(scores, labels)):
            cls_probs[i] = torch.tensor([0.01] * num_classes)  # Base probability for all classes
            label_idx = int(label) - 1  # Convert to 0-based index
            if 0 <= label_idx < num_classes:
                cls_probs[i, label_idx] = float(score)
        
        # Normalize probabilities
        cls_probs = cls_probs / cls_probs.sum(dim=1, keepdim=True)
        return cls_probs
    
    return None

def get_file_name_without_extension(file_path):
    """Extract file name without extension from the full path"""
    return Path(file_path).stem


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet with Feature Extraction-------------------------')
    
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    
    # Create feature extractor
    feature_extractor = get_feature_extractor(model)
    
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Processing sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            
            try:
                # Extract features
                pred_dicts, recall_dict, features = feature_extractor.extract_features(data_dict)
                
                # Save predictions with probabilities and feature embeddings
                output_file = save_predictions_with_probabilities(
                    pred_dicts, cfg.CLASS_NAMES, args.output_dir, idx, 
                    demo_dataset.sample_file_list[idx], features
                )
                logger.info(f'Predictions and probabilities saved to: {output_file}')
                
            except Exception as e:
                logger.error(f'Error processing sample {idx}: {str(e)}')
                import traceback
                logger.error(traceback.format_exc())
                continue

    logger.info('Demo done.')

if __name__ == '__main__':
    main()

