import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union

class FeatureExtractor:
    """Handles feature extraction from OpenPCDet models."""
    
    def __init__(self, model: nn.Module):
        """
        Initialize feature extractor.
        Args:
            model: OpenPCDet model instance
        """
        self.model = model
        
    def extract_features(self, data_dict: dict) -> Tuple[dict, dict, Optional[torch.Tensor]]:
        """
        Extract features from model.
        Args:
            data_dict: Input data dictionary
        Returns:
            Tuple containing:
            - Prediction dictionaries
            - Recall dictionaries
            - Feature embeddings tensor
        """
        # Forward pass with feature extraction
        with torch.no_grad():
            pred_dicts, recall_dict, features = self.model.forward(data_dict)
        
        return pred_dicts, recall_dict, features
