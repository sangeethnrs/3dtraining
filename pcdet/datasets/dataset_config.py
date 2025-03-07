from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from pathlib import Path

@dataclass
class DatasetConfig:
    """Unified configuration for all datasets"""
    name: str
    data_path: Path
    point_cloud_range: List[float]
    class_names: List[str]
    point_feature_encoding: Dict[str, List[str]]
    target_assigner: Dict
    aug_config: Optional[Dict] = None
    
    # Dataset specific configurations
    specific_config: Optional[Dict] = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'DatasetConfig':
        """Create config from dictionary"""
        return cls(**config_dict)
