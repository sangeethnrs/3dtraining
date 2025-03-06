from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union

class DatasetTemplate(ABC):
    """
    Abstract base class for all datasets.
    """
    def __init__(self, dataset_cfg, class_names, training=True):
        self.dataset_cfg = dataset_cfg
        self.class_names = class_names
        self.training = training
        self.point_cloud_range = np.array(dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        
        self.root_path = Path(dataset_cfg.DATA_PATH)
        if not self.root_path.exists():
            raise FileNotFoundError(f'Dataset path {self.root_path} does not exist!')

    @abstractmethod
    def get_lidar(self, idx: int) -> np.ndarray:
        """Get point cloud data for given index."""
        pass

    @abstractmethod
    def get_annotations(self, idx: int) -> Dict:
        """Get annotations for given index."""
        pass

    @abstractmethod
    def get_calib(self, idx: int) -> Dict:
        """Get calibration data for given index."""
        pass

    @abstractmethod
    def prepare_data(self, data_dict: Dict) -> Dict:
        """Prepare data dictionary for network."""
        pass
