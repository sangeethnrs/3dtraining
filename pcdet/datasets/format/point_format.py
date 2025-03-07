import numpy as np
from typing import Dict, List, Union
from enum import Enum

class PointFormat(Enum):
    XYZIR = 'xyzir'  # x,y,z,intensity,ring
    XYZI = 'xyzi'    # x,y,z,intensity
    XYZRGB = 'xyzrgb'  # x,y,z,r,g,b
    
class PointFormatConverter:
    """Convert between different point cloud formats"""
    
    @staticmethod
    def convert(points: np.ndarray, 
                source_format: PointFormat,
                target_format: PointFormat) -> np.ndarray:
        """Convert points from source format to target format"""
        if source_format == target_format:
            return points
            
        # Implement conversion logic
        conversion_map = {
            (PointFormat.XYZIR, PointFormat.XYZI): lambda x: x[:, :4],
            (PointFormat.XYZI, PointFormat.XYZIR): lambda x: np.pad(x, ((0,0), (0,1))),
            # Add more conversion methods
        }
        
        key = (source_format, target_format)
        if key not in conversion_map:
            raise ValueError(f"Unsupported conversion: {source_format} to {target_format}")
            
        return conversion_map[key](points)
