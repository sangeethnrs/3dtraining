import numpy as np
import struct
from pathlib import Path
from typing import Union


def load_pcd_file(file_path: Union[str, Path]) -> np.ndarray:
    """
    Load point cloud data from PCD file.
    
    Args:
        file_path: Path to the PCD file

    Returns:
        numpy.ndarray: Point cloud data
    """
    with open(file_path, 'rb') as f:
        # Read PCD header
        header = {}
        while True:
            line = f.readline().decode().strip()
            if line == 'DATA binary':
                break
            if line == 'DATA ascii':
                return load_pcd_ascii(file_path)
            
            if line:
                key, value = line.split(' ', 1)
                header[key] = value

        # Parse header
        width = int(header['WIDTH'])
        height = int(header['HEIGHT'])
        points = width * height
        fields = header['FIELDS'].split()
        sizes = [int(x) for x in header['SIZE'].split()]
        types = header['TYPE'].split()
        
        # Create dtype for structured array
        dtype_list = []
        for field, size, type_ in zip(fields, sizes, types):
            if type_ == 'F':
                dtype_list.append((field, 'f{}'.format(size)))
            elif type_ == 'I':
                dtype_list.append((field, 'i{}'.format(size)))
            elif type_ == 'U':
                dtype_list.append((field, 'u{}'.format(size)))

        # Read binary data
        data = np.fromfile(f, dtype=np.dtype(dtype_list))
        
        # Reshape if necessary
        if height > 1:
            data = data.reshape((height, width))

        return data


def load_pcd_ascii(file_path: Union[str, Path]) -> np.ndarray:
    """
    Load point cloud data from ASCII PCD file.
    
    Args:
        file_path: Path to the ASCII PCD file

    Returns:
        numpy.ndarray: Point cloud data
    """
    data = np.loadtxt(file_path, skiprows=11)  # Skip header
    return data


def load_bin_file(file_path: Union[str, Path]) -> np.ndarray:
    """
    Load point cloud data from binary file (KITTI format).
    
    Args:
        file_path: Path to the binary file

    Returns:
        numpy.ndarray: Point cloud data
    """
    points = np.fromfile(file_path, dtype=np.float32)
    points = points.reshape((-1, 4))  # x, y, z, intensity
    return points


def save_pcd_file(points: np.ndarray, file_path: Union[str, Path], binary: bool = True) -> None:
    """
    Save point cloud data to PCD file.
    
    Args:
        points: Point cloud data
        file_path: Output file path
        binary: Whether to save in binary format
    """
    with open(file_path, 'w') as f:
        # Write header
        f.write('# .PCD v0.7 - Point Cloud Data\n')
        f.write('VERSION 0.7\n')
        
        if points.dtype.names is None:
            fields = ['x', 'y', 'z']
            if points.shape[1] == 4:
                fields.append('intensity')
        else:
            fields = list(points.dtype.names)
        
        f.write(f'FIELDS {" ".join(fields)}\n')
        f.write('SIZE ' + ' '.join(['4'] * len(fields)) + '\n')
        f.write('TYPE ' + ' '.join(['F'] * len(fields)) + '\n')
        f.write('COUNT ' + ' '.join(['1'] * len(fields)) + '\n')
        f.write(f'WIDTH {points.shape[0]}\n')
        f.write('HEIGHT 1\n')
        f.write('VIEWPOINT 0 0 0 1 0 0 0\n')
        f.write(f'POINTS {points.shape[0]}\n')
        
        if binary:
            f.write('DATA binary\n')
            if points.dtype.names is None:
                points.astype(np.float32).tofile(f)
            else:
                points.tofile(f)
        else:
            f.write('DATA ascii\n')
            np.savetxt(f, points)
