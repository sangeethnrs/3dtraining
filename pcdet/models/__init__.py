from collections import namedtuple

import numpy as np
import torch

from .detectors import build_detector

try:
    import kornia
except:
    pass 
    # print('Warning: kornia is not installed. This package is only required by CaDDN')



def build_network(model_cfg, num_class, dataset):
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if key == 'camera_imgs':
            batch_dict[key] = val.cuda()
        elif not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib', 'image_paths','ori_shape','img_process_infos']:
            continue
        elif key in ['images']:
            batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()

def model_fn_decorator():
    def model_func(model, batch_dict):
        try:
            # Forward pass
            ret_dict, tb_dict, disp_dict = model(batch_dict)
            
            if ret_dict is None:
                raise ValueError("Model returned None for ret_dict")
            if 'loss' not in ret_dict:
                raise ValueError(f"ret_dict missing 'loss' key. Keys present: {ret_dict.keys()}")
            if not isinstance(ret_dict['loss'], (torch.Tensor, float)):
                raise ValueError(f"Loss is of unexpected type: {type(ret_dict['loss'])}")
                
            return ret_dict, tb_dict, disp_dict
            
        except Exception as e:
            print(f"Error in model_func: {str(e)}")
            print(f"batch_dict keys: {batch_dict.keys()}")
            raise
            
    return model_func
