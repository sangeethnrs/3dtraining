import json
import datetime
from pathlib import Path
import numpy as np
import torch

def save_predictions_with_probabilities(pred_dicts, class_names, output_dir, frame_id, points, feature_embeddings, tag='default'):
    """Save predictions with class probabilities and feature embeddings to JSON file"""
    num_classes = len(class_names)
    
    predictions = []
    
    # Convert feature embeddings to numpy
    if isinstance(feature_embeddings, torch.Tensor):
        feature_embeddings = feature_embeddings.cpu().numpy()
    
    for i in range(len(pred_dicts[0]['pred_boxes'])):
        pred_box = pred_dicts[0]['pred_boxes'][i].cpu().numpy()
        pred_score = float(pred_dicts[0]['pred_scores'][i].cpu().numpy())
        pred_label = int(pred_dicts[0]['pred_labels'][i].cpu().numpy())
        
        # Handle feature embeddings
        if feature_embeddings is not None and i < len(feature_embeddings):
            embedding = feature_embeddings[i]
            embedding = np.nan_to_num(embedding, nan=0.0)  # Replace NaN with 0
            embedding = embedding.tolist()
        else:
            embedding = []
        
        pred_dict = {
            'box': pred_box.tolist(),
            'score': pred_score,
            'label': pred_label,
            'class_name': class_names[pred_label - 1],
            'feature_embedding': {
                'values': embedding,
                'dimension': len(embedding),
                'type': 'combined_backbone_features'
            }
        }
        predictions.append(pred_dict)
    
    # Create output dictionary with metadata
    output_dict = {
        'frame_id': frame_id,
        'file_name': str(points),
        'timestamp': datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
        'predictions': predictions,
        'metadata': {
            'num_classes': num_classes,
            'class_names': class_names,
            'feature_embedding_info': {
                'total_dimension': feature_embeddings.shape[-1] if feature_embeddings is not None else 0,
                'extraction_method': 'backbone_features',
                'timestamp': datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                'username': 'sangeethnrs'
            }
        }
    }
    
    # Save to file
    output_file = Path(output_dir) / f'{tag}_frame_{frame_id:06d}_predictions.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(output_dict, f, indent=2)
    
    return output_file
