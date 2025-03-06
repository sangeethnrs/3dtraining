def save_predictions_with_probabilities(pred_dicts, class_names, output_dir, frame_id, points, feature_embeddings):
    """Save predictions with class probabilities and feature embeddings to JSON file"""
    num_classes = len(class_names)
    
    # Get class probabilities
    cls_probs = get_class_probabilities(pred_dicts[0], num_classes)
    
    predictions = []
    if cls_probs is not None:
        cls_probs = cls_probs.cpu().numpy()
    
    # Get feature embedding dimensionality
    feature_dim = feature_embeddings.shape[-1] if feature_embeddings is not None else 0
    
    for i in range(len(pred_dicts[0]['pred_boxes'])):
        pred_box = pred_dicts[0]['pred_boxes'][i].cpu().numpy()
        pred_score = float(pred_dicts[0]['pred_scores'][i].cpu().numpy())
        pred_label = int(pred_dicts[0]['pred_labels'][i].cpu().numpy())
        
        # Convert feature embeddings to list and handle NaN values
        if feature_embeddings is not None:
            embedding = feature_embeddings[i].cpu().numpy()
            embedding = np.nan_to_num(embedding, nan=0.0)  # Replace NaN with 0
            embedding = embedding.tolist()
        else:
            embedding = []
        
        # Create prediction dictionary with feature embedding metadata
        pred_dict = {
            'box': pred_box.tolist(),
            'score': pred_score,
            'label': pred_label,
            'class_name': class_names[pred_label - 1],
            'class_probabilities': get_class_probabilities_dict(cls_probs[i] if cls_probs is not None else None, class_names),
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
        'predictions': predictions,
        'metadata': {
            'num_classes': num_classes,
            'class_names': class_names,
            'feature_embedding_info': {
                'total_dimension': feature_dim,
                'extraction_method': 'backbone_features',
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_type': 'PointPillar'  # Update this based on your model
            }
        }
    }
    
    # Save to file
    output_file = Path(output_dir) / f'frame_{frame_id:06d}_predictions.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(output_dict, f, indent=2)
    
    return output_file
