import wandb

def log_wandb_metrics(metrics, step, prefix='train'):
    """Log metrics to wandb with a specific prefix"""
    if not wandb.run:
        return
        
    wandb_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            wandb_metrics[f'{prefix}/{key}'] = value
    
    wandb.log(wandb_metrics, step=step)
