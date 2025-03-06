import os
import glob
import torch
import tqdm
import time
import wandb  # Add this import
from torch.nn.utils import clip_grad_norm_
from pcdet.utils import common_utils, commu_utils
from pcdet.models import load_data_to_gpu

def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    """
    Save checkpoint state
    """
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {
        'epoch': epoch,
        'it': it,
        'model_state': model_state,
        'optimizer_state': optim_state,
        'version': version,
    }

def save_checkpoint(state, filename='checkpoint'):
    """
    Save checkpoint to file
    """
    filename = f'{filename}.pth'
    torch.save(state, filename)
    
def train_one_epoch(
    model, optimizer, train_loader, model_func,
    lr_scheduler, accumulated_iter, optim_cfg,
    rank, tbar, tb_log=None, leave_pbar=False,
    total_it_each_epoch=0, dataloader_iter=None,
    logger=None, logger_iter_interval=50,
    use_logger_to_record=True,
    show_gpu_stat=True,
    use_amp=False,
    wandb_logging=True  # Add wandb_logging parameter with default True
):
    if total_it_each_epoch == 0:
        dataloader_iter = iter(train_loader)
        total_it_each_epoch = len(train_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)
        data_time = common_utils.AverageMeter()
        batch_time = common_utils.AverageMeter()
        forward_time = common_utils.AverageMeter()

    optimizer.zero_grad()
    for cur_it in range(total_it_each_epoch):
        end = time.time()
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')
        
        data_timer = time.time()
        cur_data_time = data_timer - end

        try:
            load_data_to_gpu(batch)
            optimizer.zero_grad()

            forward_timer = time.time()
            ret_dict, tb_dict, disp_dict = model_func(model, batch)
            
            if ret_dict is None or 'loss' not in ret_dict:
                raise ValueError("Model returned None or missing 'loss' in return dictionary")
                
            loss = ret_dict['loss']
            cur_forward_time = time.time() - forward_timer

            loss.backward()
            clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
            optimizer.step()
            lr_scheduler.step(accumulated_iter)

            accumulated_iter += 1
            cur_batch_time = time.time() - end

            # Log to wandb
            if wandb_logging and rank == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/learning_rate': lr_scheduler.get_last_lr()[0],
                    'train/data_time': cur_data_time,
                    'train/forward_time': cur_forward_time,
                    'train/batch_time': cur_batch_time,
                    'train/accumulated_iter': accumulated_iter,
                    **{f'train/{k}': v for k, v in tb_dict.items()}
                }, step=accumulated_iter)

            # Update average meters
            if rank == 0:
                data_time.update(cur_data_time)
                batch_time.update(cur_batch_time)
                forward_time.update(cur_forward_time)
                disp_dict.update({
                    'loss': loss.item(),
                    'lr': lr_scheduler.get_last_lr()[0]
                })

                # Update progress bar
                pbar.update()
                pbar.set_postfix(dict(total_it=accumulated_iter))
                tbar.set_postfix(disp_dict)

                # Log to tensorboard if available
                if tb_log is not None:
                    tb_log.add_scalar('train/loss', loss, accumulated_iter)
                    tb_log.add_scalar('meta_data/learning_rate', lr_scheduler.get_last_lr()[0], accumulated_iter)
                    for key, val in tb_dict.items():
                        tb_log.add_scalar('train/' + key, val, accumulated_iter)

        except Exception as e:
            logger.error(f"Error in training iteration {cur_it}: {str(e)}")
            raise

    if rank == 0:
        pbar.close()
    return accumulated_iter

def train_model(
    model,
    optimizer,
    train_loader,
    model_func,
    lr_scheduler,
    optim_cfg,
    start_epoch,
    total_epochs,
    start_iter,
    rank,
    tb_log,
    ckpt_save_dir,
    train_sampler=None,
    lr_warmup_scheduler=None,
    ckpt_save_interval=1,
    max_ckpt_save_num=50,
    merge_all_iters_to_one_epoch=False,
    logger=None,
    logger_iter_interval=50,
    ckpt_save_time_interval=300,
    use_logger_to_record=True,
    show_gpu_stat=True,
    use_amp=False,
    wandb_logging=True  # Add wandb_logging parameter
):
    # Initialize wandb at the start of training if enabled
    if wandb_logging and rank == 0:
        wandb.init(
            project="openpcdet",  # Set your project name
            config={
                "model": model.__class__.__name__,
                "optimizer": optimizer.__class__.__name__,
                "learning_rate": optim_cfg.LR,
                "epochs": total_epochs,
                "batch_size": train_loader.batch_size,
            }
        )
    
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler

            # Train one epoch
            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader,
                model_func, lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,
                logger=logger,
                logger_iter_interval=logger_iter_interval,
                use_logger_to_record=use_logger_to_record,
                show_gpu_stat=show_gpu_stat,
                use_amp=use_amp,
                wandb_logging=wandb_logging  # Pass wandb_logging parameter
            )

            # Save model checkpoints
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:
                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter),
                    filename=ckpt_name,
                )
                
                if wandb_logging:
                    wandb.save(str(ckpt_name) + '.pth')

    if wandb_logging and rank == 0:
        wandb.finish()

