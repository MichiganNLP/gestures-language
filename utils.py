import os
import torch

def save_checkpoint(epoch, global_step, model, optimizer):
    '''
    Checkpointing. Saves model and optimizer state_dict() and current epoch and global training steps.
    '''
    checkpoint_path = os.path.join('./saved_models', f'roberta_bs32_2_sec_around_{epoch}_{global_step}.pt')
    n_gpu = 1
    if n_gpu > 1:
        torch.save({
            'epoch' : epoch,
            'global_step' : global_step,
            'model_state_dict' : model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)
    else:
        torch.save({
            'epoch' : epoch,
            'global_step' : global_step,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)

    logger.info("Save checkpoint to {}".format(checkpoint_path))
    return