import torch
from config import config as cfg


def save_val_model(model, epoch, optimizer, val_loss, logger):
    """
    Save model after validation (possibly with min val loss/max val accuracy)
    :param model: Model to be saved
    :param epoch: Save at epoch
    :param optimizer: Optimizer used
    :param val_loss: Min validation loss
    :param logger: Logger used
    """
    logger.info('Saving model...')
    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }, cfg.model_path)
    except FileNotFoundError as fnf_error:
        logger.error(f'{fnf_error}')
    else:
        logger.info('Saved!')


def save_end_model(model, optimizer, logger):
    """
    Save model at the end of training.
    :param model: Model to be saved
    :param optimizer: Optimizer used
    :param logger: Logger used
    """
    logger.info('Saving model at the end of training...')
    try:
        torch.save({
            'epoch': cfg.n_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, cfg.model_final_path)

    except FileNotFoundError as fnf_error:
        logger.error(f'{fnf_error}')
    else:
        logger.info('Saved!')
