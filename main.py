import torch
import argparse
import logging
import coloredlogs

from config import (config as cfg,
                    architecture as arch,
                    data_loaders as dl)

# Setup colorful logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)


# noinspection PyShadowingNames
def train(model, optimizer, criterion, resume_from_epoch, min_val_loss):
    """
    Train the model

    :param model: Model to be trained
    :param optimizer: Method to compute gradients
    :param criterion: Criterion for computing loss
    :param resume_from_epoch: Resume training from this epoch
    :param min_val_loss: Save models with lesser loss value on val set
    """
    for epoch in range(resume_from_epoch, cfg.n_epochs):
        # training
        if epoch % cfg.val_freq == 0:
            # validation & saving best model
            pass


def val(model):
    """
    Check model accuracy on validation set.

    :param model: Model to be tested
    :return: Validation accuracy
    """
    # validation


def test(model):
    # testing
    pass


if __name__ == '__main__':
    # CLI
    parser = argparse.ArgumentParser(description=f'CLI for {arch.model_name}')
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--load', type=bool, default=False)
    args = parser.parse_args()

    model = arch.model
    optimizer = arch.optimizer
    criterion = arch.criterion
    resume_from_epoch = cfg.resume_from_epoch
    max_val_accuracy = cfg.min_val_loss

    if args.load:
        # Load from checkpoint
        checkpoint = torch.load(cfg.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        resume_from_epoch = checkpoint['epoch']
        min_val_loss = checkpoint['val_loss']

    if args.phase == 'train':
        train(model, optimizer, criterion, resume_from_epoch, max_val_accuracy)

    elif args.phase == 'test':
        test(model)

    else:
        raise ValueError('Choose one of train/validate/test')
