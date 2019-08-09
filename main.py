import torch
import argparse
import logging
import coloredlogs
import numpy as np

from config import (config as cfg,
                    architecture as arch,
                    data_loaders as dl)

# Setup colorful logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)


# noinspection PyShadowingNames
def train(model,
          optimizer,
          criterion,
          n_epochs,
          eval_every,
          train_loader,
          val_loader,
          device,
          resume_from_epoch=0,
          early_stop=1,
          warmup_epoch=2,
          ):

    step = 0
    max_loss = 1e5
    no_improve_epoch = 0
    no_improve_in_previous_epoch = False
    fine_tuning = False
    train_record = []
    val_record = []
    losses = []

    for epoch in range(resume_from_epoch, n_epochs):
        # Early Stopping
        if epoch >= warmup_epoch:
            if no_improve_in_previous_epoch:
                no_improve_epoch += 1
                if no_improve_epoch >= early_stop:
                    break
            else:
                no_improve_epoch = 0
            no_improve_in_previous_epoch = True

        # Fine tuning
        if not fine_tuning and epoch >= warmup_epoch:
            model.embedding.weight.requires_grad = True
            fine_tuning = True

        train_loader.init_epoch()

        # Training in PyTorch
        for train_batch in iter(train_loader):
            step += 1
            # Set model in training mode
            model.train()

            # Move (text, label) to device
            text = train_batch.text.to(device)
            label = train_batch.label.type(torch.Tensor).to(device)

            # Standard training loop
            model.zero_grad()
            prediction = model.forward(text).view(-1)
            loss = criterion(prediction, label)

            # Collect losses
            losses.append(loss.cpu().data.numpy())
            train_record.append(loss.cpu().data.numpy())

            # Back-prop
            loss.backward()
            optimizer.step()

            # Validation every `eval_every`
            if step % eval_every == 0:
                # Set model in eval mode
                model.eval()
                model.zero_grad()

                val_loss = []
                for val_batch in iter(val_loader):
                    # Load (text, label) onto device
                    val_text = val_batch.text.to(device)
                    val_label = val_batch.text.to(device)

                    # Forward pass and collect loss
                    val_prediction = model.forward(val_text).view(-1)
                    val_loss.append(criterion(val_prediction, val_label).cpu().data.numpy())

                    val_record.append({'step': step,
                                       'loss': np.mean(val_loss)})

                    logger.info('epoch {:02} - step {:06} - train_loss {:.4f} - val_loss {:.4f} '.format(
                        epoch, step, np.mean(losses), val_record[-1]['loss']))

                    # Save best model
                    if epoch >= warmup_epoch:
                        if val_record[-1]['loss'] <= max_loss:
                            save(m=model, info={
                                'step': step,
                                'epoch': epoch,
                                'train_loss': np.mean(losses),
                                'val_loss': val_record[-1]['loss'],
                            })
                            max_loss = val_record[-1]['loss']
                            no_improve_in_previous_epoch = False


def save(m, info):
    """Helper function to save model"""
    torch.save(info, 'best_model.info')
    torch.save(m, 'best_model.m')


def load():
    """Helper function to load model"""
    m = torch.load('best_model.m')
    info = torch.load('best_model.info')
    return m, info


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
        raise ValueError('Choose one of train/test')
