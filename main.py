import torch
import torch.nn as nn
import torch.optim as optim
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
          binary=False
          ):
    """
    Training loop

    :param model: Model to be trained
    :param optimizer: Optimizer used
    :param criterion: Loss function
    :param n_epochs: Number of epochs for training
    :param eval_every: Validation frequency
    :param train_loader: Data loader for the training set
    :param val_loader: Data loader for the validation set
    :param device: 'cpu' or 'cuda'
    :param resume_from_epoch: Start training from epoch
    :param early_stop: Number of epochs to wait before ending training
    :param warmup_epoch: Number of warmup/debug epochs
    :param binary: if True models are saved with _binary in path
    """

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
                                'val_loss': val_record[-1]['loss']},
                                 binary=binary)
                            max_loss = val_record[-1]['loss']
                            no_improve_in_previous_epoch = False


def save(m, info, binary):
    """Helper function to save model"""
    if binary:
        torch.save(info, 'best_model_binary.info')
        torch.save(m, 'best_model_binary.m')
    else:
        torch.save(info, 'best_model_hex.info')
        torch.save(m, 'best_model_hex.m')


def load(binary):
    """Helper function to load model"""
    if binary:
        m = torch.load('best_model_binary.m')
        info = torch.load('best_model_binary.info')
    else:
        m = torch.load('best_model_hex.m')
        info = torch.load('best_model_hex.info')
    return m, info


# noinspection PyShadowingNames
def test(test_loader, binary=False):
    model, m_info = load(binary)
    logger.info(f'Model info: {m_info}')

    model.lstm.flatten_parameters()

    model.eval()

    test_predictions = []
    test_labels = []
    test_loader.init_epoch()
    for test_batch in iter(test_loader):
        text = test_batch.text.cuda()
        test_labels += test_batch.label.data.numpy().tolist()
        test_predictions += torch.sigmoid(model.forward(text).view(-1)).cpu().data.numpy().tolist()

    # TODO Compute metrics


if __name__ == '__main__':
    # CLI
    parser = argparse.ArgumentParser(description=f'CLI for {arch.model_name}')
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--binary', type=bool, default=False)
    args = parser.parse_args()

    model = arch.model_binary if args.binary else arch.model_hex
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=cfg.lr)

    if args.phase == 'train':
        train(model=model,
              optimizer=optimizer,
              criterion=criterion,
              n_epochs=cfg.n_epochs,
              eval_every=cfg.val_freq,
              train_loader=dl.train_loader,
              val_loader=dl.val_loader,
              device=cfg.device,
              resume_from_epoch=cfg.resume_from_epoch,
              early_stop=1,
              warmup_epoch=2,
              binary=args.binary)

    elif args.phase == 'test':
        test(model)

    else:
        raise ValueError('Choose one of train/test')
