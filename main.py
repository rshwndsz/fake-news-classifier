import torch
import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F
import argparse
import logging
import coloredlogs
from tqdm import tqdm
import numpy as np
import os
from sklearn.metrics import accuracy_score

from config import (config as cfg,
                    architecture as arch,
                    data_loaders as dl)

# Setup colorful logging
logging.basicConfig()
logger = logging.getLogger('main.py')
logger.root.setLevel(logging.DEBUG)
coloredlogs.install(level='DEBUG', logger=logger)


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

    logger.info('Training Model...')
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
        for train_batch in tqdm(iter(train_loader)):
            step += 1
            # Set model in training mode
            model.train()

            # Move (text, label) to device
            text = train_batch.text.to(device)
            label = train_batch.label.type(torch.LongTensor).to(device)
            logger.debug(f'Shape of label: {label.shape}')
            logger.debug(f'Label: {label}')

            # Standard training loop
            model.zero_grad()
            prediction = model.forward(text)
            logger.debug(f'Shape of prediction: {prediction.shape}')
            logger.debug(f'Prediction: {prediction}')
            label = label.type(torch.LongTensor)
            loss = criterion(prediction, label)

            # Collect losses
            losses.append(loss.cpu().data.numpy())
            train_record.append(loss.cpu().data.numpy())

            # Back-prop
            loss.backward()
            optimizer.step()

            # Validation every `eval_every`
            if step % eval_every == 0:
                logger.info('Validating Model...')
                # Set model in eval mode
                model.eval()
                model.zero_grad()

                val_loss = []
                for val_batch in iter(val_loader):
                    # Load (text, label) onto device
                    val_text = val_batch.text.to(device)
                    val_label = val_batch.label.to(device).type(torch.LongTensor)

                    # Forward pass and collect loss
                    val_prediction = model.forward(val_text).to(device)
                    val_loss.append(criterion(val_prediction, val_label).cpu().data.detach().numpy())

                    # One hot representation
                    # See: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/33
                    val_label = nn.functional.one_hot(val_label, num_classes=6).type(torch.FloatTensor)
                    logger.debug(f'val_label: {val_label}')
                    # Convert predictions to binary
                    view = val_prediction.view(-1, 6)
                    val_prediction = (view == view.max(dim=1, keepdim=True)[0]).view_as(val_prediction)
                    logger.debug(f'val_prediction: {val_prediction}')

                    val_record.append({'step': step,
                                       'loss': np.mean(val_loss),
                                       'accuracy': accuracy_score(val_label.view(-1).cpu().detach().numpy(),
                                                                  val_prediction.view(-1).cpu().detach().numpy())
                                       })

                    logger.info('step {}/ epoch {} - train_loss {:.4f} - val_loss {:.4f} - val_acc {:4f}'.format(
                        step, epoch, np.mean(losses), val_record[-1]['loss'], val_record[-1]['accuracy']))

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
        torch.save(info, os.path.join(cfg.results_dir,'best_model_binary.info'))
        torch.save(m, os.path.join(cfg.results_dir, 'best_model_binary.m'))
    else:
        torch.save(info, os.path.join(cfg.results_dir, 'best_model_hex.info'))
        torch.save(m, os.path.join(cfg.results_dir, 'best_model_hex.m'))


def load(binary):
    """Helper function to load model"""
    if binary:
        m = torch.load(os.path.join(cfg.results_dir, 'best_model_binary.m'))
        info = torch.load(os.path.join(cfg.results_dir, 'best_model_binary.info'))
    else:
        m = torch.load(os.path.join(cfg.results_dir, 'best_model_hex.m'))
        info = torch.load(os.path.join(cfg.results_dir, 'best_model_hex.info'))
    return m, info


# noinspection PyShadowingNames
def test(test_loader, binary=False):
    """
    Test the model on the test set

    :param test_loader: Data loader for the test set
    :param binary: if True binary classification
    :return Accuracy
    """
    model, m_info = load(binary)
    logger.info(f'Model info: {m_info}')

    model.lstm.flatten_parameters()

    model.eval()
    model.zero_grad()

    test_loss = []
    test_acc = []
    step = 0
    for test_batch in iter(test_loader):
        step += 1
        # Load (text, label) onto device
        test_text = test_batch.text.to(cfg.device)
        test_label = test_batch.label.to(cfg.device).type(torch.LongTensor)
        test_label = nn.functional.one_hot(test_label, num_classes=6).type(torch.FloatTensor)

        # Forward pass and collect loss
        test_prediction = model.forward(test_text).to(cfg.device)
        view = test_prediction.view(-1, 6)
        test_prediction = (view == view.max(dim=1, keepdim=True)[0]).view_as(test_prediction)

        test_loss.append(criterion(test_prediction, test_label).cpu().data.detach().numpy())
        test_acc.append(accuracy_score(test_label.view(-1).cpu().detach().numpy(),
                                       test_prediction.view(-1).cpu().detach().numpy()))

        logger.info('step {} - test_loss {:.4f} - test_acc {:4f}'.format(
            step, np.mean(test_loss), np.mean(test_acc)))

    return np.mean(test_acc)


if __name__ == '__main__':
    # CLI
    parser = argparse.ArgumentParser(description=f'CLI for {arch.model_name}')
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--binary', type=str, default='no')
    args = parser.parse_args()

    if args.binary == 'yes':
        args.binary = True
    elif args.binary == 'no':
        args.binary = False

    model = arch.model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=cfg.lr)

    logger.info(f'Using model: {model}')
    logger.info(f'Using criterion: {criterion}')
    logger.info(f'Using optimizer: {optimizer}')

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
        acc = test(model, args.binary)
        print(f'Final accuracy: {acc}')

    else:
        raise ValueError('Choose one of train/test.')
