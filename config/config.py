import os
import torch


# Torch-specific
use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')
num_workers = 4


# Train/val-specific
val_freq = 500
resume_from_epoch = 0


# Data-specific
project_root = '~/myProjects/fake-news-classifier'
dataset_root = os.path.join(project_root, 'datasets', 'LIAR_PLUS')
cleaned_dataset_root = os.path.join(project_root, 'cache')
model_path = os.path.join(project_root, 'checkpoints', 'model_best.pth')
model_final_path = os.path.join(project_root, 'checkpoints', 'model_final.pth')
results_dir = os.path.join(project_root, 'results')


# Hyper-parameters
batch_size = 16
val_batch_size = 16
test_batch_size = 16
n_epochs = 10
lr = 0.005
