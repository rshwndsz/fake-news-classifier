import os
import torch


# Torch-specific
use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')
num_workers = 4


# Train/val-specific
print_freq = 20
val_freq = 1
resume_from_epoch = 0
min_val_loss = 1000


# Data-specific
project_root = '~/myProjects/fake-news-classifier'
dataset_root = os.path.join(project_root, 'datasets', 'LIAR_PLUS')
cleaned_dataset_root = os.path.join(project_root, 'cache')
model_path = os.path.join(project_root, 'checkpoints', 'model_best.pth')
model_final_path = os.path.join(project_root, 'checkpoints', 'model_final.pth')
results_dir = os.path.join(project_root, 'results')


# Hyper-parameters
batch_size = 8
val_batch_size = 1
test_batch_size = 1
n_epochs = 2
lr = 0.01
