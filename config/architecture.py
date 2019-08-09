import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from . import config as cfg
from models import BiLSTM
from data import liar_plus


model_name = 'FakeNewsDetector'
model = BiLSTM(liar_plus.pretrained_vectors,
               lstm_layer=2,
               padding_idx='<unk>',
               hidden_dim=128).to(cfg.device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())
