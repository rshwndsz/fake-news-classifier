import torch.nn as nn
import torch.optim as optim

from . import config as cfg
from models import BiLSTM
from data import liar_plus


model_name = 'FakeNewsDetector'
model = BiLSTM(liar_plus.text.vocab.vectors,
               lstm_layer=2,
               padding_idx=liar_plus.text.vocab.stoi[liar_plus.text.pad_token],
               hidden_dim=128
               ).to(cfg.device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=cfg.lr)
