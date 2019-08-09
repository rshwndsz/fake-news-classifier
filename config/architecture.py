from . import config as cfg
from models import BiLSTM
from data import liar_plus


model_name = 'BiDirectional LSTM'
model = BiLSTM(liar_plus.text.vocab.vectors,
               lstm_layer=2,
               padding_idx=liar_plus.text.vocab.stoi[liar_plus.text.pad_token],
               hidden_dim=128
               ).to(cfg.device)
