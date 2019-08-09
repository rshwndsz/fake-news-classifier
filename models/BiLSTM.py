import torch
import torch.nn as nn
import logging
import coloredlogs

logger = logging.getLogger('models/BiLSTM.py')
coloredlogs.install(level='DEBUG', logger=logger)


class BiLSTM(nn.Module):
    def __init__(self,
                 pretrained_lm,
                 padding_idx,
                 static=True,
                 hidden_dim=128,
                 lstm_layer=2,
                 dropout_prob=0.2,
                 binary=False):
        """Baseline Bidirectional LSTM """

        super(BiLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding.from_pretrained(pretrained_lm)
        self.embedding.padding_idx = padding_idx

        if static:
            self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layer,
                            dropout=dropout_prob,
                            bidirectional=True)

        self.classifier = nn.Linear(hidden_dim * lstm_layer * 2, 6)

        self.bn = nn.BatchNorm1d()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        # logger.debug(f'Input shape: {x.shape}')
        x = self.embedding(x)
        # logger.debug(f'Shape of x: {x.shape}')
        x = torch.transpose(x, dim0=1, dim1=0)
        x = self.bn(x)
        # logger.debug(f'Shape of x: {x.shape}')
        lstm_out, (h_n, c_n) = self.lstm(x)
        # logger.debug(f'Shape of lstm_out: {lstm_out.shape} h_n: {h_n.shape} c_n: {c_n.shape}')
        out = self.classifier(self.dropout(torch.cat([c_n[i, :, :] for i in range(c_n.shape[0])], dim=1)))
        out = torch.sigmoid(out)
        # logger.debug(f'Shape of out: {out.shape}')
        return out
