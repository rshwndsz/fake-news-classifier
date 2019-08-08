from torchtext import data
from config import config as cfg

ID = data.Field(lower=True, sequential=False)
LABEL = data.Field(lower=True, sequential=False)
TEXT = data.Field(tokenize="spacy")
WORD = data.Field(sequential=False)
COUNT = data.Field(sequential=False, use_vocab=False)

fields = [('id', None),
          ('json_id', WORD),
          ('label', LABEL),
          ('statement', TEXT),
          ('subjects', WORD),
          ('speaker', WORD),
          ('speaker_title', WORD),
          ('state_info', WORD),
          ('party', WORD),
          ('barely_true', COUNT),
          ('false', COUNT),
          ('half_true', COUNT),
          ('mostly_true', COUNT),
          ('pants_on_fire', COUNT),
          ('context', TEXT),
          ('justification', TEXT)
          ]

train_set, val_set, test_set = data.TabularDataset.splits(
                                path=cfg.dataset_root,
                                train='train2.tsv',
                                validation='val2.tsv',
                                test='test2.tsv',
                                format='tsv',
                                fields=fields
                               )

TEXT.build_vocab(train_set, val_set)
WORD.build_vocab(train_set, val_set)
LABEL.build_vocab(train_set, val_set)
