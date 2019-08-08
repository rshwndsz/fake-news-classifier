import torch
from torchtext import data, vocab
import spacy
import logging
from config import config as cfg

logger = logging.getLogger('liar_plus')

en = spacy.load('en')


def tokenize(sentence):
    return [tok.text for tok in en.tokenizer(sentence)]


# pre-trained vectors support only lower
ID = data.Field(lower=True, sequential=False)
LABEL = data.Field(lower=True, sequential=False)
STATEMENT = data.Field(lower=True, tokenize=tokenize)
CONTEXT = data.Field(lower=True, tokenize=tokenize)
JUSTIFICATION = data.Field(lower=True, tokenize=tokenize)
WORD = data.Field(lower=True, sequential=False)
LIST = data.Field(lower=True, tokenize=tokenize)
COUNT = data.Field(sequential=False, dtype=torch.float64, use_vocab=False)

fields = [('id', None),
          ('json_id', WORD),
          ('label', LABEL),
          ('statement', STATEMENT),
          ('subjects', LIST),
          ('speaker', WORD),
          ('speaker_title', WORD),
          ('state_info', WORD),
          ('party', WORD),
          ('barely_true', COUNT),
          ('false', COUNT),
          ('half_true', COUNT),
          ('mostly_true', COUNT),
          ('pants_on_fire', COUNT),
          ('context', CONTEXT),
          ('justification', JUSTIFICATION)
          ]

logger.debug('Reading TSV files...')
train_set, val_set, test_set = data.TabularDataset.splits(
                                path=cfg.dataset_root,
                                train='train2.tsv',
                                validation='val2.tsv',
                                test='test2.tsv',
                                format='tsv',
                                fields=fields
                               )


logger.debug('Reading glove vectors...')
vec = vocab.Vectors('glove.6B.100d.txt', '../datasets/glove_embeddings')

logger.debug('Building vocabulary...')
STATEMENT.build_vocab(train_set, val_set, max_size=100000, vectors=vec)
JUSTIFICATION.build_vocab(train_set, val_set, max_size=100000, vectors=vec)
CONTEXT.build_vocab(train_set, val_set, max_size=100000, vectors=vec)
WORD.build_vocab(train_set, val_set)
LABEL.build_vocab(train_set, val_set)
LIST.build_vocab(train_set, val_set)

logger.debug('Done preparing datasets.')

train_dl = data.BucketIterator(train_set,
                               batch_size=cfg.batch_size,
                               device=cfg.device,
                               shuffle=True
                               )

val_dl = data.BucketIterator(val_set,
                             batch_size=cfg.val_batch_size,
                             device=cfg.device,
                             shuffle=True
                             )

test_dl = data.BucketIterator(test_set,
                              batch_size=cfg.test_batch_size,
                              device=cfg.device
                              )
