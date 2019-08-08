import torch
from torchtext import data, vocab
import spacy
import os
from config import config as cfg

en = spacy.load('en')


def tokenize(sentence):
    return [tok.text for tok in en.tokenizer(sentence)]


ID = data.Field(lower=True, sequential=False)
LABEL = data.Field(lower=True, sequential=False)
STATEMENT = data.Field(tokenize=tokenize)
CONTEXT = data.Field(tokenize=tokenize)
JUSTIFICATION = data.Field(tokenize=tokenize)
WORD = data.Field(sequential=False)
LIST = data.Field(tokenize=tokenize)
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

train_set, val_set, test_set = data.TabularDataset.splits(
                                path=cfg.dataset_root,
                                train='train2.tsv',
                                validation='val2.tsv',
                                test='test2.tsv',
                                format='tsv',
                                fields=fields
                               )


vec = vocab.Vectors('glove.6B.100d.txt', '../datasets/glove_embeddings')

STATEMENT.build_vocab(train_set, val_set, max_size=100000, vectors=vec)
JUSTIFICATION.build_vocab(train_set, val_set, max_size=100000, vectors=vec)
CONTEXT.build_vocab(train_set, val_set, max_size=100000, vectors=vec)
WORD.build_vocab(train_set, val_set)
LABEL.build_vocab(train_set, val_set)
LIST.build_vocab(train_set, val_set)


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


class BatchGenerator:
    def __init__(self, dl, attributes):
        """A wrapper class for torch-text batches"""
        self.dl, self.attributes = dl, attributes

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for batch in self.dl:
            result = {}
            for attribute in self.attributes:
                result[attribute] = getattr(batch, attribute)
            yield result


train_loader = BatchGenerator(train_dl, train_set.fields)
