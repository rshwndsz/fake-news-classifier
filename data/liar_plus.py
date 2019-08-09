import os
from torchtext import data, vocab
import logging
import spacy

from config import config as cfg
from . import preprocessing

logger = logging.getLogger(__name__)
en = spacy.load('en_core_web_sm')


def tokenize(sentence):
    return [tok.text for tok in en.tokenizer(sentence)]


logger.debug('Pre-processing TSV files...')
preprocessing.prepare_tsv(os.path.join(cfg.dataset_root, 'train2.tsv'), 'train')
preprocessing.prepare_tsv(os.path.join(cfg.dataset_root, 'val2.tsv'), 'val')
preprocessing.prepare_tsv(os.path.join(cfg.dataset_root, 'test2.tsv'), 'test')


text = data.Field(lower=True, batch_first=True, tokenize=tokenize)
fields = [('text', text),
          ('label', data.Field(lower=True, sequential=False, use_vocab=False, is_target=True)),
          ('metadata', None)]

logger.debug('Reading cleaned TSV files...')
train_set, val_set, test_set = data.TabularDataset.splits(
                                path=cfg.dataset_root,
                                train='train2.tsv',
                                validation='val2.tsv',
                                test='test2.tsv',
                                format='tsv',
                                fields=fields
                               )

logger.debug('Building vocabulary...')
text.build_vocab(train_set, val_set, min_freq=3)

logger.debug('Loading glove vectors...')
pretrained_vectors = vocab.Vectors(os.path.join(cfg.dataset_root,
                                                'glove_embeddings/glove.6B.100d.txt'))
text.vocab.load_vectors(pretrained_vectors)

logger.debug(f'Done preparing datasets.\nVocab Vector shape: {text.vocab.vectors.shape}')

train_loader = data.BucketIterator(train_set,
                                   batch_size=cfg.batch_size,
                                   sort_key=lambda x: len(x.text),
                                   device=cfg.device,
                                   shuffle=True,
                                   sort=False
                                   )

val_loader = data.BucketIterator(val_set,
                                 batch_size=cfg.val_batch_size,
                                 sort_key=lambda x: len(x.text),
                                 device=cfg.device,
                                 shuffle=True,
                                 sort=False
                                 )

test_loader = data.BucketIterator(test_set,
                                  batch_size=cfg.test_batch_size,
                                  sort_key=lambda x: len(x.text),
                                  device=cfg.device
                                  )
logger.debug('Created iterators for train/val/test. Done with data prep!')
