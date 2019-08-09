import os
from torchtext import data, vocab
import logging
import coloredlogs
import spacy

from config import config as cfg
from . import preprocessing

logger = logging.getLogger('data/liar_plus.py')
coloredlogs.install(level='DEBUG', logger=logger)

en = spacy.load('en_core_web_sm')


def tokenize(sentence):
    return [tok.text for tok in en.tokenizer(sentence)]


logger.debug('Pre-processing TSV files...')
preprocessing.prepare_tsv(os.path.join(cfg.dataset_root, 'train2.tsv'), 'train')
preprocessing.prepare_tsv(os.path.join(cfg.dataset_root, 'val2.tsv'), 'val')
preprocessing.prepare_tsv(os.path.join(cfg.dataset_root, 'test2.tsv'), 'test')


text = data.Field(lower=True, batch_first=True, tokenize=tokenize)
label = data.Field(lower=True, sequential=False, use_vocab=False)
metadata = data.Field()
fields = [('text', text),
          ('label', label),
          ('metadata', metadata)]

logger.info('Reading cleaned TSV files...')
train_set, val_set, test_set = data.TabularDataset.splits(
                                path=cfg.cleaned_dataset_root,
                                train='train.tsv',
                                validation='val.tsv',
                                test='test.tsv',
                                format='tsv',
                                fields=fields
                               )

logger.info('Building vocabulary...')
text.build_vocab(train_set, val_set, min_freq=3)
metadata.build_vocab(train_set, val_set, min_freq=3)

logger.info('Loading vocab vectors from wiki-news-300d-1M.vec...')
text.vocab.load_vectors(vocab.Vectors(name='datasets/word_embeddings/wiki-news-300d-1M.vec'))

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
                                 train=False,
                                 shuffle=True,
                                 sort=False
                                 )

test_loader = data.BucketIterator(test_set,
                                  batch_size=cfg.test_batch_size,
                                  sort_key=lambda x: len(x.text),
                                  device=cfg.device
                                  )

logger.info('Created iterators for train/val/test. Done with data prep!')
