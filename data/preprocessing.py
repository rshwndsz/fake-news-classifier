import pandas as pd
import os
import logging
import coloredlogs
from config import config as cfg

logger = logging.getLogger('data/liar_plus.py')
coloredlogs.install(level='DEBUG', logger=logger)


def prepare_tsv(tsv_path, mode='train'):
    """
    Cleanup TSV files

    :param tsv_path: Path to .tsv file
    :param mode: Train/val/test
    :return:
    """
    df = pd.read_csv(tsv_path, sep='\t', names=('id',
                                                'json_id',
                                                'label',
                                                'statement',
                                                'subjects',
                                                'speaker',
                                                'speaker_title',
                                                'state_info',
                                                'party',
                                                'barely_true',
                                                'false',
                                                'half_true',
                                                'mostly_true',
                                                'pants_fire',
                                                'context',
                                                'justification'))
    # Remove incomplete rows
    df = df.dropna()

    # Convert labels to integers
    # See: https://stackoverflow.com/a/23307361
    labels = {'true': 1,
              'mostly-true': 2,
              'half-true': 3,
              'barely-true': 4,
              'false': 5,
              'pants-fire': 6}
    df['label'] = df['label'].map(labels)
    # See: https://stackoverflow.com/a/22391554
    logger.info(f"{mode}_set stats:\n{df['label'].value_counts()}")

    # Remove line endings and replace by space
    df['statement'] = df['statement'].str.replace("\n", " ")
    df['context'] = df['context'].str.replace("\n", " ")
    df['justification'] = df['justification'].str.replace("\n", " ")

    # See: https://stackoverflow.com/a/32529152
    df['text'] = df[['statement', 'justification', 'context']].apply(lambda x: ' '.join(x), axis=1)

    # See: https://stackoverflow.com/a/33378952
    df['metadata'] = df.apply(lambda row: {'subjects': row['subjects'],
                                           'speaker': row['speaker'],
                                           'speaker_title': row['speaker_title'],
                                           'state_info': row['state_info'],
                                           'party': row['party'],
                                           'credit_history': {
                                               'barely_true': row['barely_true'],
                                               'false': row['false'],
                                               'half_true': row['half_true'],
                                               'mostly_true': row['mostly_true'],
                                               'pants_fire': row['pants_fire']
                                                }
                                           }, axis=1)

    # See: https://stackoverflow.com/a/34683105
    clean_df = df.filter(['text', 'label', 'metadata'])

    # Serve up some rows for a sanity check
    logger.info(f'Sanity check rows: \n{clean_df.head()}')

    # Save cleaned up files in cache/
    if mode == 'train':
        clean_df.to_csv(os.path.join(cfg.cleaned_dataset_root, 'train.tsv'), sep='\t', index=False, header=False)
    elif mode == 'val':
        clean_df.to_csv(os.path.join(cfg.cleaned_dataset_root, 'val.tsv'), sep='\t', index=False, header=False)
    elif mode == 'test':
        clean_df.to_csv(os.path.join(cfg.cleaned_dataset_root, 'test.tsv'), sep='\t', index=False, header=False)
