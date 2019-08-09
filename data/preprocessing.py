import pandas as pd
import os
from config import config as cfg


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
                                                'pants_on_fire',
                                                'context',
                                                'justification'))
    # Remove incomplete rows
    df = df.dropna()

    # Remove line endings and replace by space
    df['statement'] = df['statement'].str.replace("\n", " ")
    df['context'] = df['context'].str.replace("\n", " ")
    df['justification'] = df['justification'].str.replace("\n", " ")

    # Save cleaned up files in cache/
    if mode == 'train':
        df.to_csv(os.path.join(cfg.project_root, 'cache', 'train.tsv'), sep='\t', index=False, header=False)
    elif mode == 'val':
        df.to_csv(os.path.join(cfg.project_root, 'cache', 'val.tsv'), sep='\t', index=False, header=False)
    elif mode == 'test':
        df.to_csv(os.path.join(cfg.project_root, 'cache', 'test.tsv'), sep='\t', index=False, header=False)
