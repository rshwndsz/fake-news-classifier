import pandas as pd
from config import config as cfg


def prepare_tsv(tsv_path, mode='train'):
    """
    Make the tsv file more BERT friendly
    :param tsv_path: Path to .tsv file
    :param mode: Train/val/test
    :return:
    """
    df = pd.read_csv(tsv_path, sep='\t')
    df['statement'] = df.statement.str.replace("\n", " ")
    df['context'] = df.context.str.replace("\n", " ")
    df['justification'] = df.justification.str.replace("\n", " ")

    if mode == 'train':
        df.to_csv('cache/train.tsv', sep='\t', index=False, header=False)
    elif mode == 'val':
        df.to_csv('cache/val.tsv', sep='\t', index=False, header=False)
    elif mode == 'test':
        df.to_csv('cache/test.tsv', sep='\t', index=False, header=False)
