from typing import Dict
import logging
import os
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer

from config import config as cfg


logger = logging.getLogger(__name__)


@DatasetReader.register('liar_plus')
class LiarPlusDatasetReader(DatasetReader):

    TRAIN_FILE = os.path.join(cfg.dataset_root, 'train2.tsv')
    TEST_FILE = os.path.join(cfg.dataset_root, 'test2.tsv')

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Tokenizer = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy)

        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        if file_path == 'train':
            data_dir = self.TRAIN_FILE

        elif file_path == 'test':
            data_dir = self.TEST_FILE

        else:
            raise ValueError('Only train/test')

    @overrides
    def text_to_instance(self):
        pass
