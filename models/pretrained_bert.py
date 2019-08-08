from pytorch_transformers import AdamW, WarmupLinearSchedule
from pytorch_transformers import (BertForSequenceClassification,
                                  BertTokenizer)
import logging

logger = logging.getLogger(__name__)

pretrained_weights = 'bert-base-uncased'

tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
model = BertForSequenceClassification.from_pretrained(pretrained_weights)


