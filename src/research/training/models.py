"""The first milestone implements only the five-label detector head."""

from transformers import BertConfig, BertForTokenClassification

from .features import LABELS


def build_model(config):
    return BertForTokenClassification(BertConfig(**{**config, 'num_labels': len(LABELS),
        'id2label': dict(enumerate(LABELS)), 'label2id': {name: i for i, name in enumerate(LABELS)},
        'classifier_dropout': .1}))
