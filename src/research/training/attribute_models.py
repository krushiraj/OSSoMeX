"""Independent SciBERT span and pair classifiers with masked supervision."""

import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertConfig, BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from .attribute_features import SENTIMENT_LABELS, STAGES
from ..contracts import INTENT_BITS
from .runner import BASE_MODEL, BASE_REVISION

STAGE_LABELS = {'linker': ['linked'], 'alias': ['alias'],
                'intent': list(INTENT_BITS), 'sentiment': SENTIMENT_LABELS}


class AttributeModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.stage = config.attribute_stage
        if self.stage not in STAGES:
            raise ValueError('unknown attribute stage')
        self.bert = BertModel(config, add_pooling_layer=False)
        pair = self.stage in ('linker', 'alias')
        if pair:
            self.distance_embedding = nn.Embedding(257, 16)
        width = config.hidden_size * (3 if pair else 2) + (17 if pair else 0)
        self.head = nn.Sequential(nn.Linear(width, 256), nn.ReLU(), nn.Dropout(.1),
                                  nn.Linear(256, len(STAGE_LABELS[self.stage])))
        self.post_init()

    def get_input_embeddings(self):
        return self.bert.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.bert.set_input_embeddings(value)

    def _mean(self, hidden, mask, attention_mask):
        if mask.shape != attention_mask.shape:
            raise ValueError('pool mask shape mismatch')
        mask = mask.bool() & attention_mask.bool()
        if not mask.any(dim=1).all():
            raise ValueError('empty endpoint or context pool mask')
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1)

    def forward(self, input_ids, attention_mask, first_mask, second_mask,
                context_mask, distance_bucket, order_flag):
        hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        first = self._mean(hidden, first_mask, attention_mask)
        if self.stage in ('linker', 'alias'):
            second = self._mean(hidden, second_mask, attention_mask)
            if ((distance_bucket < 0) | (distance_bucket > 256)).any():
                raise ValueError('invalid distance bucket')
            if ((order_flag != 0) & (order_flag != 1)).any():
                raise ValueError('invalid order flag')
            values = [first, second, (first - second).abs(),
                      self.distance_embedding(distance_bucket.long()), order_flag.to(hidden.dtype).unsqueeze(-1)]
        else:
            values = [first, self._mean(hidden, context_mask, attention_mask)]
        return SequenceClassifierOutput(logits=self.head(torch.cat(values, dim=-1)))


def build_attribute_model(config: dict, stage: str):
    if stage not in STAGES:
        raise ValueError('unknown attribute stage')
    tiny = 'hidden_size' in config and 'base_model' not in config
    if tiny:
        encoder_config = BertConfig(**config)
        encoder_config.plumbing_test = True
        encoder = None
    else:
        if config.get('base_model') != BASE_MODEL or config.get('base_revision') != BASE_REVISION:
            raise ValueError('pinned SciBERT base required')
        encoder = BertModel.from_pretrained(config.get('_base_path', BASE_MODEL), revision=BASE_REVISION,
            local_files_only=True, trust_remote_code=False, attn_implementation='eager', add_pooling_layer=False)
        encoder_config = encoder.config
        encoder_config.plumbing_test = bool(config.get('plumbing_test'))
    encoder_config.attribute_stage = stage
    encoder_config.num_labels = len(STAGE_LABELS[stage])
    encoder_config.id2label = dict(enumerate(STAGE_LABELS[stage]))
    encoder_config.label2id = {name: index for index, name in enumerate(STAGE_LABELS[stage])}
    model = AttributeModel(encoder_config)
    if encoder is not None:
        model.bert = encoder
    return model


def attribute_loss(logits, targets, known, stage: str):
    if stage not in STAGES:
        raise ValueError('unknown attribute stage')
    width = 1 if stage == 'sentiment' else len(STAGE_LABELS[stage])
    if logits.ndim != 2 or logits.shape[1] != len(STAGE_LABELS[stage]) or targets.shape != (logits.shape[0], width) or known.shape != targets.shape:
        raise ValueError('attribute loss shape mismatch')
    active = known.bool()
    if not active.any():
        return None
    selected = targets[active]
    if not torch.isfinite(selected).all():
        raise ValueError('invalid active target')
    if stage == 'sentiment':
        if ((selected < 0) | (selected >= 4) | (selected != selected.long())).any():
            raise ValueError('invalid active sentiment target')
        return F.cross_entropy(logits[active[:, 0]], selected.long())
    if ((selected != 0) & (selected != 1)).any():
        raise ValueError('invalid active binary target')
    return F.binary_cross_entropy_with_logits(logits[active], selected.to(logits.dtype))
