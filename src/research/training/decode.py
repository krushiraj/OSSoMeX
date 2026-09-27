"""Maximum-context logit stitching followed by document-level BIO repair."""

import math

import torch

from .features import LABELS, _owner


def stitch_logits(windows, logits, token_count):
    if len(windows) != len(logits):
        raise ValueError('window/logit count mismatch')
    boundaries = [(w['content_start'], w['content_end']) for w in windows]
    result = torch.empty(token_count, len(LABELS))
    seen = set()
    for wi, (window, values) in enumerate(zip(windows, logits)):
        if values.shape != (len(window['token_indices']), len(LABELS)) or not torch.isfinite(values).all():
            raise ValueError('invalid window logits')
        for position, index in enumerate(window['token_indices']):
            if index >= 0 and _owner([index], boundaries) == wi:
                result[index] = values[position].detach().cpu()
                seen.add(index)
    if seen != set(range(token_count)):
        raise ValueError('missing source tokens')
    return result


def decode_bio(label_ids, offsets, scores):
    if not len(label_ids) == len(offsets) == len(scores):
        raise ValueError('decode shape mismatch')
    result, current, probabilities = [], None, []

    def finish():
        if current:
            result.append({**current, 'score': math.exp(sum(math.log(max(p, 1e-30)) for p in probabilities) / len(probabilities))})

    for label_id, (start, end), score in zip(label_ids, offsets, scores):
        label = LABELS[label_id]
        if label == 'O':
            finish()
            current, probabilities = None, []
            continue
        kind = label[2:]
        if label.startswith('B-') or current is None or current['label'] != kind:
            finish()
            current, probabilities = {'label': kind, 'start': start, 'end': end}, [score]
        else:
            current['end'] = end
            probabilities.append(score)
    finish()
    return result
