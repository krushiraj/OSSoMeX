"""Maximum-context logit stitching followed by document-level BIO repair."""

import math

import torch

from .features import LABELS, _owner


def inference_decoder(manifest):
    policy = manifest.get('inference', {'decoder': 'greedy-bio-v1'})
    if (not isinstance(policy, dict) or set(policy) != {'decoder'}
            or policy['decoder'] not in ('greedy-bio-v1', 'wordpiece-bio-v1')):
        raise ValueError('invalid inference decoder')
    return policy['decoder']


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


def decode_wordpiece(logits, offsets, word_ids):
    """Maximum-score legal BIO path; software cannot fragment inside a word.

    An explicit B-VERSION may still end a name inside a word (e.g. Tool2).
    Scores use the selected labels, including any low-probability repairs.
    """
    size = len(offsets)
    if (logits.shape != (size, len(LABELS)) or len(word_ids) != size
            or not torch.isfinite(logits).all()):
        raise ValueError('invalid wordpiece decode shape or logits')
    for i, ((start, end), word) in enumerate(zip(offsets, word_ids)):
        if (type(start) is not int or type(end) is not int or not 0 <= start < end
                or type(word) is not int or word < 0
                or i and (start < offsets[i - 1][1] or word < word_ids[i - 1])):
            raise ValueError('invalid wordpiece source alignment')
    if not size:
        return []
    emissions = logits.detach().cpu().log_softmax(-1).tolist()
    previous, backpointers = [0., -math.inf, -math.inf, -math.inf, -math.inf], []
    for i, emission in enumerate(emissions):
        within_word = i > 0 and word_ids[i] == word_ids[i - 1]
        scores, parents = [], []
        for current in range(len(LABELS)):
            candidates = []
            for prior, score in enumerate(previous):
                legal = current not in (2, 4) or prior in (current - 1, current)
                if within_word:
                    legal &= current != 1 and (prior not in (1, 2) or current in (2, 3))
                candidates.append(score if legal else -math.inf)
            parent = max(range(len(LABELS)), key=candidates.__getitem__)
            scores.append(candidates[parent] + emission[current])
            parents.append(parent)
        previous = scores
        backpointers.append(parents)
    current = max(range(len(LABELS)), key=previous.__getitem__)
    labels = []
    for parents in reversed(backpointers):
        labels.append(current)
        current = parents[current]
    labels.reverse()
    return decode_bio(labels, offsets, [math.exp(row[label]) for row, label in zip(emissions, labels)])
