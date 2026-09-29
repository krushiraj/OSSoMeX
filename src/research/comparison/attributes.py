"""Exact-occurrence projection for existing field metrics, without label guessing."""

from copy import deepcopy

from ..evaluation.metrics import prf
from .alignment import native_span


def pipeline_occurrences(document, native):
    if any(native.get(key) != document[key] for key in ('document_id', 'text_revision')):
        raise ValueError('pipeline input identity mismatch')
    rows = []
    for field in native['field_predictions']:
        start, end = field['name_span']['start'], field['name_span']['end']
        if document['text'][start:end] != field['name'] or not 0 <= start < end <= len(document['text']):
            raise ValueError('unaligned pipeline name')
        row = {key: document[key] for key in ('document_id', 'text_revision')}
        row.update(name=field['name'], name_span=deepcopy(field['name_span']),
                   version_links=[], intents=None, sentiment=None, invalid_fields=[])
        for source, target in [('versions', 'version_links'), ('intents', 'intents'), ('sentiment', 'sentiment')]:
            if field[source]['status'] == 'success':
                row[target] = deepcopy(field[source]['value'])
            else:
                row['invalid_fields'].append(source)
        rows.append(row)
    return rows


def softcite_occurrences(document, chunks, offset_unit):
    rows = []
    for chunk in chunks:
        window = chunk['window']
        body = chunk['raw'].get('native')
        if body is None and chunk['raw'].get('status_code') == 204:
            continue
        if not isinstance(body, dict):
            raise ValueError('missing Softcite native evidence')
        mentions = body.get('software', body.get('mentions'))
        if not isinstance(mentions, list):
            raise ValueError('missing Softcite mentions')
        for mention in mentions:
            def span(key):
                value = mention[key]
                return native_span(window['text'], {'text': value['rawForm'],
                    'start': value['offsetStart'], 'end': value['offsetEnd']},
                    unit=offset_unit, offset_base=window['start'])
            name = span('software-name')
            row = {key: document[key] for key in ('document_id', 'text_revision')}
            row.update(name=name['text'], name_span={key: name[key] for key in ('start', 'end')},
                       version_links=[], intents=None, sentiment=None, invalid_fields=[])
            if mention.get('version') is not None:
                version = span('version')
                row['version_links'] = [{'text': version['text'],
                    'span': {key: version[key] for key in ('start', 'end')}}]
            context = mention.get('mentionContextAttributes', {})
            labels = ('created', 'used', 'shared')
            if all(isinstance(context.get(label), dict) and type(context[label].get('value')) is bool for label in labels):
                row['intents'] = [label for label in labels if context[label]['value']] or ['mentioned']
            else:
                row['invalid_fields'].append('intents')
            rows.append(row)
    return rows


def score_alias_pairs(references, predictions, *, supported):
    """Score only explicitly reviewed pairs, including missed positive edges."""
    def key(row):
        if row.get('label') not in ('alias', 'not_alias'):
            raise ValueError('invalid alias label')
        return (row['document_id'], *sorted((tuple(row['left']), tuple(row['right']))))
    known = {}
    for row in references:
        pair = key(row)
        if pair in known and known[pair] != row['label']:
            raise ValueError('conflicting alias reference')
        known[pair] = row['label']
    gold = {pair for pair, label in known.items() if label == 'alias'}
    positive = {key(row) for row in predictions if row['label'] == 'alias'}
    predicted = positive & set(known)
    result = prf(len(gold & predicted), len(predicted - gold), len(gold - predicted))
    if not supported:
        result = dict.fromkeys(result)
    return {**result, 'supported': supported, 'positive_support': len(gold),
            'reviewed_pairs': len(known), 'unreviewed_positive_predictions': len(positive - set(known)),
            'scope': 'explicitly_reviewed_exact_endpoint_pairs_only_not_end_to_end'}
