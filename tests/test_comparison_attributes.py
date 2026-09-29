import pytest

from research.contracts import validate_document


def module():
    from research.comparison import attributes
    return attributes


def test_softcite_uses_mention_context_not_document_aggregate_and_utf16():
    doc = validate_document({'document_id': 'd', 'text': '🧪 We used R 4.0.'})
    mention = {'software-name': {'rawForm': 'R', 'offsetStart': 11, 'offsetEnd': 12},
               'version': {'rawForm': '4.0', 'offsetStart': 13, 'offsetEnd': 16},
               'mentionContextAttributes': {k: {'value': k == 'used'} for k in ('created', 'used', 'shared')},
               'documentContextAttributes': {k: {'value': True} for k in ('created', 'used', 'shared')}}
    chunks = [{'window': {'text': doc['text'], 'start': 0}, 'raw': {'native': {'mentions': [mention]}}}]
    rows = module().softcite_occurrences(doc, chunks, 'utf16')
    assert rows[0]['name_span'] == {'start': 10, 'end': 11}
    assert rows[0]['version_links'] == [{'text': '4.0', 'span': {'start': 12, 'end': 15}}]
    assert rows[0]['intents'] == ['used']
    assert rows[0]['sentiment'] is None


def test_missing_context_does_not_turn_into_mentioned_or_neutral():
    doc = validate_document({'document_id': 'd', 'text': 'R'})
    chunks = [{'window': {'text': 'R', 'start': 0}, 'raw': {'native': {'mentions': [
        {'software-name': {'rawForm': 'R', 'offsetStart': 0, 'offsetEnd': 1}}]}}}]
    row = module().softcite_occurrences(doc, chunks, 'codepoint')[0]
    assert row['intents'] is None and row['sentiment'] is None
    assert 'intents' in row['invalid_fields']


def test_failed_pipeline_fields_preserve_detected_name_without_fabricated_attributes():
    doc = validate_document({'document_id': 'd', 'text': 'R'})
    native = {'document_id': 'd', 'text_revision': doc['text_revision'], 'field_predictions': [
        {'name': 'R', 'name_span': {'start': 0, 'end': 1},
         'software': {'status': 'success', 'value': 'R'},
         'versions': {'status': 'success', 'value': []},
         'intents': {'status': 'failed', 'value': None},
         'sentiment': {'status': 'unavailable', 'value': None}}]}
    row = module().pipeline_occurrences(doc, native)[0]
    assert row['name'] == 'R' and row['intents'] is None and row['sentiment'] is None
    assert set(row['invalid_fields']) == {'intents', 'sentiment'}
    native['text_revision'] = 'wrong'
    with pytest.raises(ValueError, match='identity'):
        module().pipeline_occurrences(doc, native)


def test_alias_score_keeps_unsupported_and_no_positive_support_distinct():
    gold = [{'document_id': 'd', 'left': [0, 1], 'right': [5, 6], 'label': 'alias'}]
    predictions = [{'document_id': 'd', 'left': [5, 6], 'right': [0, 1], 'label': 'alias'}]
    assert module().score_alias_pairs(gold, predictions, supported=True)['f1'] == 1.0
    assert module().score_alias_pairs(gold, [], supported=True)['fn'] == 1
    assert module().score_alias_pairs(gold, [], supported=False)['f1'] is None
    negatives = [{**gold[0], 'label': 'not_alias'}]
    result = module().score_alias_pairs(negatives, [], supported=True)
    assert result['positive_support'] == 0 and result['f1'] is None
    assert module().score_alias_pairs(negatives, predictions, supported=True)['fp'] == 1
