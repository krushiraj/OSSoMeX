from copy import deepcopy

import pytest

from research.comparison.contracts import CAPABILITY_FIELDS, validate_result
from research.contracts import text_revision


def document(text='🧪 NumPy and NumPy 1.24.'):
    return {'document_id': 'd', 'text': text, 'text_revision': text_revision(text)}


def capabilities():
    return {field: field in ('software_spans', 'version_spans') for field in CAPABILITY_FIELDS}


def window_result(source, start, end, name, spans=None):
    text = source['text'][start:end]
    return {'window_id': name, 'status': 'success' if spans else 'no_mentions',
            'spans': spans or [], 'unresolved': [], 'raw': {'native': name}, 'error': None,
            'window': {'window_id': name, 'document_id': source['document_id'],
                       'text_revision': source['text_revision'], 'start': start, 'end': end,
                       'text': text, 'window_text_revision': text_revision(text), 'content_tokens': 5}}


def span(text='NumPy', start=2, end=7, score=0.2):
    return {'label': 'SOFTWARE', 'text': text, 'start': start, 'end': end,
            'score': score, 'score_kind': 'native_score' if score is not None else None,
            'alignment_method': 'native_codepoint'}


def test_native_utf16_and_codepoint_offsets_are_exact():
    from research.comparison.alignment import native_span
    result = native_span('🧪 NumPy', {'text': 'NumPy', 'start': 3, 'end': 8}, unit='utf16')
    assert (result['start'], result['end']) == (2, 7)
    assert result['alignment_method'] == 'native_utf16'
    result = native_span('🧪 NumPy', span(), unit='unicode_codepoint_half_open', offset_base=10)
    assert (result['start'], result['end']) == (12, 17)


@pytest.mark.parametrize('value,unit,base', [
    ({'text': '🧪', 'start': 0, 'end': 1}, 'utf16', 0),
    ({'text': 'NumPy', 'start': True, 'end': 7}, 'codepoint', 0),
    ({'text': 'NumPy', 'start': 2, 'end': 80}, 'codepoint', 0),
    ({'text': 'NumPy', 'start': -1, 'end': 7}, 'codepoint', 0),
    ({'text': 'NumPy', 'start': 7, 'end': 2}, 'codepoint', 0),
    ({'text': 'NumPy', 'start': 2, 'end': 2}, 'codepoint', 0),
    ({'text': 'wrong', 'start': 2, 'end': 7}, 'codepoint', 0),
    ({'text': 'NumPy', 'start': 2, 'end': 7}, 'unknown', 0),
    ({'text': 'NumPy', 'start': 2, 'end': 7}, 'codepoint', True),
    ({'text': 'NumPy', 'start': 2, 'end': 7}, 'codepoint', -1),
])
def test_invalid_native_offsets_never_search_for_a_fallback(value, unit, base):
    from research.comparison.alignment import native_span
    with pytest.raises(ValueError):
        native_span('🧪 NumPy', value, unit=unit, offset_base=base)


@pytest.mark.parametrize('text,value,context,expected', [
    ('NumPy and NumPy', 'NumPy', None, None),
    ('NumPy and NumPy', 'NumPy', 'and NumPy', (10, 15)),
    ('NumPy and NumPy', 'NumPy', 'NumPy', None),
    ('NumPy and NumPy', 'NumPy', 'NumPy and NumPy', None),
    ('NumPy and NumPy', 'NumPy', 'missing NumPy', None),
    ('🧪 NumPy', 'NumPy', None, (2, 7)),
    ('NumPy', 'numpy', None, None),
    ('aaaa', 'aa', None, None),
    ('NumPy', '', None, None),
])
def test_exact_unique_alignment_requires_unique_occurrence_evidence(text, value, context, expected):
    from research.comparison.alignment import align_unique
    result = align_unique(text, value, context)
    if expected is None:
        assert result['status'] == 'unresolved'
        assert 'start' not in result and 'end' not in result
        assert result['reason']
    else:
        assert result['status'] == 'resolved'
        assert (result['start'], result['end']) == expected
        assert result['alignment_method'] == ('exact_unique_context' if context else 'exact_unique')


def test_reducer_keeps_all_contributors_and_earliest_score_even_out_of_order():
    from research.comparison.alignment import reduce_windows
    source = document()
    early = window_result(source, 0, 22, 'early', [span(start=12, end=17, score=0.2)])
    late = window_result(source, 8, len(source['text']), 'late', [span(start=4, end=9, score=0.9),
                          span(text='Num', start=4, end=7, score=None)])
    original = deepcopy([late, early])
    result = reduce_windows(source, 'test', [late, early], capabilities())
    assert [late, early] == original
    assert result['status'] == 'success'
    assert len(result['spans']) == 2
    duplicate = result['spans'][0]
    assert (duplicate['start'], duplicate['end'], duplicate['score']) == (12, 17, 0.2)
    assert duplicate['window_ids'] == ['early', 'late']
    assert duplicate['native_scores'] == [
        {'window_id': 'early', 'score': 0.2, 'score_kind': 'native_score'},
        {'window_id': 'late', 'score': 0.9, 'score_kind': 'native_score'}]
    assert result['raw_artifact'] is None
    validate_result(source, {**result, 'raw_artifact': 'raw/d/index.json'})


@pytest.mark.parametrize('failure', ['failed', 'unresolved', 'bad_span', 'bad_score', 'bad_revision',
                                    'bad_window_text', 'bad_window_id', 'missing_window', 'invalid_status',
                                    'empty_window', 'window_hash', 'window_document', 'window_bounds',
                                    'missing_score', 'missing_raw', 'error_on_success', 'wrong_status',
                                    'unavailable', 'malformed_spans', 'unsupported_label'])
def test_any_bad_window_clears_document_spans_but_retains_raw_diagnostics(failure):
    from research.comparison.alignment import reduce_windows
    source = document()
    good = window_result(source, 0, len(source['text']), 'good', [span()])
    bad = window_result(source, 8, len(source['text']), 'bad', [span(start=4, end=9)])
    if failure == 'failed':
        bad.update(status='failure', spans=[], error='timeout')
    elif failure == 'unresolved':
        bad['unresolved'] = [{'text': 'NumPy', 'reason': 'ambiguous'}]
    elif failure == 'bad_span':
        bad['spans'][0]['start'] = True
    elif failure == 'bad_score':
        bad['spans'][0]['score'] = float('nan')
    elif failure == 'bad_revision':
        bad['window']['text_revision'] = text_revision('other')
    elif failure == 'bad_window_text':
        bad['window']['text'] = 'NumPy'
    elif failure == 'bad_window_id':
        bad['window']['window_id'] = 'other'
    elif failure == 'missing_window':
        del bad['window']
    elif failure == 'empty_window':
        bad['window'] = {}
    elif failure == 'window_hash':
        bad['window']['window_text_revision'] = text_revision('other')
    elif failure == 'window_document':
        bad['window']['document_id'] = 'other'
    elif failure == 'window_bounds':
        bad['window']['start'] = False
    elif failure == 'missing_score':
        del bad['spans'][0]['score']
    elif failure == 'missing_raw':
        del bad['raw']
    elif failure == 'error_on_success':
        bad['error'] = 'transport error'
    elif failure == 'wrong_status':
        bad['status'] = 'no_mentions'
    elif failure == 'unavailable':
        bad.update(status='unavailable', spans=[], error='service missing')
    elif failure == 'malformed_spans':
        bad['spans'] = None
    elif failure == 'unsupported_label':
        bad['spans'][0]['label'] = 'INTENT'
    else:
        bad['status'] = 'invented'
    result = reduce_windows(source, 'test', [good, bad], capabilities())
    assert result['status'] == 'failure' and result['spans'] == []
    assert result['reason']
    assert result['chunks'][0]['raw'] == {'native': 'good'}
    if failure != 'missing_raw':
        assert result['chunks'][1]['raw'] == {'native': 'bad'}
    if failure == 'unresolved':
        assert result['unresolved'][0]['text'] == 'NumPy'
    validate_result(source, {**result, 'raw_artifact': 'raw/d/index.json'})


def test_empty_success_and_missing_windows_are_distinct():
    from research.comparison.alignment import reduce_windows
    source = document()
    result = reduce_windows(source, 'test', [window_result(source, 0, len(source['text']), 'w')], capabilities())
    assert result['status'] == 'no_mentions'
    validate_result(source, {**result, 'raw_artifact': 'raw/d/index.json'})
    result = reduce_windows(source, 'test', [], capabilities())
    assert result['status'] == 'failure' and result['reason']


def test_equal_window_starts_keep_original_order_for_displayed_score():
    from research.comparison.alignment import reduce_windows
    source = document()
    first = window_result(source, 0, len(source['text']), 'z-first', [span(score=0.1)])
    second = window_result(source, 0, 8, 'a-second', [span(score=0.8)])
    result = reduce_windows(source, 'test', [first, second], capabilities())
    assert result['spans'][0]['score'] == 0.1
    assert result['spans'][0]['window_ids'] == ['z-first', 'a-second']


def test_duplicate_result_ids_cannot_hide_a_missing_window():
    from research.comparison.alignment import reduce_windows
    source = document()
    output = window_result(source, 0, len(source['text']), 'w', [span()])
    result = reduce_windows(source, 'test', [output, output], capabilities())
    assert result['status'] == 'failure'
    assert result['spans'] == [] and len(result['chunks']) == 2
    validate_result(source, result)
