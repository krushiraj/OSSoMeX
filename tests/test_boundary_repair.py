import pytest

from research.training import boundary
from research.training.decode import inference_decoder


def test_missing_policy_key_disables_repair():
    assert boundary.boundary_repair_policy({}) == boundary.NONE
    assert boundary.boundary_repair_policy({'inference': {'decoder': 'wordpiece-bio-v1'}}) == boundary.NONE


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError):
        boundary.boundary_repair_policy({'inference': {'decoder': 'wordpiece-bio-v1',
                                                         'boundary_repair': 'made-up'}})


def test_decoder_accepts_policy_key_and_still_rejects_others():
    assert inference_decoder({'inference': {'decoder': 'wordpiece-bio-v1',
                                            'boundary_repair': boundary.SOFTWARE_SPAN_RULES_V1}}) == 'wordpiece-bio-v1'
    with pytest.raises(ValueError):
        inference_decoder({'inference': {'decoder': 'wordpiece-bio-v1', 'typo': 'x'}})


def _repair(text, spans, policy=boundary.SOFTWARE_SPAN_RULES_V1):
    return boundary.repair_spans(text, [dict(span, text=text[span['start']:span['end']]) for span in spans], policy)


def _bounds(spans):
    return sorted((span['text'], span['start'], span['end']) for span in spans)


def test_trademark_suffix_is_removed():
    text = 'We used ImageJ® here.'
    spans, events = _repair(text, [{'label': 'SOFTWARE', 'start': 8, 'end': 15}])
    assert _bounds(spans) == [('ImageJ', 8, 14)]
    assert [event['reason'] for event in events] == ['trademark_suffix']


def test_bibliography_letter_is_cut_using_the_year_outside_the_span():
    text = 'Scikit Learn-h, 2021, is a tool.'
    start = text.index('Scikit')
    end = start + len('Scikit Learn-h')
    spans, events = _repair(text, [{'label': 'SOFTWARE', 'start': start, 'end': end}])
    assert _bounds(spans) == [('Scikit Learn', start, start + 12)]
    assert [event['reason'] for event in events] == ['citation_suffix']


def test_trailing_hyphen_is_kept_when_no_year_follows():
    text = 'The tool is called abc-h today.'
    start = text.index('abc-h')
    spans, events = _repair(text, [{'label': 'SOFTWARE', 'start': start, 'end': start + len('abc-h')}])
    assert _bounds(spans) == [('abc-h', start, start + len('abc-h'))]
    assert events == []


def test_glued_function_word_is_cut_from_a_camel_case_tail():
    text = 'Scikit-LearnThe package'
    start = text.index('Scikit')
    spans, events = _repair(text, [{'label': 'SOFTWARE', 'start': start, 'end': start + 15}])
    assert _bounds(spans) == [('Scikit-Learn', start, start + 12)]
    assert [event['reason'] for event in events] == ['glued_function_word']


def test_name_ending_in_a_short_word_is_not_cut():
    text = 'We visited Berlin and Rome.'
    start = text.index('Berlin')
    spans, events = _repair(text, [{'label': 'SOFTWARE', 'start': start, 'end': start + 6}])
    assert _bounds(spans) == [('Berlin', start, start + 6)]
    assert events == []


def test_absorbed_version_is_promoted_to_a_version_span():
    text = 'scikit-learn 12 is out'
    start = text.index('scikit')
    spans, events = _repair(text, [{'label': 'SOFTWARE', 'start': start, 'end': start + 15}])
    assert _bounds(spans) == [('12', 13, 15), ('scikit-learn', start, 12)]
    assert [event['reason'] for event in events] == ['absorbed_version', 'whitespace_trim']
    promoted = [event for event in events if event['reason'] == 'absorbed_version'][0]['promoted']
    assert (promoted['label'], promoted['text']) == ('VERSION', '12')


def test_absorbed_version_covers_dotted_release_numbers():
    text = 'install numpy 1.21.0 now'
    start, end = text.index('numpy'), text.index('now') - 1
    spans, _ = _repair(text, [{'label': 'SOFTWARE', 'start': start, 'end': end}])
    assert _bounds(spans) == [('1.21.0', 14, 20), ('numpy', start, 13)]


def test_digit_suffix_inside_a_name_is_left_alone():
    text = 'used Python3 daily'
    start = text.index('Python3')
    spans, events = _repair(text, [{'label': 'SOFTWARE', 'start': start, 'end': start + 7}])
    assert _bounds(spans) == [('Python3', start, start + 7)]
    assert events == []


def test_name_entirely_inside_a_code_identifier_is_dropped():
    text = 'we call check_python_libraries() first'
    start = text.index('python')
    spans, events = _repair(text, [{'label': 'SOFTWARE', 'start': start, 'end': start + 6}])
    assert spans == []
    assert [event['reason'] for event in events] == ['identifier_internal']


def test_name_at_a_word_edge_is_kept():
    text = 'python and check_python_libraries'
    start = text.index('python')
    spans, events = _repair(text, [{'label': 'SOFTWARE', 'start': start, 'end': start + 6}])
    assert _bounds(spans) == [('python', start, start + 6)]
    assert events == []


def test_nested_span_is_resolved_in_favour_of_the_longer_one():
    text = 'SPSS and SPSS.GIS are both here'
    spans, events = _repair(text, [{'label': 'SOFTWARE', 'start': 0, 'end': 4},
                                   {'label': 'SOFTWARE', 'start': 9, 'end': 17}])
    assert _bounds(spans) == [('SPSS', 0, 4), ('SPSS.GIS', 9, 17)]
    assert events == []


def test_disabled_policy_returns_input_untouched():
    text = 'We used ImageJ® and check_python_libraries'
    given = [{'label': 'SOFTWARE', 'start': 8, 'end': 15, 'text': 'ImageJ®'}]
    spans, events = boundary.repair_spans(text, given, boundary.NONE)
    assert spans == given
    assert events == []


def test_repaired_span_always_matches_the_source_text():
    text = 'SPSS.GIS via ImageJ® plus scikit-learn 12'
    for name in ['SPSS.GIS', 'ImageJ®', 'scikit-learn 12']:
        start = text.index(name)
        spans, _ = _repair(text, [{'label': 'SOFTWARE', 'start': start, 'end': start + len(name)}])
        for span in spans:
            assert text[span['start']:span['end']] == span['text']


def test_span_outside_the_document_is_rejected():
    with pytest.raises(ValueError):
        _repair('short', [{'label': 'SOFTWARE', 'start': 0, 'end': 99}])


def test_span_text_disagreeing_with_the_source_is_rejected():
    with pytest.raises(ValueError):
        boundary.repair_spans('ImageJ here', [{'label': 'SOFTWARE', 'start': 0, 'end': 6,
                                               'text': 'SciPy'}], boundary.SOFTWARE_SPAN_RULES_V1)
