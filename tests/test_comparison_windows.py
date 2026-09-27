import pytest

from research.contracts import text_revision


@pytest.fixture
def tiny_tokenizer(tmp_path):
    transformers = pytest.importorskip('transformers')
    vocabulary = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]',
                  'We', 'used', 'NumPy', 'Tool', '##X', '1', '.', '24', 'and', 'later']
    vocab = tmp_path / 'vocab.txt'
    vocab.write_text('\n'.join(vocabulary) + '\n')
    return transformers.BertTokenizerFast(vocab_file=str(vocab), do_lower_case=False)


def document(text, name='d'):
    return {'document_id': name, 'text': text, 'text_revision': text_revision(text)}


def test_windows_round_trip_and_fit_capacity(tiny_tokenizer):
    from research.comparison.windows import freeze_windows
    source = document('🧪 We used ToolX 1.24.\r\n' * 150 + 'NumPy\n')
    windows = freeze_windows([source], tiny_tokenizer)
    assert len(windows) > 2
    assert windows[0]['start'] == 0
    assert windows[-1]['end'] == len(source['text'])
    assert windows[-1]['text'].endswith('NumPy\n')
    assert '🧪' in windows[0]['text'] and '\r\n' in windows[0]['text']
    assert len({w['window_id'] for w in windows}) == len(windows)
    for window in windows:
        assert source['text'][window['start']:window['end']] == window['text']
        assert window['document_id'] == source['document_id']
        assert window['text_revision'] == source['text_revision']
        assert window['window_text_revision'] == text_revision(window['text'])
        tokens = tiny_tokenizer(window['text'], add_special_tokens=False)['input_ids']
        assert window['content_tokens'] == len(tokens) <= 480
    for left, right in zip(windows, windows[1:]):
        assert left['start'] < right['start'] <= left['end'] < right['end']
        overlap = source['text'][right['start']:left['end']]
        assert len(tiny_tokenizer(overlap, add_special_tokens=False)['input_ids']) == 64
    assert freeze_windows([source], tiny_tokenizer) == windows


def test_overlong_whitespace_free_sequence_preserves_all_characters(tiny_tokenizer):
    from research.comparison.windows import freeze_windows
    source = document('Tool' + 'X' * 80)
    windows = freeze_windows([source], tiny_tokenizer, max_content_tokens=8, overlap_tokens=2)
    assert len(windows) > 1
    assert windows[-1]['end'] == len(source['text'])
    covered = set()
    for window in windows:
        assert window['text'] == source['text'][window['start']:window['end']]
        assert len(tiny_tokenizer(window['text'], add_special_tokens=False)['input_ids']) <= 8
        covered.update(range(window['start'], window['end']))
    assert covered == set(range(len(source['text'])))


@pytest.mark.parametrize('capacity,overlap', [(0, 0), (481, 64), (True, 0), (8, -1), (8, 8), (8, True)])
def test_invalid_window_configuration_is_rejected(tiny_tokenizer, capacity, overlap):
    from research.comparison.windows import freeze_windows
    with pytest.raises(ValueError):
        freeze_windows([document('NumPy')], tiny_tokenizer,
                       max_content_tokens=capacity, overlap_tokens=overlap)


def test_invalid_revision_and_duplicate_documents_are_rejected(tiny_tokenizer):
    from research.comparison.windows import freeze_windows
    source = document('NumPy')
    with pytest.raises(ValueError, match='revision'):
        freeze_windows([{**source, 'text_revision': text_revision('other')}], tiny_tokenizer)
    with pytest.raises(ValueError, match='duplicate'):
        freeze_windows([source, source], tiny_tokenizer)


def test_tokenizer_that_cannot_fit_one_character_fails_explicitly():
    from research.comparison.windows import freeze_windows

    def tokenizer(text, **kwargs):
        return {'input_ids': [1] * 10, 'offset_mapping': [(0, len(text))] * 10}

    with pytest.raises(ValueError, match='cannot_fit'):
        freeze_windows([document('🧪')], tokenizer, max_content_tokens=2, overlap_tokens=0)


def test_retokenization_expansion_shrinks_at_codepoint_boundaries():
    from research.comparison.windows import freeze_windows

    def tokenizer(text, **kwargs):
        offsets = [(0, 2), (2, 4), (4, 5), (5, 6)] if text == 'abcdef' else [
            (index, index + 1) for index in range(len(text))]
        return {'input_ids': list(range(len(offsets))), 'offset_mapping': offsets}

    windows = freeze_windows([document('abcdef')], tokenizer, max_content_tokens=2, overlap_tokens=1)
    assert [(window['start'], window['end']) for window in windows] == [
        (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
    assert all(window['content_tokens'] == 2 for window in windows)
