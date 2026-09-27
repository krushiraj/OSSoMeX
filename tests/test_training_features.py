import math

import pytest

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')


@pytest.fixture
def tiny_tokenizer(tmp_path):
    vocabulary = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]',
                  '[unused1]', '[unused2]', '[unused3]', '[unused4]',
                  'We', 'used', 'NumPy', 'Tool', '##X', '1', '.', '24', 'and', 'later']
    vocab = tmp_path / 'vocab.txt'
    vocab.write_text('\n'.join(vocabulary) + '\n')
    return transformers.BertTokenizerFast(vocab_file=str(vocab), do_lower_case=False)


def test_unknown_positions_have_no_loss_or_gradient():
    from research.training.losses import partial_token_loss
    logits = torch.zeros(1, 3, 5, requires_grad=True)
    allowed = torch.tensor([[[True, False, False, True, True],
                             [False, True, False, False, False], [True] * 5]])
    active = torch.tensor([[True, True, False]])
    loss = partial_token_loss(logits, allowed, active)
    assert loss.item() == pytest.approx((math.log(5 / 3) + math.log(5)) / 2)
    loss.backward()
    assert logits.grad[0, 2].tolist() == [0] * 5
    assert partial_token_loss(logits, allowed, torch.zeros_like(active)) is None
    with pytest.raises(ValueError, match='allowed'):
        partial_token_loss(logits, torch.zeros_like(allowed), active)


def test_partial_coverage_keeps_versions_unknown_and_unicode_exact(tiny_tokenizer):
    from research.training.features import build_token_features
    document = {'document_id': 'd', 'text': '🧪 We used NumPy 1.24 and ToolX.'}
    labels = [{'name': 'NumPy', 'name_span': {'start': 10, 'end': 15},
               'known': {'software': True, 'versions': False}, 'version_links': []}]
    coverage = [{'start': 2, 'end': 20, 'fields': {'software': True, 'versions': False}}]
    result = build_token_features(document, labels, coverage, tiny_tokenizer, {})
    assert result['exclusions'] == []
    window = result['windows'][0]
    targets = {tuple(offset): (allowed, active) for offset, allowed, active in
               zip(window['offsets'], window['allowed_labels'], window['active_mask']) if offset != [0, 0]}
    assert targets[(10, 15)] == ([False, True, False, False, False], True)
    assert targets[(16, 17)] == ([True, False, False, True, True], True)
    assert targets[(0, 1)][1] is False
    assert targets[(25, 29)][1] is False
    assert window['active_mask'][0] is False and window['active_mask'][-1] is False


def test_subword_versions_and_duplicate_windows_have_one_loss_owner(tiny_tokenizer):
    from research.training.features import build_token_features
    text = 'We used ToolX 1.24. ' * 8
    labels = [{'name': 'ToolX', 'name_span': {'start': 8, 'end': 13},
               'known': {'software': True, 'versions': True},
               'version_links': [{'text': '1.24', 'span': {'start': 14, 'end': 18}}]}]
    result = build_token_features({'document_id': 'd', 'text': text}, labels,
                                 [{'start': 0, 'end': 19, 'fields': {'software': True, 'versions': True}}],
                                 tiny_tokenizer, {'max_length': 14, 'overlap': 6})
    assert len(result['windows']) > 1
    supervised = [(tuple(o), a.index(True)) for w in result['windows']
                  for o, a, active in zip(w['offsets'], w['allowed_labels'], w['active_mask']) if active]
    assert len(supervised) == len({offset for offset, label in supervised})
    assert ((8, 12), 1) in supervised and ((12, 13), 2) in supervised
    assert ((14, 15), 3) in supervised and ((15, 16), 4) in supervised and ((16, 18), 4) in supervised
    spans = {tuple(o) for w in result['windows'] for o in w['offsets'] if o != [0, 0]}
    assert (158, 159) in spans


def test_incompatible_and_overlapping_spans_are_masked_not_negatives(tiny_tokenizer):
    from research.training.features import build_token_features
    document = {'document_id': 'd', 'text': 'We used NumPy.'}
    labels = [{'name': 'Num', 'name_span': {'start': 8, 'end': 11},
               'known': {'software': True, 'versions': False}, 'version_links': []}]
    result = build_token_features(document, labels,
                                 [{'start': 0, 'end': 14, 'fields': {'software': True}}], tiny_tokenizer, {})
    assert result['exclusions'][0]['reason'] == 'token_boundary'
    w = result['windows'][0]
    assert not w['active_mask'][w['offsets'].index([8, 13])]


def test_absolute_offsets_and_invalid_source_spans(tiny_tokenizer):
    from research.training.features import build_token_features
    document = {'document_id': 'd', 'text': 'We used NumPy.', 'offset_base': 100}
    label = {'name': 'NumPy', 'name_span': {'start': 108, 'end': 113},
             'known': {'software': True}, 'version_links': []}
    result = build_token_features(document, [label], [], tiny_tokenizer, {})
    assert [108, 113] in result['windows'][0]['offsets']
    label['name'] = 'wrong'
    with pytest.raises(ValueError, match='source'):
        build_token_features(document, [label], [], tiny_tokenizer, {})


def test_entity_larger_than_window_is_excluded_everywhere(tiny_tokenizer):
    from research.training.features import build_token_features
    text = 'ToolX ' * 10
    labels = [{'name': text.rstrip(), 'name_span': {'start': 0, 'end': 59},
               'known': {'software': True}, 'version_links': []}]
    result = build_token_features({'document_id': 'd', 'text': text}, labels,
                                 [{'start': 0, 'end': 60, 'fields': {'software': True}}],
                                 tiny_tokenizer, {'max_length': 10, 'overlap': 2})
    assert result['exclusions'][0]['reason'] == 'window_limit'
    assert not any(any(w['active_mask']) for w in result['windows'])
