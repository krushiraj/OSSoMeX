import math

import pytest
import torch

from research.training import decode


def test_inconsistent_subwords_form_one_complete_name_with_honest_score():
    # Real failure pattern: greedy B/O/I emitted `na` and `i`.
    probabilities = torch.tensor([
        [.422, .573, .003, .001, .001],
        [.558, .004, .437, .0005, .0005],
        [.359, .002, .637, .001, .001],
    ])
    result = decode.decode_wordpiece(probabilities.log(), [(0, 2), (2, 5), (5, 6)], [0, 0, 0])
    assert len(result) == 1
    assert result[0] == {'label': 'SOFTWARE', 'start': 0, 'end': 6,
                         'score': pytest.approx((.573 * .437 * .637) ** (1 / 3))}


def test_digit_name_suffix_is_not_dropped_or_called_a_version():
    probabilities = torch.tensor([
        [.109, .885, .005, .0005, .0005],
        [.05, .001, .948, .0005, .0005],
        [.932, .001, .065, .001, .001],
    ])
    result = decode.decode_wordpiece(probabilities.log(), [(0, 5), (5, 6), (6, 7)], [0, 0, 0])
    assert [(s['label'], s['start'], s['end']) for s in result] == [('SOFTWARE', 0, 7)]
    assert result[0]['score'] < .5


def test_orphan_suffix_does_not_promote_confident_nonsoftware_word():
    probabilities = torch.tensor([[.97, .02, .008, .001, .001], [.11, .001, .887, .001, .001]])
    assert decode.decode_wordpiece(probabilities.log(), [(0, 5), (5, 6)], [0, 0]) == []


def test_explicit_attached_version_can_end_software_inside_a_word():
    logits = torch.full((3, 5), -10.)
    logits[0, 1], logits[1, 3], logits[2, 4] = 10., 10., 10.
    result = decode.decode_wordpiece(logits, [(0, 4), (4, 5), (5, 6)], [0, 0, 0])
    assert [(s['label'], s['start'], s['end']) for s in result] == [('SOFTWARE', 0, 4), ('VERSION', 4, 6)]


def test_adjacent_mentions_multiword_names_and_versions_stay_separate():
    logits = torch.full((7, 5), -10.)
    for i, label in enumerate([1, 1, 2, 0, 3, 4, 4]):
        logits[i, label] = 10.
    result = decode.decode_wordpiece(logits, [(0, 1), (2, 7), (8, 14), (14, 15), (16, 17), (17, 18), (18, 20)],
                                      [0, 1, 2, 3, 4, 5, 6])
    assert [(s['label'], s['start'], s['end']) for s in result] == [
        ('SOFTWARE', 0, 1), ('SOFTWARE', 2, 14), ('VERSION', 16, 20)]


@pytest.mark.parametrize('logits,offsets,words', [
    (torch.zeros(2, 5), [(0, 1)], [0, 0]),
    (torch.zeros(2, 4), [(0, 1), (1, 2)], [0, 0]),
    (torch.tensor([[math.nan] * 5]), [(0, 1)], [0]),
    (torch.zeros(2, 5), [(0, 1), (1, 2)], [0]),
    (torch.zeros(2, 5), [(0, 2), (1, 2)], [0, 0]),
])
def test_invalid_inputs_fail_closed(logits, offsets, words):
    with pytest.raises(ValueError):
        decode.decode_wordpiece(logits, offsets, words)


def test_empty_sequence_has_no_mentions():
    assert decode.decode_wordpiece(torch.empty(0, 5), [], []) == []
