import math

import pytest

torch = pytest.importorskip('torch')

from scripts.scibert_base_probe import inspect_head_keys, rank_mask_predictions


def test_inspect_head_keys_distinguishes_pretraining_from_task_classifier():
    base_keys = [
        'bert.embeddings.word_embeddings.weight',
        'cls.predictions.decoder.weight',
        'cls.seq_relationship.weight',
    ]
    evidence = inspect_head_keys(base_keys)
    assert evidence == {
        'non_encoder_prefixes': ['cls.predictions', 'cls.seq_relationship'],
        'task_classifier_keys': [],
        'software_version_label_keys': [],
    }
    adapted = inspect_head_keys(base_keys + ['classifier.weight', 'classifier.bias'])
    assert adapted['task_classifier_keys'] == ['classifier.bias', 'classifier.weight']


def test_rank_mask_predictions_uses_only_the_mask_position():
    logits = torch.tensor([[[99., 0., 0.], [0., 2., 1.], [0., 0., 99.]]])
    predictions = rank_mask_predictions(logits, 1, ['zero', 'one', 'two'], top_k=2)
    assert [item['token'] for item in predictions] == ['one', 'two']
    assert math.isclose(predictions[0]['probability'], math.exp(2) / (1 + math.exp(2) + math.exp(1)), rel_tol=1e-6)
    assert all(item['probability'] < 1 for item in predictions)
