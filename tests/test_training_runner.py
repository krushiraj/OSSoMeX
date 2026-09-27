import json
from copy import deepcopy

import pytest

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')
from test_training_features import tiny_tokenizer


def config():
    return {'seed': 42, 'epochs': 2, 'learning_rate': .001, 'weight_decay': .01,
            'warmup_ratio': .1, 'microbatch_size': 1, 'gradient_accumulation': 2,
            'max_grad_norm': 1., 'max_length': 32, 'overlap': 4}


def test_actual_cpu_training_save_reload_and_corruption(tmp_path, tiny_tokenizer):
    from research.training.models import build_model
    from research.training.features import build_token_features
    from research.training.runner import train_model, save_detector
    from research.training.predict import Detector
    text = 'We used NumPy 1.24.'
    document = {'document_id': 'test', 'text': text}
    labels = [{'name': 'NumPy', 'name_span': {'start': 8, 'end': 13},
               'known': {'software': True, 'versions': True},
               'version_links': [{'text': '1.24', 'span': {'start': 14, 'end': 18}}]}]
    features = build_token_features(document, labels,
                                   [{'start': 0, 'end': len(text), 'fields': {'software': True, 'versions': True}}],
                                   tiny_tokenizer, config())
    torch.manual_seed(42)
    model = build_model({'vocab_size': len(tiny_tokenizer), 'hidden_size': 24, 'num_hidden_layers': 1,
                         'num_attention_heads': 2, 'intermediate_size': 32})
    before = model.classifier.weight.detach().clone()
    report = train_model(model, tiny_tokenizer, {'test': features['windows']}, config(), 'cpu')
    assert report['optimizer_steps'] == 2
    assert not torch.equal(before, model.classifier.weight.detach())
    assert report['draw_counts'] == {'test': 4}
    checkpoint = tmp_path / 'checkpoint'
    save_detector(model, tiny_tokenizer, checkpoint, config(), report, {'purpose': 'plumbing_test'})
    with pytest.raises(ValueError, match='plumbing'):
        Detector(checkpoint, 'cpu')
    detector = Detector(checkpoint, 'cpu', allow_plumbing=True)
    result = detector.predict(document)
    assert result['capabilities']['software_spans'] is True
    assert result['capabilities']['version_linking'] is False
    assert result['capabilities']['intent'] is False
    assert 'public_records' not in result and 'sentiment' not in result
    assert result['status'] in ('success', 'no_mentions')
    for span in result['spans']:
        assert text[span['start']:span['end']] == span['text']
    assert Detector(checkpoint, 'cpu', allow_plumbing=True).predict(document) == result
    with pytest.raises(ValueError, match='revision'):
        detector.predict({**document, 'text_revision': 'sha256:wrong'})
    (checkpoint / 'config.json').write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        Detector(checkpoint, 'cpu', allow_plumbing=True)


def test_unknown_features_and_invalid_recipe_do_not_train(tiny_tokenizer):
    from research.training.runner import train_model
    from research.training.models import build_model
    model = build_model({'vocab_size': len(tiny_tokenizer), 'hidden_size': 24, 'num_hidden_layers': 1,
                         'num_attention_heads': 2, 'intermediate_size': 32})
    with pytest.raises(ValueError, match='eligible'):
        train_model(model, tiny_tokenizer, {}, config(), 'cpu')
    with pytest.raises(ValueError, match='recipe'):
        train_model(model, tiny_tokenizer, {}, {**config(), 'epochs': 0}, 'cpu')
