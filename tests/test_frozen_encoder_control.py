from copy import deepcopy

import pytest

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')

from test_attribute_models import tiny_config
from test_full_label_train import recipe
from test_training_features import tiny_tokenizer


def detector_rows():
    rows = []
    for label, source in [(1, 'ecosystems'), (3, 'europepmc'), (0, 'ecosystems')]:
        allowed = [False] * 5
        allowed[label] = True
        rows.append({'feature_id': f'{source}-{label}', 'document_id': f'd-{label}',
            'work_group_id': f'w-{label}', 'source': source, 'input_ids': [2, 11, 14, 3],
            'attention_mask': [1] * 4, 'allowed_labels': [[True] * 5, allowed, [True] * 5, [True] * 5],
            'active_mask': [False, True, False, False], 'provenance': {'task_id': f'task-{label}'}})
    return rows


def detector_model(vocab_size):
    return transformers.BertForTokenClassification(transformers.BertConfig(
        **{**tiny_config(), 'vocab_size': vocab_size, 'num_labels': 5}))


def test_frozen_detector_changes_head_but_preserves_encoder(tiny_tokenizer):
    from research.training.full_label_train import train_stage_model

    model = detector_model(len(tiny_tokenizer))
    encoder_before = deepcopy(model.bert.state_dict())
    head_before = deepcopy(model.classifier.state_dict())
    report = train_stage_model(model, tiny_tokenizer, detector_rows(),
        {**recipe(), 'freeze_encoder': True}, 'detector', 'cpu')

    assert all(torch.equal(value, model.bert.state_dict()[name]) for name, value in encoder_before.items())
    assert any(not torch.equal(value, model.classifier.state_dict()[name]) for name, value in head_before.items())
    assert report['freeze_encoder'] is True
    assert report['initial_encoder_sha256'] == report['final_encoder_sha256']
    assert report['initial_head_sha256'] != report['final_head_sha256']
    assert report['encoder_trainable_parameters'] == 0
    assert report['head_trainable_parameters'] == sum(p.numel() for p in model.classifier.parameters())
    assert report['weights_changed'] is True


def test_fine_tuned_detector_changes_encoder_with_same_draws_and_schedule(tiny_tokenizer):
    from research.training.full_label_train import train_stage_model

    torch.manual_seed(19)
    initial = detector_model(len(tiny_tokenizer))
    frozen, fine_tuned = deepcopy(initial), deepcopy(initial)
    rows = detector_rows()
    frozen_report = train_stage_model(frozen, tiny_tokenizer, rows,
        {**recipe(), 'freeze_encoder': True}, 'detector', 'cpu')
    fine_tuned_report = train_stage_model(fine_tuned, tiny_tokenizer, rows, recipe(), 'detector', 'cpu')

    assert fine_tuned_report['freeze_encoder'] is False
    assert fine_tuned_report['initial_encoder_sha256'] == frozen_report['initial_encoder_sha256']
    assert fine_tuned_report['initial_head_sha256'] == frozen_report['initial_head_sha256']
    assert fine_tuned_report['initial_encoder_sha256'] != fine_tuned_report['final_encoder_sha256']
    assert fine_tuned_report['initial_head_sha256'] != fine_tuned_report['final_head_sha256']
    assert fine_tuned_report['encoder_trainable_parameters'] == sum(p.numel() for p in fine_tuned.bert.parameters())
    for key in ('feature_draw_counts', 'source_draw_counts', 'work_draw_counts',
                'optimizer_steps', 'warmup_steps', 'steps_per_epoch', 'sample_draws', 'effective_batch_size'):
        assert frozen_report[key] == fine_tuned_report[key]


@pytest.mark.parametrize('value', [0, 1, 'true', None, [], {}])
def test_freeze_encoder_rejects_nonboolean_config_before_training(tiny_tokenizer, value):
    from research.training.full_label_train import train_stage_model

    with pytest.raises(ValueError, match='freeze_encoder'):
        train_stage_model(detector_model(len(tiny_tokenizer)), tiny_tokenizer, detector_rows(),
            {**recipe(), 'freeze_encoder': value}, 'detector', 'cpu')


def test_freeze_encoder_rejects_attribute_stage(tiny_tokenizer):
    from research.training.attribute_models import build_attribute_model
    from research.training.full_label_train import train_stage_model

    with pytest.raises(ValueError, match='freeze_encoder'):
        train_stage_model(build_attribute_model(tiny_config(), 'linker'), tiny_tokenizer, [],
            {**recipe(), 'freeze_encoder': True}, 'linker', 'cpu')


def test_detector_checkpoint_records_actual_freeze_and_weight_changes(tmp_path, tiny_tokenizer):
    from research.training.full_label_train import train_stage_model
    from research.training.runner import save_detector

    model = detector_model(len(tiny_tokenizer))
    config = {**recipe(), 'freeze_encoder': True}
    report = train_stage_model(model, tiny_tokenizer, detector_rows(), config, 'detector', 'cpu')
    manifest = save_detector(model, tiny_tokenizer, tmp_path / 'detector', config, report,
        {'purpose': 'plumbing_test'})

    assert manifest['recipe']['freeze_encoder'] is True
    assert manifest['training']['freeze_encoder'] is True
    assert manifest['training']['initial_encoder_sha256'] == manifest['training']['final_encoder_sha256']
    assert manifest['training']['initial_head_sha256'] != manifest['training']['final_head_sha256']


@pytest.mark.parametrize('value', [True, False])
def test_legacy_detector_fit_rejects_freeze_option_before_data_loading(tmp_path, value):
    from research.training.runner import fit_detector

    output = tmp_path / 'detector'
    with pytest.raises(ValueError, match='full-label train --stage detector'):
        fit_detector(tmp_path / 'missing-data', {**recipe(), 'freeze_encoder': value}, output, 'cpu')
    assert not output.exists()
