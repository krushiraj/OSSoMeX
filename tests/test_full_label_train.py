from copy import deepcopy
import json

import pytest

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')
from test_training_features import tiny_tokenizer
from test_attribute_models import tiny_config


def recipe():
    return {'seed': 42, 'epochs': 1, 'learning_rate': .001, 'weight_decay': .01,
            'warmup_ratio': .1, 'microbatch_size': 1, 'gradient_accumulation': 2,
            'max_grad_norm': 1., 'min_steps_per_epoch': 2, 'plumbing_test': True,
            'base_model': 'allenai/scibert_scivocab_cased',
            'base_revision': 'ddf0be025f8e432a1870e34811997ba6725bf04a'}


def rows(stage='linker'):
    result = []
    for source, work, target in [('ecosystems', 'a', 1), ('ecosystems', 'b', 0), ('europepmc', 'c', 0)]:
        result.append({'feature_id': source + work, 'document_id': work, 'work_group_id': work,
            'source': source, 'stage': stage, 'input_ids': [2, 11, 14, 3], 'attention_mask': [1] * 4,
            'first_mask': [False, True, False, False], 'second_mask': [False, False, True, False],
            'context_mask': [False, True, True, False], 'distance_bucket': 129, 'order_flag': 1,
            'targets': [target], 'known': [True], 'provenance': {'task_id': 'task-' + work}})
    return result


def test_sampling_balances_sources_then_works_not_feature_counts():
    from research.training.full_label_train import sample_features
    features = rows()
    features.extend([{**features[0], 'feature_id': 'extra-' + str(i)} for i in range(50)])
    sampled = sample_features(features, 42, 6000)
    assert sampled == sample_features(list(reversed(features)), 42, 6000)
    assert sampled != sample_features(features, 43, 6000)
    assert 2700 < sum(row['source'] == 'europepmc' for row in sampled) < 3300
    assert 1200 < sum(row['work_group_id'] == 'a' for row in sampled) < 1800
    assert len(sampled) == 6000


def test_unknown_only_sampling_and_training_are_rejected(tiny_tokenizer):
    from research.training.full_label_train import sample_features, train_stage_model
    from research.training.attribute_models import build_attribute_model
    features = [{**rows()[0], 'known': [False]}]
    with pytest.raises(ValueError, match='eligible'):
        sample_features(features, 42, 2)
    with pytest.raises(ValueError, match='support|eligible'):
        train_stage_model(build_attribute_model(tiny_config(), 'linker'), tiny_tokenizer,
                          features, recipe(), 'linker', 'cpu')


def test_tiny_training_updates_weights_and_is_only_loadable_with_override(tmp_path, tiny_tokenizer):
    from research.training.attribute_models import build_attribute_model
    from research.training.full_label_train import train_stage_model, save_attribute_checkpoint, AttributeCheckpoint
    from research.training.attribute_features import build_inference_candidates
    policy = build_inference_candidates({'document_id': 'd', 'text': 'NumPy'},
        [{'name': 'NumPy', 'name_span': {'start': 0, 'end': 5}}], [], tiny_tokenizer, {})['config']
    model = build_attribute_model({**tiny_config(), 'vocab_size': len(tiny_tokenizer)}, 'linker')
    before = model.head[0].weight.detach().clone()
    report = train_stage_model(model, tiny_tokenizer, rows(), recipe(), 'linker', 'cpu')
    assert report['optimizer_steps'] == 2 and report['effective_optimizer_steps'] > 0
    assert report['weights_changed'] is True
    assert not torch.equal(before, model.head[0].weight)
    assert sum(report['source_draw_counts'].values()) == 4
    assert sum(report['work_draw_counts'].values()) == 4
    assert sum(report['feature_draw_counts'].values()) == 4
    output = tmp_path / 'model'
    save_attribute_checkpoint(model, tiny_tokenizer, output, 'linker', {**recipe(), 'attribute_features': policy},
                              report, {'purpose': 'plumbing_test'})
    with pytest.raises(ValueError, match='plumbing'):
        AttributeCheckpoint(output, 'cpu')
    loaded = AttributeCheckpoint(output, 'cpu', allow_plumbing=True)
    assert loaded.manifest['labels'] == ['linked']
    with pytest.raises(FileExistsError):
        save_attribute_checkpoint(model, tiny_tokenizer, output, 'linker', recipe(), report, {})
    (output / 'model.safetensors').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='changed'):
        AttributeCheckpoint(output, 'cpu', allow_plumbing=True)


@pytest.mark.parametrize('stage', ['linker', 'intent', 'sentiment'])
def test_single_class_support_cannot_publish(tmp_path, tiny_tokenizer, stage):
    from research.training.attribute_models import build_attribute_model
    from research.training.full_label_train import train_stage_model
    features = rows(stage)
    for row in features:
        row['targets'] = [1, 0, 0] if stage == 'intent' else [0]
        row['known'] = [True] * len(row['targets'])
    model = build_attribute_model(tiny_config(), stage)
    with pytest.raises(ValueError, match='support'):
        train_stage_model(model, tiny_tokenizer, features, recipe(), stage, 'cpu')
    assert not (tmp_path / 'manifest.json').exists()


def test_pipeline_partial_capabilities_and_hash_verification(tmp_path, tiny_tokenizer):
    from research.training.models import build_model
    from research.training.runner import save_detector
    from research.training.full_label_train import publish_pipeline, load_pipeline_manifest
    detector = tmp_path / 'detector'
    save_detector(build_model(tiny_config()), tiny_tokenizer, detector, recipe(),
                  {'optimizer_steps': 1, 'effective_optimizer_steps': 1}, {'purpose': 'plumbing_test'})
    with pytest.raises(ValueError, match='plumbing'):
        publish_pipeline(detector, {}, tmp_path / 'pipeline')
    bundle = publish_pipeline(detector, {}, tmp_path / 'pipeline', allow_plumbing=True)
    assert bundle['schema_version'] == 'full-label-checkpoint-1'
    assert bundle['pipeline_complete'] is False
    assert bundle['capabilities']['software_spans'] is True
    assert bundle['capabilities']['version_linking'] is False
    assert bundle['stages']['intent']['status'] == 'unavailable'
    assert bundle['stages']['detector']['path'] == '../detector'
    assert load_pipeline_manifest(tmp_path / 'pipeline', allow_plumbing=True) == bundle
    with pytest.raises(FileExistsError):
        publish_pipeline(detector, {}, tmp_path / 'pipeline', allow_plumbing=True)
    (detector / 'manifest.json').write_bytes((detector / 'manifest.json').read_bytes() + b'\n')
    with pytest.raises(ValueError, match='hash|changed'):
        load_pipeline_manifest(tmp_path / 'pipeline', allow_plumbing=True)


def training_bundle(path, documents, items):
    from research.training.data import select_supervision
    from research.data.manifest import json_bytes, write_jsonl, digest
    path.mkdir()
    for name, value in [('config.json', {}), ('forbidden.json', {'work_group_ids': [], 'text_revisions': [], 'source_ids': []})]:
        (path / name).write_bytes(json_bytes(value))
    write_jsonl(path / 'documents.jsonl', documents)
    write_jsonl(path / 'items.jsonl', items)
    (path / 'manifest.json').write_bytes(json_bytes({'schema_version': 'detector-data-1', 'role': 'train',
        'summary': select_supervision(documents, items)['summary'], 'files': [
            {'path': p.name, 'sha256': digest(p.read_bytes())} for p in sorted(path.iterdir())]}))


def test_fit_unsupported_stage_verifies_bundle_and_does_not_publish(tmp_path, tiny_tokenizer, monkeypatch):
    from test_attribute_features import bundle
    import research.training.full_label_train as module
    document, item = bundle()
    document.update(public=True, text_license='CC-BY-4.0', access_basis={'article_url': 'https://example.org'})
    item['task'].update({key: document[key] for key in ('public', 'text_license', 'access_basis')})
    item['status'] = 'agent_provisional'
    data = tmp_path / 'data'
    training_bundle(data, [document], [item])
    monkeypatch.setattr(module, '_load_base', lambda config: (tmp_path, tiny_tokenizer))
    result = module.fit_stage('sentiment', data, recipe(), tmp_path / 'sentiment', 'cpu')
    assert result['status'] == 'unavailable'
    assert 'missing_class:negative' in result['support']['unavailable_reasons']
    assert not (tmp_path / 'sentiment').exists()
    (data / 'documents.jsonl').write_text('corrupt')
    with pytest.raises(ValueError, match='changed'):
        module.fit_stage('sentiment', data, recipe(), tmp_path / 'other', 'cpu')


def test_real_recipe_rejects_cpu_and_changed_recipe_before_loading(tmp_path):
    from research.training.full_label_train import fit_stage
    config = json.loads(open('configs/scibert/full-label-poc-001.json').read())
    with pytest.raises(ValueError, match='MPS|mps'):
        fit_stage('linker', tmp_path / 'missing', config, tmp_path / 'output', 'cpu')
    with pytest.raises(ValueError, match='recipe'):
        fit_stage('linker', tmp_path / 'missing', {**config, 'epochs': 1}, tmp_path / 'output', 'mps')


def two_source_bundle(tmp_path):
    from test_attribute_features import bundle
    from research.contracts import occurrence_id
    documents, items = [], []
    for i, source in enumerate(['ecosystems', 'europepmc']):
        document, item = bundle('We used NumPy 1.24 and ToolX.' if i == 0 else 'ToolX and NumPy 1.24 later.')
        document.update(document_id='d-' + str(i), work_group_id='work-' + str(i), source=source,
            public=True, text_license='CC-BY-4.0', access_basis={'article_url': 'https://example.org/' + str(i)})
        item['task'].update({key: document[key] for key in ('document_id', 'source', 'public', 'text_license', 'access_basis')})
        item['task']['task_id'] = 'task-' + str(i)
        item['status'] = 'agent_provisional'
        for occurrence in item['annotation']['occurrences']:
            span = occurrence['name_span']
            occurrence.update(document_id=document['document_id'], mention_id=occurrence_id(
                document['document_id'], document['text_revision'], span['start'], span['end']))
        documents.append(document)
        items.append(item)
    data = tmp_path / 'data'
    training_bundle(data, documents, items)
    return data


@pytest.mark.parametrize('stage', ['linker', 'detector'])
def test_tiny_fit_uses_verified_two_source_data_and_safe_checkpoint(tmp_path, tiny_tokenizer, monkeypatch, stage):
    import research.training.full_label_train as module
    from research.training.predict import Detector
    data = two_source_bundle(tmp_path)
    base = tmp_path / 'base'
    encoder = transformers.BertModel(transformers.BertConfig(**{**tiny_config(), 'vocab_size': len(tiny_tokenizer)}))
    encoder.save_pretrained(base, safe_serialization=True)
    tiny_tokenizer.save_pretrained(base)
    monkeypatch.setattr(module, '_load_base', lambda config: (base, tiny_tokenizer))
    output = tmp_path / stage
    manifest = module.fit_stage(stage, data, {**recipe(), 'min_steps_per_epoch': 10}, output, 'cpu')
    assert set(manifest['training']['source_draw_counts']) == {'ecosystems', 'europepmc'}
    assert manifest['training']['weights_changed'] is True
    assert manifest['provenance']['features_sha256']
    assert manifest['provenance']['base_revision'] == 'ddf0be025f8e432a1870e34811997ba6725bf04a'
    if stage == 'linker':
        assert manifest['attribute_features']['marker_token_ids'] == [19, 20, 21, 22]
        module.AttributeCheckpoint(output, 'cpu', allow_plumbing=True)
    else:
        Detector(output, 'cpu', allow_plumbing=True)


def test_build_real_attribute_model_preserves_pretrained_encoder_weights(tmp_path):
    from research.training.attribute_models import build_attribute_model
    torch.manual_seed(17)
    base = transformers.BertModel(transformers.BertConfig(**tiny_config()), add_pooling_layer=False)
    base.save_pretrained(tmp_path, safe_serialization=True)
    model = build_attribute_model({**recipe(), '_base_path': str(tmp_path)}, 'intent')
    assert torch.equal(model.bert.embeddings.word_embeddings.weight, base.embeddings.word_embeddings.weight)
    assert torch.equal(model.bert.encoder.layer[0].attention.self.query.weight,
                       base.encoder.layer[0].attention.self.query.weight)


def test_zero_update_checkpoint_and_stale_support_are_rejected(tmp_path, tiny_tokenizer):
    from research.training.full_label_train import save_attribute_checkpoint
    from research.training.attribute_models import build_attribute_model
    with pytest.raises(ValueError, match='untrained'):
        save_attribute_checkpoint(build_attribute_model(tiny_config(), 'linker'), tiny_tokenizer,
            tmp_path / 'model', 'linker', recipe(), {'optimizer_steps': 1, 'effective_optimizer_steps': 0}, {})
    assert not (tmp_path / 'model').exists()


@pytest.mark.parametrize('tamper,error', [('support', 'support'), ('policy', 'policy')])
def test_reload_rejects_forged_trainability_and_frozen_marker_policy(tmp_path, tiny_tokenizer, tamper, error):
    from research.training.attribute_features import build_inference_candidates
    from research.training.attribute_models import build_attribute_model
    from research.training.full_label_train import train_stage_model, save_attribute_checkpoint, AttributeCheckpoint
    policy = build_inference_candidates({'document_id': 'd', 'text': 'NumPy'},
        [{'name': 'NumPy', 'name_span': {'start': 0, 'end': 5}}], [], tiny_tokenizer, {})['config']
    model = build_attribute_model({**tiny_config(), 'vocab_size': len(tiny_tokenizer)}, 'intent')
    features = rows('intent')
    for i, row in enumerate(features):
        row.update(targets=[int(i == 0), int(i == 1), int(i == 2)], known=[True] * 3)
    report = train_stage_model(model, tiny_tokenizer, features, recipe(), 'intent', 'cpu')
    output = tmp_path / 'model'
    manifest = save_attribute_checkpoint(model, tiny_tokenizer, output, 'intent',
        {**recipe(), 'attribute_features': policy}, report, {'purpose': 'plumbing_test'})
    forged = deepcopy(manifest)
    if tamper == 'support':
        forged['support']['counts']['created']['positive'] = 0
    else:
        forged['attribute_features']['pair_policy'] = 'unrestricted'
        forged['recipe']['attribute_features']['pair_policy'] = 'unrestricted'
    (output / 'manifest.json').write_text(json.dumps(forged))
    with pytest.raises(ValueError, match=error):
        AttributeCheckpoint(output, 'cpu', allow_plumbing=True)


def test_sampler_cannot_multiply_duplicate_feature_support():
    from research.training.full_label_train import train_stage_model
    from research.training.attribute_models import build_attribute_model
    features = rows('intent')
    features[0].update(targets=[1, 1, 1], known=[True] * 3)
    features[1].update(feature_id=features[0]['feature_id'], targets=[0, 0, 0], known=[True] * 3)
    features[2].update(targets=[1, 1, 1], known=[True] * 3)
    with pytest.raises(ValueError, match='duplicate'):
        train_stage_model(build_attribute_model(tiny_config(), 'intent'), None, features, recipe(), 'intent', 'cpu')


def test_identical_duplicates_do_not_inflate_training_support(tiny_tokenizer):
    from research.training.full_label_train import train_stage_model
    from research.training.attribute_models import build_attribute_model
    features = rows()
    report = train_stage_model(build_attribute_model(tiny_config(), 'linker'), tiny_tokenizer,
        features + [deepcopy(features[0])] * 12, recipe(), 'linker', 'cpu')
    assert report['eligible_examples'] == 3
    assert report['support']['counts']['linked']['positive'] == 1
