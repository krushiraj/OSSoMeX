from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

transformers = pytest.importorskip('transformers')


class Logits(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = list(values)
        self.index = 0

    def forward(self, **kwargs):
        value = self.values[min(self.index, len(self.values) - 1)]
        self.index += 1
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(logits=torch.tensor([value], dtype=torch.float32))


@pytest.fixture
def pipeline_factory(monkeypatch, tmp_path):
    from research.training import full_label as module
    from research.training.attribute_features import STAGES
    from research.contracts import text_revision
    vocab = tmp_path / 'vocab.txt'
    vocab.write_text('\n'.join(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]',
        'We', 'used', 'NumPy', 'ToolX', '工具', '1', '2', '.', '24', '26', 'and', 'A', 'B', 'C', 'word']) + '\n')
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(vocab), do_lower_case=False)
    (tmp_path / 'manifest.json').write_text('{}')

    def make(text='We used NumPy 1.24 and ToolX 2.26.', names=('NumPy', 'ToolX'), versions=('1.24', '2.26'),
             missing=(), values=None, detector_status='success'):
        document = {'document_id': 'd', 'text': text}
        spans = []
        for label, items in [('SOFTWARE', names), ('VERSION', versions)]:
            for value in items:
                start = text.index(value)
                spans.append({'label': label, 'start': start, 'end': start + len(value), 'text': value, 'score': .95})
        detector_result = {'schema_version': 'detector-prediction-1', 'document_id': 'd',
            'text_revision': text_revision(text), 'status': detector_status, 'spans': spans,
            'chunks': [{'window': 0, 'status': 'failure' if detector_status == 'failure' else 'success'}]}
        stages = {'detector': {'status': 'available', 'path': 'detector', 'manifest_sha256': 'd' * 64, 'support': None}}
        stages.update({stage: {'status': 'unavailable', 'unavailable_reasons': ['missing_stage']} if stage in missing else
            {'status': 'available', 'path': stage, 'manifest_sha256': 'a' * 64, 'support': {'trainable': True}} for stage in STAGES})
        manifest = {'stages': stages, 'capabilities': {stage: stage not in missing for stage in stages}}
        defaults = {'linker': [[5], [-5], [-5], [5]], 'intent': [[5, 5, -5], [-5, -5, -5]],
                    'sentiment': [[-3, -3, -3, 3]], 'alias': [[-5]]}
        models = {stage: SimpleNamespace(model=Logits((values or {}).get(stage, defaults[stage])), tokenizer=tokenizer,
            device='cpu', manifest={'recipe': {}}, identity='a' * 64) for stage in STAGES}
        monkeypatch.setattr(module, 'load_pipeline_manifest', lambda checkpoint: manifest)
        monkeypatch.setattr(module, 'Detector', lambda *args: SimpleNamespace(tokenizer=tokenizer, predict=lambda doc: deepcopy(detector_result)))
        monkeypatch.setattr(module, 'AttributeCheckpoint', lambda path, device: models[Path(path).name])
        return module.FullLabelPipeline(tmp_path), document
    return make


def test_two_names_two_versions_and_independent_intents(pipeline_factory):
    pipeline, document = pipeline_factory()
    result = pipeline.predict(document)
    assert result['pipeline_complete'] and result['public_contract_complete']
    assert [(r['name'], r['version'], r['intents']) for r in result['public_rows']] == [
        ('NumPy', '1.24', ['created', 'used']), ('ToolX', '2.26', ['mentioned'])]
    assert len(result['pair_predictions']['linker']) == 4
    assert result['occurrences'][0]['scores']['intents']['shared'] < .5
    assert all(result['stage_status'][stage]['occurrence_count'] == 2 for stage in ('detector', 'linker', 'intent', 'sentiment', 'alias'))
    assert 'known' not in result['occurrences'][0]
    assert result['field_predictions'][0]['intents']['evidence_method'] == 'model_input_context'


def test_multiple_versions_expand_one_occurrence(pipeline_factory):
    pipeline, document = pipeline_factory('We used NumPy 1.24 and 1.26.', ('NumPy',), ('1.24', '1.26'),
        values={'linker': [[5], [5]]})
    result = pipeline.predict(document)
    assert len(result['occurrences']) == 1
    assert [row['version'] for row in result['public_rows']] == ['1.24', '1.26']
    assert [row['version_edge_ordinal'] for row in result['public_mappings']] == [0, 1]


def test_unicode_exact_source_and_unavailable_head_has_no_defaults(pipeline_factory):
    pipeline, document = pipeline_factory('🧪 We used 工具.', ('工具',), (), missing=('sentiment',))
    result = pipeline.predict(document)
    assert result['field_predictions'][0]['name_span'] == {'start': 10, 'end': 12}
    assert result['field_predictions'][0]['sentiment']['value'] is None
    assert not result['public_rows'] and not result['public_contract_complete']
    assert result['stage_status']['sentiment']['status'] == 'unavailable'


@pytest.mark.parametrize('stage', ['linker', 'intent', 'sentiment'])
def test_failed_nonfinite_head_cannot_become_a_default(pipeline_factory, stage):
    pipeline, document = pipeline_factory(values={stage: [[float('nan')] * (3 if stage == 'intent' else 4 if stage == 'sentiment' else 1)]})
    result = pipeline.predict(document)
    assert result['status'] == 'partial'
    assert not result['public_rows']
    field = 'versions' if stage == 'linker' else 'intents' if stage == 'intent' else stage
    assert result['field_predictions'][0][field]['value'] is None


def test_detector_failure_blocks_all_downstream_but_keeps_diagnostics(pipeline_factory):
    pipeline, document = pipeline_factory(detector_status='failure')
    result = pipeline.predict(document)
    assert result['status'] == 'failure' and not result['public_rows']
    assert result['detector_diagnostics']['spans']
    assert all(result['stage_status'][stage]['status'] == 'blocked' for stage in ('linker', 'intent', 'sentiment', 'alias'))


def test_alias_failure_keeps_independently_complete_public_rows(pipeline_factory):
    pipeline, document = pipeline_factory(values={'alias': [RuntimeError('broken')]})
    result = pipeline.predict(document)
    assert result['public_contract_complete'] and len(result['public_rows']) == 2
    assert not result['pipeline_complete'] and result['status'] == 'partial'
    assert result['alias_predictions']['pairs'][0]['scores'] is None


def test_alias_group_negative_chord_retains_pairs_and_no_attribute_copy(pipeline_factory):
    pipeline, document = pipeline_factory('A and B and C.', ('A', 'B', 'C'), (),
        values={'alias': [[5], [-5], [5]], 'intent': [[5, -5, -5], [-5, 5, -5], [-5, -5, 5]]})
    result = pipeline.predict(document)
    group = result['alias_predictions']['groups'][0]
    assert len(group['member_mention_ids']) == 3 and len(group['contributing_pair_ids']) == 2
    assert group['negative_chord_pair_ids'] and group['relation_type'] is None
    assert group['preferred_mention_id'] is None and 'confidence' not in group
    assert [row['intents'] for row in result['public_rows']] == [['created'], ['used'], ['shared']]


def test_overlength_pair_is_incomplete_not_absent(pipeline_factory):
    text = 'NumPy ' + 'word ' * 510 + '1.24.'
    pipeline, document = pipeline_factory(text, ('NumPy',), ('1.24',))
    result = pipeline.predict(document)
    assert any(row['stage'] == 'linker' and row['reason'] == 'context_limit' for row in result['exclusions'])
    assert result['field_predictions'][0]['versions']['value'] is None
    assert not result['public_rows']


def test_no_mentions_requires_successful_detector_coverage(pipeline_factory):
    pipeline, document = pipeline_factory('word.', (), ())
    result = pipeline.predict(document)
    assert result['status'] == 'no_mentions' and result['public_contract_complete']


def test_validator_rejects_forged_complete_mapping_and_source(pipeline_factory):
    from research.training.full_label import validate_full_label_prediction
    pipeline, document = pipeline_factory()
    result = pipeline.predict(document)
    result['public_rows'][0]['name'] = 'invented'
    with pytest.raises(ValueError):
        validate_full_label_prediction(result, document)
    result = pipeline.predict(document)
    result['field_predictions'][0]['name_span']['start'] += 1
    with pytest.raises(ValueError):
        validate_full_label_prediction(result, document)


@pytest.mark.parametrize('mutation', ['status', 'score', 'field_label', 'stage', 'alias_subtype'])
def test_validator_rejects_inconsistent_stage_field_and_score_metadata(pipeline_factory, mutation):
    from research.training.full_label import validate_full_label_prediction
    pipeline, document = pipeline_factory()
    result = pipeline.predict(document)
    if mutation == 'status':
        result['status'] = 'no_mentions'
    elif mutation == 'score':
        result['field_predictions'][0]['intents']['scores']['used'] = float('nan')
    elif mutation == 'field_label':
        result['field_predictions'][0]['sentiment']['value'] = 'positive'
    elif mutation == 'stage':
        result['stage_status']['intent']['status'] = 'failed'
        result['pipeline_complete'] = False
        result['status'] = 'partial'
    else:
        result['alias_predictions']['pairs'][0]['relation_type'] = 'invented'
    with pytest.raises(ValueError):
        validate_full_label_prediction(result, document)


def test_unavailable_alias_retains_abstaining_pair_and_complete_rows(pipeline_factory):
    pipeline, document = pipeline_factory(missing=('alias',))
    result = pipeline.predict(document)
    assert result['public_contract_complete'] and not result['pipeline_complete']
    pair = result['alias_predictions']['pairs'][0]
    assert pair['scores'] is None and pair['label'] is None and pair['reasons']


def test_no_eligible_version_candidate_resolves_absence_without_linker(pipeline_factory):
    pipeline, document = pipeline_factory('NumPy.\n\n1.24.', ('NumPy',), ('1.24',), missing=('linker',))
    result = pipeline.predict(document)
    assert result['field_predictions'][0]['versions']['value'] == []
    assert result['public_rows'][0]['version'] is None


def test_partial_pair_failure_keeps_successful_rows_and_diagnostics(pipeline_factory):
    pipeline, document = pipeline_factory(values={'linker': [[5], [-5], RuntimeError('pair failed'), [5]]})
    result = pipeline.predict(document)
    assert result['stage_status']['linker']['status'] == 'partial'
    assert [(row['name'], row['version']) for row in result['public_rows']] == [('NumPy', '1.24')]
    assert result['field_predictions'][1]['versions']['value'] is None
    assert result['pair_predictions']['linker'][-1]['scores']['linked'] > .5


def test_low_confidence_keeps_chosen_labels_and_reviews_canonical_occurrence(pipeline_factory):
    pipeline, document = pipeline_factory(values={'intent': [[0, -1, -1]], 'sentiment': [[0, 0, 0, .1]],
                                                  'linker': [[0], [-1], [-1], [0]]})
    result = pipeline.predict(document)
    assert result['public_rows'][0]['intents'] == ['created']
    assert result['public_rows'][0]['version'] == '1.24'
    assert result['status'] == 'success'
    assert {'low_confidence:intent', 'low_confidence:linker', 'low_confidence:sentiment'} <= set(result['occurrences'][0]['review_reasons'])


def test_real_tiny_cpu_attribute_models_use_production_feature_and_batch_contract(pipeline_factory):
    from research.training.attribute_models import build_attribute_model
    pipeline, document = pipeline_factory()
    for stage, checkpoint in pipeline.stages.items():
        checkpoint.tokenizer.add_special_tokens({'additional_special_tokens': ['[SW1]', '[/SW1]', '[SW2]', '[/SW2]']})
        checkpoint.model = build_attribute_model({'vocab_size': len(checkpoint.tokenizer), 'hidden_size': 8,
            'num_hidden_layers': 1, 'num_attention_heads': 2, 'intermediate_size': 16}, stage).eval()
    result = pipeline.predict(document)
    assert result['pipeline_complete'] and len(result['public_rows']) >= 2
    assert all(0 <= value <= 1 for row in result['field_predictions'] for value in row['sentiment']['scores'].values())


def test_schema_accepts_complete_partial_and_blocked_envelopes(pipeline_factory):
    import json
    import jsonschema
    schema = json.loads((Path(__file__).resolve().parents[1] / 'schemas/scibert-v2/full-label-prediction.schema.json').read_bytes())
    for kwargs in ({}, {'missing': ('sentiment',)}, {'detector_status': 'failure'}):
        pipeline, document = pipeline_factory(**kwargs)
        jsonschema.Draft202012Validator(schema).validate(pipeline.predict(document))


def test_validator_cannot_hide_a_detected_occurrence(pipeline_factory):
    from research.training.full_label import validate_full_label_prediction
    pipeline, document = pipeline_factory()
    result = pipeline.predict(document)
    result['field_predictions'].pop()
    result['occurrences'].pop()
    result['public_rows'].pop()
    result['public_mappings'].pop()
    result['alias_predictions'] = {'pairs': [], 'groups': []}
    with pytest.raises(ValueError):
        validate_full_label_prediction(result, document)


def test_detector_runtime_exception_returns_failed_envelope(pipeline_factory):
    pipeline, document = pipeline_factory()
    def fail(document):
        raise RuntimeError('detector crashed')
    pipeline.detector.predict = fail
    result = pipeline.predict(document)
    assert result['status'] == 'failure' and not result['public_rows']
    assert result['stage_status']['intent']['status'] == 'blocked'


def test_candidate_construction_failure_has_no_absence_or_attribute_defaults(pipeline_factory, monkeypatch):
    from research.training import full_label
    pipeline, document = pipeline_factory()
    def fail(*args):
        raise RuntimeError('candidate construction failed')
    monkeypatch.setattr(full_label, 'build_inference_candidates', fail)
    result = pipeline.predict(document)
    assert result['status'] == 'partial'
    assert result['stage_status']['detector']['status'] == 'success'
    assert all(result['field_predictions'][0][field]['value'] is None for field in ('versions', 'intents', 'sentiment'))
    assert not result['public_rows']


def test_excluded_alias_pair_abstains_without_claiming_consumed_model_context(pipeline_factory):
    text = 'NumPy ' + 'word ' * 510 + 'ToolX.'
    pipeline, document = pipeline_factory(text, ('NumPy', 'ToolX'), ())
    pair = pipeline.predict(document)['alias_predictions']['pairs'][0]
    assert pair['label'] is None and pair['scores'] is None
    assert pair['evidence_method'] is None


@pytest.mark.parametrize('linker_status', ['missing', 'failed'])
def test_incomplete_linker_resolves_absence_only_for_occurrence_without_eligible_pair(pipeline_factory, linker_status):
    pipeline, document = pipeline_factory('NumPy 1.24.\n\nToolX.', ('NumPy', 'ToolX'), ('1.24',),
        missing=('linker',) if linker_status == 'missing' else (),
        values={'linker': [RuntimeError('linker failed')]} if linker_status == 'failed' else None)
    result = pipeline.predict(document)
    assert result['field_predictions'][0]['versions']['value'] is None
    assert result['field_predictions'][1]['versions']['value'] == []
    assert [(row['name'], row['version']) for row in result['public_rows']] == [('ToolX', None)]
    assert not result['public_contract_complete'] and result['status'] == 'partial'


@pytest.mark.parametrize('mutation', ['linker_first', 'linker_second', 'alias_first', 'alias_second', 'linker_context', 'alias_context'])
def test_validator_binds_pair_spans_and_consumed_context_to_detected_endpoints(pipeline_factory, mutation):
    from research.training.full_label import validate_full_label_prediction
    pipeline, document = pipeline_factory(missing=('intent',))
    result = pipeline.predict(document)
    pair = result['pair_predictions']['linker'][0] if mutation.startswith('linker') else result['alias_predictions']['pairs'][0]
    if mutation.endswith('first'):
        pair['first_span'] = {'start': 14, 'end': 18}
    elif mutation.endswith('second'):
        pair['second_span'] = {'start': 8, 'end': 13}
    else:
        pair['context_text'] = 'invented model input'
    with pytest.raises(ValueError):
        validate_full_label_prediction(result, document)


def test_validator_binds_unavailable_linker_endpoint_in_detector_only_result(pipeline_factory):
    from research.training.full_label import validate_full_label_prediction
    pipeline, document = pipeline_factory('We used NumPy 1.24 and 1.26.', ('NumPy',), ('1.24', '1.26'),
        missing=('linker', 'intent', 'sentiment', 'alias'))
    result = pipeline.predict(document)
    assert validate_full_label_prediction(result, document) == result
    result['pair_predictions']['linker'][0]['first_span'] = {'start': 14, 'end': 18}
    with pytest.raises(ValueError):
        validate_full_label_prediction(result, document)


@pytest.mark.parametrize(('block_kind', 'context_kind'), [
    ('table', 'table'), ('table-wrap', 'table'), ('ref', 'reference'),
    ('ref-list', 'reference'), ('reference', 'reference')])
@pytest.mark.parametrize('metadata_location', ['paragraphs', 'metadata'])
def test_nonprose_diagnostic_context_preserves_containing_source_block(pipeline_factory, block_kind, context_kind, metadata_location):
    pipeline, document = pipeline_factory('Before prose.\n\nNumPy 1.24.\n\nAfter prose.', ('NumPy',), ('1.24',))
    paragraphs = [{'start': 0, 'end': 13, 'kind': 'p'},
                  {'start': 15, 'end': 26, 'kind': block_kind},
                  {'start': 28, 'end': 40, 'kind': 'p'}]
    if metadata_location == 'paragraphs':
        document['paragraphs'] = paragraphs
    else:
        document['metadata'] = {'paragraphs': paragraphs}
    result = pipeline.predict(document)
    occurrence = result['field_predictions'][0]
    assert occurrence['context_span'] == {'start': 15, 'end': 26}
    assert occurrence['context_sentence'] == 'NumPy 1.24.'
    assert occurrence['context_kind'] == context_kind
    assert all(occurrence[field]['value'] is None for field in ('versions', 'intents', 'sentiment'))
    assert all('unsupported_context' in occurrence[field]['reasons'] for field in ('intents', 'sentiment'))
    assert {row['stage'] for row in result['exclusions'] if row['reason'] == 'unsupported_context'} == {'linker', 'intent', 'sentiment'}
    assert result['status'] == 'partial' and not result['public_rows']
