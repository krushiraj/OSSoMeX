import hashlib
import json
from copy import deepcopy

import pytest

from research.contracts import occurrence_id, validate_document


def _timing_dir(tmp_path, native, *, checkpoint=None, text='same', config_hash=None, softcite=False):
    from research.comparison.pipeline_timing import _summary

    document = validate_document({'document_id': 'd', 'text': 'We used Alpha.'})
    if text != 'same':
        document['text'] = text
    checkpoint = checkpoint or 'b' * 64
    config = b'{}'
    config_hash = config_hash or hashlib.sha256(config).hexdigest()
    timing = tmp_path / 'timing'
    timing.mkdir()
    timing_arm = 'softcite' if softcite else 'different-timing-name'
    rows = [{'arm_id': timing_arm, 'repeat': 1, 'document_id': document['document_id'],
             'text_revision': document['text_revision'], 'input_sha256': document['text_revision'],
             'status': 'success', 'elapsed_seconds': 2.0, 'error': None}]
    first = [{'arm_id': timing_arm, 'document_id': document['document_id'],
              'input_sha256': document['text_revision'], 'native': native, 'measurement_error': None}]
    files = {'inputs.jsonl': (json.dumps(document) + '\n').encode(),
             'requests.jsonl': (json.dumps(rows[0]) + '\n').encode(),
             'first_pass.jsonl': (json.dumps(first[0]) + '\n').encode(),
             'softcite-config.json': config}
    for name, payload in files.items():
        (timing / name).write_bytes(payload)
    manifest = {'schema_version': 'pipeline-timing-1', 'seed': 42, 'repeats': 1,
                'batch_size': 1, 'order': [document['document_id']], 'document_count': 1,
                'input_file_sha256': hashlib.sha256(files['inputs.jsonl']).hexdigest(),
                'softcite_config_sha256': config_hash,
                'hardware': {'device': 'cpu', 'machine': 'arm64'},
                'timing_policy': 'full document predict; load and warmup excluded',
                'arms': [{'arm_id': timing_arm,
                          'checkpoint_manifest_sha256': None if softcite else checkpoint,
                          'load_seconds': 1.0, 'load_semantics': 'service_health_check_only' if softcite else 'local_checkpoint_initialization',
                          'load_status': 'ready', 'summary': _summary(rows)}],
                'files': [{'path': name, 'sha256': hashlib.sha256(payload).hexdigest()}
                          for name, payload in files.items()]}
    (timing / 'manifest.json').write_text(json.dumps(manifest))
    return timing


def test_timing_joins_by_checkpoint_and_copies_verified_evidence(tmp_path):
    from research.comparison.html_report import _timing_evidence

    native = {'document_id': 'd', 'text_revision': validate_document({'document_id': 'd', 'text': 'We used Alpha.'})['text_revision'],
              'run_id': 'timing-run'}
    timing = _timing_dir(tmp_path, native)
    docs = [validate_document({'document_id': 'd', 'text': 'We used Alpha.'})]
    models = [{'arm_id': 'full-004', 'kind': 'pipeline', 'identity': {'checkpoint_sha256': 'b' * 64}, 'timing': None}]
    frozen = {}
    _timing_evidence([timing], docs, models,
                     {('full-004', 'd'): {**native, 'run_id': 'scored-run'}}, {}, frozen)
    pipeline = models[0]
    assert pipeline['timing'][0]['summary']['latency_median_seconds'] == 2.0
    assert pipeline['timing'][0]['summary']['successful_per_second'] == 0.5
    assert pipeline['timing'][0]['device'] == 'cpu'
    assert frozen['timing-0/requests.jsonl'] == (timing / 'requests.jsonl').read_bytes()
    assert pipeline['timing'][0]['timing_arm_id'] == 'different-timing-name'


def test_softcite_timing_requires_exact_scored_config_hash(tmp_path):
    from research.comparison.html_report import _timing_evidence

    doc = validate_document({'document_id': 'd', 'text': 'We used Alpha.'})
    timing = _timing_dir(tmp_path, {'window_id': 'd', 'status': 'success'}, softcite=True)
    model = {'arm_id': 'scored-softcite', 'kind': 'softcite', 'identity': {}, 'timing': None}
    with pytest.raises(ValueError, match='config|unmatched'):
        _timing_evidence([timing], [doc], [model], {},
                         {'scored-softcite': {'config_source': {'sha256': 'c' * 64}}}, {})
    frozen = {}
    _timing_evidence([timing], [doc], [model], {},
                     {'scored-softcite': {'config_source': {
                         'sha256': hashlib.sha256(b'{}').hexdigest()}}}, frozen)
    assert model['timing'][0]['load_semantics'] == 'service_health_check_only'


def test_ready_arm_keeps_first_pass_when_presynchronization_prevents_attempt(tmp_path):
    from research.comparison.html_report import _timing_evidence
    from research.comparison.pipeline_timing import _summary

    doc = validate_document({'document_id': 'd', 'text': 'We used Alpha.'})
    native = {'document_id': 'd', 'text_revision': doc['text_revision']}
    timing = _timing_dir(tmp_path, native)
    request_path = timing / 'requests.jsonl'
    row = json.loads(request_path.read_text())
    row.update(status='failure', elapsed_seconds=None, error='presynchronization RuntimeError')
    request_path.write_text(json.dumps(row) + '\n')
    manifest_path = timing / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['arms'][0]['summary'] = _summary([row])
    next(item for item in manifest['files'] if item['path'] == 'requests.jsonl')['sha256'] = hashlib.sha256(request_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    model = {'arm_id': 'full', 'kind': 'pipeline', 'identity': {'checkpoint_sha256': 'b' * 64}, 'timing': None}
    _timing_evidence([timing], [doc], [model], {('full', 'd'): native}, {}, {})
    assert model['timing'][0]['summary']['not_attempted'] == 1


@pytest.mark.parametrize('damage,match', [
    ('checkpoint', 'checkpoint|unmatched'),
    ('text', 'document|population|text'),
    ('row', 'summary|request|row'),
    ('hash', 'hash'),
])
def test_timing_rejects_mismatched_or_tampered_evidence(tmp_path, damage, match):
    from research.comparison.html_report import _timing_evidence

    native = {'document_id': 'd', 'text_revision': validate_document({'document_id': 'd', 'text': 'We used Alpha.'})['text_revision']}
    timing = _timing_dir(tmp_path, native, checkpoint='a' * 64 if damage == 'checkpoint' else None,
                         text='wrong text' if damage == 'text' else 'same')
    if damage in ('row', 'hash'):
        path = timing / 'requests.jsonl'
        row = json.loads(path.read_text())
        row['elapsed_seconds'] = 200.0
        path.write_text(json.dumps(row) + '\n')
        if damage == 'row':
            manifest_path = timing / 'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            next(record for record in manifest['files'] if record['path'] == 'requests.jsonl')['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        _timing_evidence([timing], [validate_document({'document_id': 'd', 'text': 'We used Alpha.'})],
                         [{'arm_id': 'full-004', 'kind': 'pipeline',
                           'identity': {'checkpoint_sha256': 'b' * 64}, 'timing': None}],
                         {('full-004', 'd'): native}, {}, {})


def test_fields_score_only_selected_review_and_keep_unsupported_na():
    from research.comparison.html_report import _field_evidence

    doc = validate_document({'document_id': 'd', 'text': 'R'})
    gold = {'schema_version': '2.0', 'document_id': 'd', 'text_revision': doc['text_revision'], 'name': 'R',
            'name_span': {'start': 0, 'end': 1}, 'version_links': [],
            'mention_id': occurrence_id('d', doc['text_revision'], 0, 1),
            'context_sentence': 'R', 'context_span': {'start': 0, 'end': 1},
            'context_kind': 'paragraph', 'version_status': 'absent',
            'intents': ['used'], 'sentiment': 'not_expressed',
            'evidence': {'intents': [{'start': 0, 'end': 1}], 'sentiment': []},
            'known': {'software': True, 'versions': True, 'created': True,
                      'used': True, 'shared': True, 'sentiment': True},
            'review_kind': 'agent_provisional'}
    coverage = {'document_id': 'd', 'text_revision': doc['text_revision'],
                'start': 0, 'end': 1,
                'fields': {'software': True, 'versions': True, 'created': True,
                           'used': True, 'shared': True, 'sentiment': True},
                'review_kind': 'agent_provisional'}
    refs = [{**doc, 'attribute_occurrences': [gold], 'attribute_coverage': [coverage], 'alias_pairs': []}]
    native = {'document_id': 'd', 'text_revision': doc['text_revision'], 'status': 'success',
              'capabilities': {'software_spans': True, 'version_linking': True,
                               'intent': True, 'sentiment': True, 'aliases': True},
              'field_predictions': [{'mention_id': 'm1', 'name': 'R', 'name_span': {'start': 0, 'end': 1},
                                     'versions': {'status': 'success', 'value': []},
                                     'intents': {'status': 'success', 'value': ['used']},
                                     'sentiment': {'status': 'success', 'value': 'not_expressed'}}],
              'alias_predictions': {'groups': []}}
    models = [{'arm_id': 'full', 'kind': 'pipeline', 'identity': {}, 'status': 'complete'},
              {'arm_id': 'soft', 'kind': 'softcite', 'identity': {'offset_unit': 'codepoint'}, 'status': 'ready'}]
    mention = {'software-name': {'rawForm': 'R', 'offsetStart': 0, 'offsetEnd': 1},
               'mentionContextAttributes': {k: {'value': k == 'used'} for k in ('created', 'used', 'shared')},
               'documentContextAttributes': {k: {'value': True} for k in ('created', 'used', 'shared')}}
    result = {'arm_id': 'soft', 'document_id': 'd', 'status': 'success', 'chunks': [
        {'window': {'text': 'R', 'start': 0}, 'raw': {'native': {'mentions': [mention]}}}]}
    scores = _field_evidence([doc], refs, models, {('full', 'd'): native},
                             {('soft', 'd'): result}, 'agent_provisional')
    assert scores['full']['intents']['per_label']['used']['tp'] == 1
    assert scores['full']['sentiment']['per_class']['not_expressed']['tp'] == 1
    assert scores['full']['intents']['macro']['f1'] is None
    assert scores['full']['intents']['missing_classes']
    assert scores['soft']['intents']['per_label']['used']['tp'] == 1
    assert scores['soft']['sentiment']['macro_f1'] is None
    assert scores['soft']['complete_occurrence']['f1'] is None
    human = _field_evidence([doc], refs, models, {('full', 'd'): native},
                            {('soft', 'd'): result}, 'human_reviewed')
    assert human['full']['intents']['per_label']['used']['tp'] == 0
    assert human['full']['intents']['per_label']['used']['f1'] is None
    native['capabilities'].update(intent=False, sentiment=False, aliases=False, version_linking=False)
    unsupported = _field_evidence([doc], refs, models, {('full', 'd'): native},
                                  {('soft', 'd'): result}, 'agent_provisional')['full']
    assert unsupported['intents']['per_label']['used']['f1'] is None
    assert unsupported['sentiment']['observed_macro_f1'] is None
    assert unsupported['version']['strict_edges']['f1'] is None
    assert unsupported['aliases']['supported'] is False
    assert unsupported['complete_occurrence']['f1'] is None


def test_optional_tables_escape_user_values_and_explain_timing_semantics():
    from research.comparison.html_render import render_dashboard
    from test_comparison_html_render import sample_data

    data = sample_data()
    data['timing_rows'] = [{'arm_id': '<script>bad</script>', 'timing_arm_id': 'measured',
                            'device': 'cpu', 'hardware': {'machine': 'arm64'},
                            'load_seconds': 1.0, 'load_semantics': 'service_health_check_only',
                            'summary': {'scheduled': 2, 'successful': 1, 'failed': 1,
                                        'latency_median_seconds': 2.0, 'latency_p95_seconds': 3.0,
                                        'successful_per_second': 0.25}}]
    data['field_scores'] = {'<script>bad</script>': {
        'intents': {'per_label': {'used': {'tp': 1, 'fp': 0, 'fn': 0, 'f1': 1.0}},
                'macro': {'f1': None}, 'observed_macro_f1': 1.0, 'exact_set_accuracy': 1.0,
                    'missing_classes': ['created']},
        'sentiment': {'per_class': {}, 'macro_f1': None, 'observed_macro_f1': None,
                      'missing_classes': ['positive']},
        'complete_occurrence': {'f1': None}, 'aliases': {'f1': None,
            'reviewed_pairs': 0, 'unreviewed_positive_predictions': 2}}}
    page = render_dashboard(data)
    assert '<script>bad</script>' not in page
    assert '&lt;script&gt;bad&lt;/script&gt;' in page
    assert 'Full-pipeline timing' in page
    assert 'service health' in page.lower()
    assert 'Resident service; client cpu / arm64' in page
    assert 'does not guarantee every attribute head' in page
    assert 'Observed-class macro' in page
    assert 'Support' in page
    assert 'Sentiment observed-class macro' in page
    assert 'unreviewed positives' in page


def test_unavailable_softcite_unicode_needs_no_offset_conversion():
    from research.comparison.html_report import _field_evidence

    doc = validate_document({'document_id': 'd', 'text': '🧪 R'})
    refs = [{**doc, 'attribute_occurrences': [], 'attribute_coverage': [], 'alias_pairs': []}]
    models = [{'arm_id': 'soft', 'kind': 'softcite', 'identity': {}, 'status': 'unavailable'}]
    results = {('soft', 'd'): {'document_id': 'd', 'status': 'unavailable', 'chunks': []}}
    scores = _field_evidence([doc], refs, models, {}, results, 'agent_provisional')
    assert scores['soft']['failures']['failed_documents'] == 1


def test_ignored_version_endpoint_masks_complete_fields_without_mutating_references():
    from research.contracts import FIELDS
    from research.comparison.html_report import _field_evidence

    docs = [validate_document({'document_id': ident, 'text': text})
            for ident, text in [('ambiguous', 'A B 7'), ('control', 'R 4')]]
    refs, natives = [], {}
    for doc in docs:
        name = doc['text'][0]
        links = [] if doc['document_id'] == 'ambiguous' else [
            {'text': '4', 'span': {'start': 2, 'end': 3}, 'status': 'explicit_local'}]
        gold = {'schema_version': '2.0', 'document_id': doc['document_id'], 'text_revision': doc['text_revision'],
                'mention_id': occurrence_id(doc['document_id'], doc['text_revision'], 0, 1),
                'name': name, 'name_span': {'start': 0, 'end': 1}, 'context_sentence': doc['text'],
                'context_span': {'start': 0, 'end': len(doc['text'])}, 'context_kind': 'paragraph',
                'version_links': links, 'version_status': 'explicit' if links else 'absent',
                'intents': ['mentioned'], 'sentiment': 'not_expressed', 'known': dict.fromkeys(FIELDS, True),
                'evidence': {'intents': [], 'sentiment': []}, 'review': {'status': 'agent_provisional'}}
        ref = {**doc, 'attribute_occurrences': [gold], 'alias_pairs': [], 'ignored_versions': [],
               'attribute_coverage': [{'document_id': doc['document_id'], 'text_revision': doc['text_revision'],
                   'start': 0, 'end': len(doc['text']), 'fields': dict.fromkeys(FIELDS, True),
                   'review_kind': 'agent_provisional'}]}
        names = [(name, 0)]
        if doc['document_id'] == 'ambiguous':
            ref['ignored_versions'] = [{'start': 4, 'end': 5, 'text': '7'}]
            names.append(('B', 2))
            links = [{'text': '7', 'span': {'start': 4, 'end': 5}}]
        refs.append(ref)
        natives[('full', doc['document_id'])] = {**doc, 'status': 'success',
            'capabilities': dict.fromkeys(('software_spans', 'version_linking', 'intent', 'sentiment', 'aliases'), True),
            'field_predictions': [{'name': n, 'name_span': {'start': start, 'end': start + 1},
                'versions': {'status': 'success', 'value': links},
                'intents': {'status': 'success', 'value': ['mentioned']},
                'sentiment': {'status': 'success', 'value': 'not_expressed'}} for n, start in names]}
    before = deepcopy(refs)
    scores = _field_evidence(docs, refs, [{'arm_id': 'full', 'kind': 'pipeline'}], natives, {}, 'agent_provisional')['full']
    assert refs == before
    assert scores['version']['strict_edges']['tp'] == 1
    assert scores['version']['strict_edges']['fp'] == 0
    assert scores['complete_occurrence']['tp'] == 1
    assert scores['complete_occurrence']['fp'] == 0
    assert scores['mention_detection']['fp'] == 1
    assert scores['version_field_masked_documents'] == ['ambiguous']


def test_source_only_wording_requires_explicit_flags_on_every_reference():
    from research.comparison.html_report import _reference_limit

    source_only = {'provenance': {'source_only': True, 'prediction_exposed': False}}
    assert 'source-only fresh diagnostic sample' in _reference_limit([source_only, source_only]).lower()
    assert 'Development diagnostic' in _reference_limit([source_only, {'provenance': {}}])
