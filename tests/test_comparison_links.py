from copy import deepcopy
import json

import pytest

from research.contracts import text_revision
from test_full_label_prediction import pipeline_factory


def fixture():
    text = 'Alpha 1.0 and Beta 2.0.'
    doc = {'document_id': 'd', 'text': text, 'text_revision': text_revision(text)}
    spans = [{'start': a, 'end': b, 'text': text[a:b], 'label': label}
             for label, a, b in [('SOFTWARE', 0, 5), ('VERSION', 6, 9),
                                 ('SOFTWARE', 14, 18), ('VERSION', 19, 22)]]
    links = [{'software': spans[0], 'version': spans[1]},
             {'software': spans[2], 'version': spans[3]}]
    ref = {**doc, 'spans': spans, 'coverage': [], 'version_links': links,
           'link_coverage': [{'start': 0, 'end': len(text), 'complete': True,
                              'review_kind': 'agent_provisional'}],
           'ignored_versions': [], 'provenance': {'review_kind': 'agent_provisional'}}
    pred = {'document_id': 'd', 'text_revision': doc['text_revision'], 'arm_id': 'model',
            'status': 'success', 'supported': True, 'spans': spans, 'version_links': links}
    return doc, ref, pred


def score(docs, refs, preds, kind='agent_provisional'):
    from research.comparison.links import score_links
    return score_links(docs, refs, preds, review_kind=kind)['arms']['model']


def test_wrong_owner_is_fp_and_fn_even_with_perfect_version_detection():
    doc, ref, pred = fixture()
    pred['version_links'] = [ref['version_links'][0],
                            {'software': ref['spans'][0], 'version': ref['spans'][3]}]
    scored = score([doc], [ref], [pred])
    assert scored['operational'] == {'tp': 1, 'fp': 1, 'fn': 1,
                                     'precision': .5, 'recall': .5, 'f1': .5}
    assert scored['conditional_on_detected_endpoints'] == scored['operational']


def test_detected_endpoint_diagnostic_does_not_hide_missed_name_in_primary():
    doc, ref, pred = fixture()
    pred['spans'] = pred['spans'][:2]
    pred['version_links'] = pred['version_links'][:1]
    scored = score([doc], [ref], [pred])
    assert (scored['operational']['tp'], scored['operational']['fn']) == (1, 1)
    assert scored['conditional_on_detected_endpoints']['f1'] == 1
    assert scored['conditional_gold_edges'] == 1


def test_duplicate_edges_are_deduplicated_and_multiple_versions_are_allowed():
    doc, ref, pred = fixture()
    edge = {'software': ref['spans'][0], 'version': ref['spans'][3]}
    ref['version_links'].append(edge)
    pred['version_links'] = ref['version_links'] * 2
    assert score([doc], [ref], [pred])['operational']['tp'] == 3


def test_unknown_version_ownership_does_not_become_negative_and_review_kinds_stay_separate():
    doc, ref, pred = fixture()
    ref['ignored_versions'] = [ref['spans'][3]]
    ref['version_links'] = ref['version_links'][:1]
    pred['version_links'].append({'software': ref['spans'][0], 'version': ref['spans'][3]})
    assert score([doc], [ref], [pred])['operational']['fp'] == 0
    assert score([doc], [ref], [pred], 'human_reviewed')['operational']['f1'] is None


def test_positive_only_and_cross_region_edges_are_not_precision_evidence():
    doc, ref, pred = fixture()
    ref['link_coverage'] = [{'start': 0, 'end': 5, 'complete': True,
                            'review_kind': 'agent_provisional'}]
    assert score([doc], [ref], [pred])['operational']['f1'] is None


def test_unknown_introductory_name_ownership_is_not_checked_absence():
    doc, ref, pred = fixture()
    ref['ignored_software'] = [ref['spans'][0]]
    pred['version_links'] = [{'software': ref['spans'][0], 'version': ref['spans'][3]}]
    result = score([doc], [ref], [pred])
    assert result['operational']['fp'] == 0
    assert result['operational']['fn'] == 1


@pytest.mark.parametrize('status,supported', [('unavailable', True), ('unsupported', False)])
def test_inactive_arms_are_na_not_zero(status, supported):
    doc, ref, pred = fixture()
    pred.update(status=status, supported=supported, spans=[], version_links=[])
    assert score([doc], [ref], [pred])['operational']['fn'] is None


def test_failed_or_missing_document_is_operational_miss_once_arm_started():
    doc, ref, pred = fixture()
    other = {**doc, 'document_id': 'other'}
    other_ref = {**ref, 'document_id': 'other'}
    pred.update(status='failure', spans=[], version_links=[])
    result = score([doc, other], [ref, other_ref], [pred])
    assert result['operational']['fn'] == 4
    assert result['completed_documents'] == 0


def test_failed_linker_retains_detected_endpoint_denominator():
    doc, ref, pred = fixture()
    pred.update(status='failure', version_links=[])
    result = score([doc], [ref], [pred])
    assert result['conditional_gold_edges'] == 2
    assert result['conditional_on_detected_endpoints']['fn'] == 2


@pytest.mark.parametrize('bad', ['revision', 'slice', 'missing_endpoint', 'duplicate_doc',
                                'duplicate_pred', 'coverage', 'unknown_document', 'hidden_test'])
def test_invalid_reference_or_population_fails_closed(bad):
    doc, ref, pred = fixture()
    docs, refs, preds = [doc], [ref], [pred]
    if bad == 'revision': ref['text_revision'] = 'stale'
    if bad == 'slice': ref['version_links'] = [deepcopy(ref['version_links'][0])]; ref['version_links'][0]['version']['text'] = '3.0'
    if bad == 'missing_endpoint': ref['spans'] = ref['spans'][:2]
    if bad == 'duplicate_doc': docs.append(doc)
    if bad == 'duplicate_pred': preds.append(pred)
    if bad == 'coverage': ref['link_coverage'][0]['complete'] = False
    if bad == 'unknown_document': pred['document_id'] = 'other'
    if bad == 'hidden_test': ref['provenance']['role'] = 'test'
    with pytest.raises(ValueError): score(docs, refs, preds)


def test_softcite_preserves_native_ownership_and_utf16_nonzero_window_offsets():
    from research.comparison.links import softcite_links
    text = '😀 Alpha 1.0'
    native = {'software': [{'software-name': {'rawForm': 'Alpha', 'offsetStart': 3, 'offsetEnd': 8},
                            'version': {'rawForm': '1.0', 'offsetStart': 9, 'offsetEnd': 12}}]}
    chunk = {'window': {'start': 7, 'text': text}, 'raw': {'native': native}}
    links = softcite_links([chunk], 'utf16')
    assert links == [{'software': {'start': 9, 'end': 14, 'text': 'Alpha'},
                      'version': {'start': 15, 'end': 18, 'text': '1.0'}}]


def test_llm_preserves_record_ownership_with_exact_context_alignment():
    from research.comparison.links import ollama_links
    records = [{'name': 'Alpha', 'version': '1.0', 'context_sentence': 'Alpha 1.0.',
                'intents': ['used'], 'sentiment': 'not_expressed'}]
    chunk = {'window': {'start': 7, 'text': 'Alpha 1.0. Alpha 2.0.'},
             'raw': {'content': '```json\n' + json.dumps(records) + '\n```'}}
    assert ollama_links([chunk]) == [{'software': {'start': 7, 'end': 12, 'text': 'Alpha'},
                                     'version': {'start': 13, 'end': 16, 'text': '1.0'}}]
    records[0]['context_sentence'] = None
    chunk['raw']['content'] = json.dumps(records)
    with pytest.raises(ValueError): ollama_links([chunk])


def test_invalid_predicted_link_fails_document_instead_of_cherry_picking():
    doc, ref, pred = fixture()
    pred['version_links'] = deepcopy(pred['version_links'])
    pred['version_links'][1]['version']['text'] = 'wrong'
    result = score([doc], [ref], [pred])
    assert result['operational']['tp'] == 0
    assert result['operational']['fn'] == 2
    assert result['invalid_documents'] == 1


def test_cli_registers_link_report_with_native_predictions(tmp_path):
    import argparse
    from research.comparison.cli import register
    parser = argparse.ArgumentParser()
    register(parser.add_subparsers(required=True))
    args = parser.parse_args(['comparison', 'links', '--run', 'r', '--references', 'g',
                                     '--full-label', 'model=p.jsonl', '--output', str(tmp_path / 'out')])
    assert callable(args.func)
    assert args.full_label == ['model=p.jsonl']


def test_full_label_projection_preserves_links_despite_unrelated_missing_sentiment(pipeline_factory):
    from research.comparison.link_report import full_label_rows
    from research.contracts import validate_document
    pipeline, doc = pipeline_factory(missing=('sentiment',))
    native = pipeline.predict(doc)
    native['capabilities']['version_linking'] = True
    rows, identity = full_label_rows([validate_document(doc)], [native], 'full')
    assert not native['occurrences']
    assert rows[0]['status'] == 'success'
    assert [(e['software']['text'], e['version']['text']) for e in rows[0]['version_links']] == [
        ('NumPy', '1.24'), ('ToolX', '2.26')]
    assert identity == native['checkpoint_sha256']


def test_full_label_projection_marks_failed_linker_and_rejects_partial_population(pipeline_factory):
    from research.comparison.link_report import full_label_rows
    from research.contracts import validate_document
    pipeline, doc = pipeline_factory(values={'linker': [[float('nan')]]})
    doc = validate_document(doc)
    native = pipeline.predict(doc)
    native['capabilities']['version_linking'] = True
    rows, _ = full_label_rows([doc], [native], 'full')
    assert rows[0]['status'] == 'failure'
    assert rows[0]['version_links'] == []
    assert len(rows[0]['spans']) == 4
    with pytest.raises(ValueError, match='population'):
        full_label_rows([doc, {**doc, 'document_id': 'other'}], [native], 'full')


def test_link_report_roundtrip_verifies_run_and_publishes_native_ownership(tmp_path):
    from research.comparison.backends import capabilities
    from research.comparison.runner import run_comparison
    from research.comparison.link_report import build_link_report
    doc, ref, _ = fixture()
    spans = [{**s, 'score': None, 'score_kind': None, 'alignment_method': 'native_codepoint'} for s in ref['spans']]
    class NativeBackend:
        def load(self, arm):
            return {'status': 'ready', 'capabilities': capabilities(), 'identity': {'offset_unit': 'codepoint'}}
        def predict(self, window):
            if window['document_id'] != 'd':
                return {'window_id': window['window_id'], 'status': 'no_mentions', 'spans': [],
                        'unresolved': [], 'raw': {'native': {'software': []}}, 'error': None}
            mentions = [{'software-name': {'rawForm': s['text'], 'offsetStart': s['start'], 'offsetEnd': s['end']},
                         'version': {'rawForm': v['text'], 'offsetStart': v['start'], 'offsetEnd': v['end']}}
                        for s, v in [(spans[0], spans[1]), (spans[2], spans[3])]]
            return {'window_id': window['window_id'], 'status': 'success', 'spans': spans,
                    'unresolved': [], 'raw': {'native': {'software': mentions}}, 'error': None}
        def close(self): pass
    window = {**doc, 'window_id': 'w', 'window_text_revision': doc['text_revision'],
              'start': 0, 'end': len(doc['text']), 'content_tokens': 7}
    run = tmp_path / 'run'
    run_comparison([doc], [window], [{'arm_id': 'softcite', 'backend': 'softcite', 'config': {}}], run,
                   adapter_factory=lambda arm: NativeBackend())
    refs = tmp_path / 'refs.jsonl'
    refs.write_text(json.dumps(ref) + '\n')
    report = build_link_report(run, refs, [], tmp_path / 'report')
    assert report['references']['agent_provisional']['arms']['softcite']['operational']['tp'] == 2
    assert report['references']['human_reviewed']['arms']['softcite']['operational']['f1'] is None
    assert (tmp_path / 'report/manifest.json').is_file()
    with pytest.raises(FileExistsError): build_link_report(run, refs, [], tmp_path / 'report')
    (run / 'results.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='hash'): build_link_report(run, refs, [], tmp_path / 'tampered')
    assert not (tmp_path / 'tampered').exists()
