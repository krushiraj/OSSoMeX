import argparse
from copy import deepcopy
import json

import pytest

from test_full_label_prediction import pipeline_factory


def artifacts(tmp_path, pipeline_factory, **pipeline_kwargs):
    from research.comparison.backends import capabilities
    from research.comparison.runner import run_comparison
    from research.comparison.report import build_report
    from research.comparison.link_report import build_link_report
    from research.contracts import validate_document

    pipeline, doc = pipeline_factory(**pipeline_kwargs)
    doc = validate_document(doc)
    native = pipeline.predict(doc)
    native['capabilities']['version_linking'] = True
    spans = [{**s, 'score': None, 'score_kind': None, 'alignment_method': 'native_codepoint'}
             for s in native['detector_diagnostics']['spans']]

    class Backend:
        def load(self, arm):
            return {'status': 'ready', 'capabilities': capabilities(),
                    'identity': {'checkpoint_sha256': 'd' * 64, 'verified': True}}

        def predict(self, window):
            return {'window_id': window['window_id'], 'status': 'success' if window['document_id'] == 'd' else 'no_mentions',
                    'spans': spans if window['document_id'] == 'd' else [], 'unresolved': [], 'raw': {}, 'error': None}

        def close(self):
            pass

    run = tmp_path / 'run'
    window = {**doc, 'window_id': 'w', 'window_text_revision': doc['text_revision'],
              'start': 0, 'end': len(doc['text']), 'content_tokens': 12}
    run_comparison([doc], [window], [{'arm_id': 'detector-unrelated-name', 'backend': 'scibert', 'config': {}}],
                   run, adapter_factory=lambda arm: Backend())
    reference = {**doc, 'spans': spans,
                 'coverage': [{'start': 0, 'end': len(doc['text']), 'label': label,
                               'complete': True, 'review_kind': 'agent_provisional'} for label in ('SOFTWARE', 'VERSION')],
                 'version_links': [{'software': spans[0], 'version': spans[2]}],
                 'ignored_versions': [spans[3]], 'ignored_software': [],
                 'link_coverage': [{'start': 0, 'end': len(doc['text']), 'complete': True,
                                    'review_kind': 'agent_provisional'}],
                 'provenance': {'review_kind': 'agent_provisional', 'role': 'diagnostic', 'prediction_exposed': True}}
    refs = tmp_path / 'references.jsonl'
    refs.write_text(json.dumps(reference) + '\n')
    full = tmp_path / 'full.jsonl'
    full.write_text(json.dumps(native) + '\n')
    build_report(run, tmp_path / 'spans', references=refs)
    build_link_report(run, refs, ['full-004=' + str(full)], tmp_path / 'links')
    return run, native


def build(tmp_path, **kwargs):
    from research.comparison.html_report import build_html_report
    return build_html_report(tmp_path / 'run', tmp_path / 'spans', [tmp_path / 'links'],
                             tmp_path / 'html', **kwargs)


def test_verified_pipeline_identity_join_masks_and_offline_publication(tmp_path, pipeline_factory):
    artifacts(tmp_path, pipeline_factory)
    data = build(tmp_path)
    model = next(m for m in data['models'] if m['arm_id'] == 'full-004')
    assert model['span_source_arm'] == 'detector-unrelated-name'
    assert model['metrics']['software']['f1'] == 1
    assert model['metrics']['links']['tp'] == 1
    assert model['excluded_predictions'] == 1
    assert model['timing'] is None
    assert data['provenance']['review_kind'] == 'agent_provisional'
    assert data['documents'][0]['ignored_versions'][0]['text'] == '2.26'
    assert data['documents'][0]['arms'][-1]['fields']
    assert (tmp_path / 'html/report.html').is_file()
    manifest = json.loads((tmp_path / 'html/manifest.json').read_bytes())
    from research.comparison.runner import verify_files
    verify_files(tmp_path / 'html', manifest['files'])
    from research.comparison.report import _verified_run
    _verified_run(tmp_path / 'html/sources/run')
    with pytest.raises(FileExistsError):
        build(tmp_path)


@pytest.mark.parametrize('target', ['run/results.jsonl', 'spans/report.json', 'links/predictions.jsonl', 'links/full-label-0.jsonl'])
def test_tampered_inputs_fail_before_output(tmp_path, pipeline_factory, target):
    artifacts(tmp_path, pipeline_factory)
    (tmp_path / target).write_text('{}\n')
    with pytest.raises(ValueError, match='hash'):
        build(tmp_path)
    assert not (tmp_path / 'html').exists()


def test_sidecar_inventory_and_run_identity_fail_closed(tmp_path, pipeline_factory):
    artifacts(tmp_path, pipeline_factory)
    (tmp_path / 'links/unlisted.txt').write_text('not frozen')
    with pytest.raises(ValueError, match='inventory'):
        build(tmp_path)


def test_human_metrics_do_not_inherit_provisional_scores(tmp_path, pipeline_factory):
    artifacts(tmp_path, pipeline_factory)
    data = build(tmp_path, review_kind='human_reviewed')
    assert all(model['metrics']['software']['f1'] is None for model in data['models'])
    assert all(model['metrics']['links']['f1'] is None for model in data['models'])
    assert not data['documents'][0]['reference_spans']
    assert not data['documents'][0]['reference_links']
    assert not any('names 0 correct' in line for line in data['documents'][0]['summaries'])


def test_cli_regenerates_report_without_loading_models(tmp_path, pipeline_factory, capsys):
    artifacts(tmp_path, pipeline_factory)
    from research.comparison.cli import register
    parser = argparse.ArgumentParser()
    register(parser.add_subparsers(required=True))
    args = parser.parse_args(['comparison', 'html', '--run', str(tmp_path / 'run'),
        '--spans', str(tmp_path / 'spans'), '--links', str(tmp_path / 'links'), '--output', str(tmp_path / 'html')])
    assert args.func(args) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'reported'


def test_nonempty_missed_edges_roundtrip_through_json(tmp_path, pipeline_factory):
    artifacts(tmp_path, pipeline_factory)
    from research.comparison.link_report import build_link_report
    from research.comparison.html_report import build_html_report
    from research.comparison.report import build_report
    refs = tmp_path / 'references.jsonl'
    reference = json.loads(refs.read_text())
    reference['version_links'].append({'software': reference['spans'][1], 'version': reference['spans'][2]})
    refs = tmp_path / 'changed-reference.jsonl'
    refs.write_text(json.dumps(reference) + '\n')
    build_report(tmp_path / 'run', tmp_path / 'new-spans', references=refs)
    build_link_report(tmp_path / 'run', refs, ['full-004=' + str(tmp_path / 'full.jsonl')], tmp_path / 'new-links')
    data = build_html_report(tmp_path / 'run', tmp_path / 'new-spans', [tmp_path / 'new-links'], tmp_path / 'html')
    assert data['models'][-1]['metrics']['links']['fn'] == 1


def test_observed_appendix_must_match_probe_receipt(tmp_path, pipeline_factory):
    artifacts(tmp_path, pipeline_factory)
    appendix = tmp_path / 'appendix.json'
    appendix.write_text(json.dumps({'examples': [{'kind': 'observed_base_masked_token', 'input': 'example',
                                                'output': {'top5_tokens': []}}]}))
    with pytest.raises(ValueError, match='probe'):
        build(tmp_path, appendix=appendix)
    assert not (tmp_path / 'html').exists()


def test_repeated_link_sources_are_deduplicated(tmp_path, pipeline_factory):
    artifacts(tmp_path, pipeline_factory)
    from research.comparison.html_report import build_html_report
    data = build_html_report(tmp_path / 'run', tmp_path / 'spans', [tmp_path / 'links'] * 2, tmp_path / 'html')
    assert [m['arm_id'] for m in data['models']] == ['detector-unrelated-name', 'full-004']
    assert data['models'][-1]['metrics']['links']['tp'] == 1


def test_partial_pipeline_is_not_complete_just_because_linking_succeeded(tmp_path, pipeline_factory):
    artifacts(tmp_path, pipeline_factory, missing=('sentiment',))
    data = build(tmp_path)
    model = data['models'][-1]
    assert model['metrics']['links']['tp'] == 1
    assert model['status'] == 'partial'
    assert data['documents'][0]['arms'][-1]['status'] == 'partial'


def test_failed_operational_arm_is_not_reported_ready(tmp_path):
    from test_comparison_runner import execute, fixture_inputs
    from research.comparison.report import build_report
    from research.comparison.link_report import build_link_report
    execute(tmp_path, 'failing')
    docs, _ = fixture_inputs()
    refs = [{**d, 'spans': [], 'coverage': [], 'version_links': [], 'link_coverage': [],
             'provenance': {'review_kind': 'agent_provisional'}} for d in docs]
    path = tmp_path / 'refs.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in refs))
    build_report(tmp_path / 'run', tmp_path / 'spans', references=path)
    build_link_report(tmp_path / 'run', path, [], tmp_path / 'links')
    data = build(tmp_path)
    assert next(m for m in data['models'] if m['arm_id'] == '../../b')['status'] == 'partial'
