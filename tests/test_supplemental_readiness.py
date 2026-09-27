import importlib
import json
import re
import subprocess
import sys

import pytest

from research.annotations import review_store
from research.annotations.snapshots import read_policy_provenance
from research.data.manifest import digest, json_bytes, read_jsonl
from review_workflow_fixtures import batch
from test_supplemental_annotation import acquired, imported


class LocalTokenizer:
    is_fast = True
    do_lower_case = False
    def num_special_tokens_to_add(self, pair=False):
        return 2
    def build_inputs_with_special_tokens(self, ids):
        return [101, *ids, 102]
    def __call__(self, text, **kwargs):
        spans = [(m.start(), m.end()) for m in re.finditer(r'\w+|[^\w\s]', text)]
        return {'input_ids': list(range(len(spans))), 'offset_mapping': spans}


def snapshot(monkeypatch, tmp_path, *, human=False, versions=False, sources=('europepmc',)):
    bundle, _, imported_path = imported(monkeypatch, tmp_path, sources=sources, versions=versions)
    meta = json.loads((imported_path / 'manifest.json').read_bytes())
    items = read_jsonl(imported_path / 'items.jsonl')
    c = review_store.open_store(tmp_path / 'review.sqlite')
    try:
        review_store.import_items(c, items, 'train', read_policy_provenance(imported_path, meta))
        if human:
            for index, item in enumerate(items):
                current = review_store.get_item(c, item['task']['task_id'])
                occurrence = current['annotation']['occurrences'][0]
                operation = {'operation_id': 'accept', 'action': 'accept_fields',
                             'target_name_span': occurrence['name_span'],
                             'fields': ['software'], 'reason_code': 'accept_proposal'}
                review_store.apply_decision(c, batch(current, [operation], ident=f'review-{index}'))
        review_store.export_reference(c, tmp_path / 'snapshot')
    finally:
        c.close()
    return bundle, tmp_path / 'snapshot'


def readiness(monkeypatch):
    module = importlib.import_module('research.data.supplemental_readiness')
    monkeypatch.setattr(module, '_load_tokenizer', lambda: LocalTokenizer())
    return module


def test_no_snapshot_reports_missing_labels_without_ml_import(monkeypatch, tmp_path):
    module = readiness(monkeypatch)
    bundle = acquired(monkeypatch, tmp_path)
    monkeypatch.setattr(module, '_load_tokenizer', lambda: pytest.fail('no snapshot must not load tokenizer'))
    result = module.assess_readiness(bundle, None, tmp_path / 'report')
    assert result['status'] == 'reported' and result['annotation_ready'] is True
    assert result['detector_bundle_ready'] is False and result['fit_ready'] is False
    assert result['full_benchmark_ready'] is False
    assert {'snapshot_missing', 'sampler_unsupported'} <= set(result['blockers'])
    assert result['support']['by_label']['VERSION']['human_reviewed'] == 0
    script = "import sys; import research.data.supplemental_readiness; assert not {'torch','transformers'} & set(sys.modules)"
    subprocess.run([sys.executable, '-c', script], check=True)


@pytest.mark.parametrize('human', [False, True])
def test_readiness_preserves_field_review_and_unknown_versions(monkeypatch, tmp_path, human):
    module = readiness(monkeypatch)
    bundle, snap = snapshot(monkeypatch, tmp_path, human=human)
    result = module.assess_readiness(bundle, snap, tmp_path / 'report')
    assert result['detector_bundle_ready'] is True and result['fit_ready'] is False
    assert 'sampler_unsupported' in result['blockers']
    labels = result['support']['by_label']
    assert labels['SOFTWARE']['human_reviewed'] == (6 if human else 0)
    assert labels['SOFTWARE']['agent_provisional'] == (0 if human else 6)
    assert labels['VERSION'] == {'human_reviewed': 0, 'agent_provisional': 0}
    assert result['support']['fields']['versions']['unknown'] == 6
    assert result['support']['fields']['versions']['known_absent'] == 0
    assert len(result['support']['by_paper']) == 1
    assert result['support']['by_arm']['targeted']['passages'] == 6
    assert result['version_goal']['explicit_spans'] == 0


def test_mixed_sources_are_annotation_ready_but_sampler_unsupported(monkeypatch, tmp_path):
    module = readiness(monkeypatch)
    bundle, snap = snapshot(monkeypatch, tmp_path, sources=('ecosystems', 'europepmc'), versions=True)
    result = module.assess_readiness(bundle, snap, tmp_path / 'report')
    assert result['annotation_ready'] and result['detector_bundle_ready']
    assert not result['fit_ready'] and 'sampler_unsupported' in result['blockers']
    assert result['sources'] == ['ecosystems', 'europepmc']
    assert result['version_goal']['explicit_spans'] == 12
    assert result['version_goal']['papers'] == 2


def test_missing_local_tokenizer_is_explicit(monkeypatch, tmp_path):
    module = readiness(monkeypatch)
    bundle, snap = snapshot(monkeypatch, tmp_path, sources=('ecosystems',))
    def missing():
        raise OSError('cache absent')
    monkeypatch.setattr(module, '_load_tokenizer', missing)
    result = module.assess_readiness(bundle, snap, tmp_path / 'report')
    assert result['detector_bundle_ready'] is True
    assert result['fit_ready'] is False and 'tokenizer_cache_missing' in result['blockers']
    assert result['tokenizer_alignment']['status'] == 'unavailable'


def test_local_alignment_masks_boundary_errors_and_keeps_goal_nonblocking(monkeypatch, tmp_path):
    module = readiness(monkeypatch)
    bundle, snap = snapshot(monkeypatch, tmp_path, sources=('ecosystems',), versions=True)
    class BadBoundaryTokenizer(LocalTokenizer):
        def __call__(self, text, **kwargs):
            result = super().__call__(text, **kwargs)
            start = text.index('ImageJ')
            result['offset_mapping'] = [(s - 1 if s == start else s, e) for s, e in result['offset_mapping']]
            return result
    monkeypatch.setattr(module, '_load_tokenizer', lambda: BadBoundaryTokenizer())
    result = module.assess_readiness(bundle, snap, tmp_path / 'report')
    assert result['fit_ready'] is True
    assert result['blockers'] == []
    assert result['goal_shortfalls'] == ['version_goal_shortfall']
    assert result['tokenizer_alignment']['exclusions'][0]['reason'] == 'token_boundary'
    assert result['tokenizer_alignment']['exclusions'][0]['kind'] == 'SOFTWARE'
    assert result['version_goal']['explicit_spans'] == 6


def test_tokenizer_loader_pins_local_only_without_constructing_model(monkeypatch):
    module = importlib.import_module('research.data.supplemental_readiness')
    from types import SimpleNamespace
    expected = LocalTokenizer()
    def cached(model, *, revision, local_files_only):
        assert model == 'allenai/scibert_scivocab_cased'
        assert revision == 'ddf0be025f8e432a1870e34811997ba6725bf04a'
        assert local_files_only is True
        return '/verified-local-tokenizer'
    def tokenizer(path, *, do_lower_case, use_fast, local_files_only, trust_remote_code):
        assert path == '/verified-local-tokenizer'
        assert do_lower_case is False and use_fast is True
        assert local_files_only is True and trust_remote_code is False
        return expected
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(snapshot_download=cached))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer)))
    assert module._load_tokenizer() is expected


def test_readiness_rechecks_snapshot_before_publishing(monkeypatch, tmp_path):
    module = readiness(monkeypatch)
    bundle, snap = snapshot(monkeypatch, tmp_path)
    def changed():
        (snap / 'items.jsonl').write_text('{}\n')
        return LocalTokenizer()
    monkeypatch.setattr(module, '_load_tokenizer', changed)
    with pytest.raises(ValueError):
        module.assess_readiness(bundle, snap, tmp_path / 'report')
    assert not (tmp_path / 'report').exists()


def test_readiness_rejects_snapshot_selected_from_another_frozen_region(monkeypatch, tmp_path):
    module = readiness(monkeypatch)
    bundle, snap = snapshot(monkeypatch, tmp_path)
    # The snapshot reader checks internal provenance; D3 must additionally bind it to D2.
    from research.annotations.snapshots import read_review_snapshot
    manifest, data = read_review_snapshot(snap)
    data['items.jsonl'][0]['task']['whole_passage_audit'] = not data['items.jsonl'][0]['task']['whole_passage_audit']
    monkeypatch.setattr(module, 'read_review_snapshot', lambda _: (manifest, data))
    with pytest.raises(ValueError, match='SUPPLEMENTAL_SNAPSHOT_TASK_MISMATCH'):
        module.assess_readiness(bundle, snap, tmp_path / 'report')


def test_zero_parent_acquisition_reports_failure_without_fake_rows(monkeypatch, tmp_path):
    module = readiness(monkeypatch)
    from research.data import supplemental
    from test_supplemental_acquisition import FakeFetch, config
    monkeypatch.setattr(supplemental, 'fetch_public', FakeFetch())
    bundle = tmp_path / 'empty'
    supplemental.collect_supplemental(config(tmp_path), bundle)
    result = module.assess_readiness(bundle, None, tmp_path / 'report')
    assert result['annotation_ready'] is False
    assert result['detector_bundle_ready'] is False and result['fit_ready'] is False
    assert result['support']['by_paper'] == {}
    assert result['support']['passages'] == 0
    assert 'no_eligible_parents' in result['blockers']


@pytest.mark.parametrize('corruption', ['missing_prompt', 'stale_policy', 'invented_human'])
def test_readiness_rejects_invalid_snapshot(monkeypatch, tmp_path, corruption):
    module = readiness(monkeypatch)
    bundle, snap = snapshot(monkeypatch, tmp_path)
    path = snap / 'manifest.json'
    meta = json.loads(path.read_bytes())
    if corruption == 'missing_prompt':
        meta['prompt_sources'].pop('check')
    elif corruption == 'stale_policy':
        meta['policy']['policy_hash'] = '0' * 64
    else:
        rows = read_jsonl(snap / 'store-items.jsonl')
        annotation = json.loads(rows[0]['annotation'])
        annotation['review_workflow'] = {'schema_version': '1.0', 'field_reviews': {},
             'source_issues': [], 'approval': {'decision_id': 'invented'}}
        rows[0]['annotation'] = json.dumps(annotation)
        target = snap / 'store-items.jsonl'
        target.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        next(r for r in meta['files'] if r['path'] == target.name)['sha256'] = digest(target.read_bytes())
    path.write_bytes(json_bytes(meta))
    with pytest.raises(ValueError):
        module.assess_readiness(bundle, snap, tmp_path / 'report')
    assert not (tmp_path / 'report/manifest.json').exists()


@pytest.mark.parametrize('change', ['license', 'diagnostic_identity'])
def test_readiness_blocks_untrainable_parent_provenance(monkeypatch, tmp_path, change):
    module = readiness(monkeypatch)
    bundle = acquired(monkeypatch, tmp_path)
    from research.data.supplemental import load_supplemental
    from research.data.exposure import build_exposures, load_exposures
    loaded = load_supplemental(bundle)
    if change == 'license':
        loaded['documents'][0]['text_license'] = 'CC-BY-3.0'
    else:
        source = tmp_path / 'identity-only.jsonl'
        source.write_text(json.dumps({'source_ids': loaded['documents'][0]['source_ids']}) + '\n')
        exclusion = tmp_path / 'new-exposures'
        build_exposures({'inputs': [{'path': str(source), 'sha256': digest(source.read_bytes()),
                        'kind': 'documents_jsonl', 'role': 'diagnostic'}]}, exclusion)
        loaded['exposures'] = load_exposures(exclusion)
    monkeypatch.setattr(module, 'load_supplemental', lambda _: loaded)
    result = module.assess_readiness(bundle, None, tmp_path / 'report')
    assert result['annotation_ready'] is False and result['fit_ready'] is False
    assert ('text_license_unsupported' if change == 'license' else 'forbidden_exposure_overlap') in result['blockers']
    if change == 'diagnostic_identity':
        assert result['exposure_exclusions'][0]['reason'] == 'identifier_overlap'
