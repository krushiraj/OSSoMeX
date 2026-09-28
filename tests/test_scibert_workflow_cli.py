"""Public workflow dispatch and checked-in configuration contracts."""

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import pytest

from research.cli import main


ROOT = Path(__file__).resolve().parents[1]


def read_config(name):
    return json.loads((ROOT / 'configs/scibert' / name).read_bytes())


def test_openalex_config_is_bounded_and_anonymous():
    config = read_config('openalex-snippets-001.json')
    assert config == {
        'endpoint': 'https://api.openalex.org/funder-search',
        'queries': ['java', 'Python', 'R', 'ImageJ', 'scikit-learn', 'NumPy',
                    'GROMACS', 'MATLAB', 'SPSS', 'BLAST'],
        'page': 1, 'per_page': 5, 'snippets_per_work': 2,
        'max_candidates': 100, 'timeout_seconds': 35, 'max_retries': 2,
        'max_concurrency': 1, 'max_bytes': 16777216,
    }


def test_exposure_seed_config_freezes_all_required_inputs():
    config = read_config('exposures-001.json')
    expected = {
        'data/scibert-v2/ecosystems-pilot-002/bundle/manifest.json': ('bundle_manifest', 'train_reserved'),
        'data/scibert-v2/ecosystems-expansion-001/bundle/manifest.json': ('bundle_manifest', 'train_reserved'),
        'data/scibert-v2/corpus-001/documents.jsonl': ('documents_jsonl', 'native_excluded'),
        'inputs/sofair/sofair_docs.jsonl': ('documents_jsonl', 'historical_exposed'),
        'inputs/somesci/somesci_docs.jsonl': ('documents_jsonl', 'historical_exposed'),
        'inputs/openalex_snippets_full.jsonl': ('diagnostic_inputs', 'diagnostic'),
        'inputs/dev/openalex_snippets_dev.jsonl': ('diagnostic_inputs', 'diagnostic'),
        'runs/scibert-v2/openalex-parallel-probe-20260927-001/inputs.jsonl': ('diagnostic_inputs', 'diagnostic'),
    }
    assert len(config['inputs']) == len(expected)
    assert {row['path']: (row['kind'], row['role']) for row in config['inputs']} == expected
    for row in config['inputs']:
        assert len(row['sha256']) == 64
        assert set(row['sha256']) <= set('0123456789abcdef')
        # Local data are intentionally ignored. Check bytes when available;
        # collection/load tests independently exercise missing-source failures.
        source = ROOT / row['path']
        if source.is_file():
            assert hashlib.sha256(source.read_bytes()).hexdigest() == row['sha256']


def test_sampling_reserves_both_existing_training_bundles():
    config = read_config('sampling.json')
    assert config['training_reservations'] == [
        'data/scibert-v2/ecosystems-pilot-002/bundle',
        'data/scibert-v2/ecosystems-expansion-001/bundle',
    ]
    assert config['quotas'] == {
        'sofair': {'train': 40, 'dev': 15, 'test': 25},
        'somesci': {'train': 20, 'dev': 7, 'test': 13},
        'ecosystems': {'train': 10, 'dev': 4, 'test': 6},
        'openalex': {'train': 10, 'dev': 4, 'test': 6},
    }


def test_supplemental_config_has_only_approved_acquisition_settings():
    assert read_config('supplemental-001.json') == {
        'seed': 42, 'metadata_fallback': True, 'ecosystems_projects': {},
        'annotation_policy': 'annotations/scibert-v2/policy-2.1-d17.md',
    }


def test_comparison_config_keeps_requested_arms_and_controls_distinct():
    config = read_config('comparison-001.json')
    assert config['schema_version'] == 'comparison-config-1'
    assert config['tokenizer_checkpoint'] == 'checkpoints/scibert-detector-002'
    assert (config['max_content_tokens'], config['overlap_tokens']) == (480, 64)
    assert config['warmup_text'] == 'We used NumPy.'
    arms = {arm['arm_id']: arm for arm in config['arms']}
    assert len(arms) == len(config['arms']) == 13
    assert set(arms) == {
        'scibert-base', 'scibert-detector-002', 'softcite-0.8.1',
        'gemma2-2b-L0', 'gemma2-2b-L1', 'gemma3-4b-L1',
        'qwen25-3b-L1', 'llama32-3b-L1', 'gemma2-historical-grammar',
        'fastdict', 'softcite-dict-union', 'modernbert',
        'scibert-frozen-encoder-probe',
    }
    assert arms['scibert-base']['backend'] == 'base'
    assert arms['scibert-base']['config'] == {
        'model_id': 'allenai/scibert_scivocab_cased',
        'revision': 'ddf0be025f8e432a1870e34811997ba6725bf04a',
    }
    assert arms['scibert-frozen-encoder-probe']['disabled_reason'] == 'approval_required'
    assert arms['gemma2-historical-grammar']['unavailable_reason'] == 'historical_request_provenance_incomplete'
    assert arms['gemma2-2b-L0']['variant'] == 'L0'
    assert arms['gemma2-2b-L1']['variant'] == 'L1'
    assert arms['gemma2-2b-L0']['config_path'] == arms['gemma2-2b-L1']['config_path'] == 'configs/llm.json'
    for arm in arms.values():
        assert ('config' in arm) != ('config_path' in arm)
        assert arm['request_options'] == {}


def test_comparison_config_resolves_exact_native_bytes_without_model_loads():
    from research.comparison.backends import resolve_arm
    for arm in read_config('comparison-001.json')['arms']:
        resolved = resolve_arm(arm, repo_root=ROOT)
        source = resolved['config_source']
        assert hashlib.sha256(source['bytes_utf8'].encode()).hexdigest() == source['sha256']
        assert json.loads(source['bytes_utf8']) == resolved['config']
        assert resolved['arm_id'] == arm['arm_id']
        assert resolved.get('variant') == arm.get('variant')
        if 'config_path' in arm:
            assert source['bytes_utf8'].encode() == (ROOT / arm['config_path']).read_bytes()


@pytest.mark.parametrize('command,module,function,status', [
    ('openalex-snippets', 'research.data.openalex_snippets', 'collect_snippets', 'completed'),
    ('exposures', 'research.data.exposure', 'build_exposures', 'complete'),
    ('supplemental-tasks', 'research.annotations.supplemental', 'prepare_supplemental_tasks', 'ready_for_annotation'),
    ('supplemental-readiness', 'research.data.supplemental_readiness', 'assess_readiness', 'reported'),
])
def test_all_workflow_commands_dispatch_without_network(command, module, function, status,
                                                       tmp_path, monkeypatch, capsys):
    calls = []
    fake = ModuleType(module)
    setattr(fake, function, lambda *args: calls.append(args) or {'status': status, 'fit_ready': False})
    monkeypatch.setitem(sys.modules, module, fake)
    config = tmp_path / 'config.json'
    config.write_bytes(b'{"inputs": []}\n')
    output, bundle = tmp_path / 'output', tmp_path / 'bundle'
    flags = ['--bundle', str(bundle)] if command.startswith('supplemental-') else ['--config', str(config)]
    assert main(['data', command, *flags, '--output', str(output)]) == 0
    assert json.loads(capsys.readouterr().out)['status'] == status
    assert len(calls) == 1
    if command == 'supplemental-tasks':
        assert calls[0] == (bundle, output)
    elif command == 'supplemental-readiness':
        assert calls[0] == (bundle, None, output)
    else:
        effective, destination = calls[0]
        assert destination == output
        assert effective['inputs'] == []
        assert effective['config_source'] == {
            'path': str(config.resolve()),
            'sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
            'bytes_utf8': config.read_text(),
        }
    assert config.read_bytes() == b'{"inputs": []}\n'


@pytest.mark.parametrize('status,expected', [('completed', 0), ('partial', 2), ('failed', 2)])
def test_collection_status_exit_codes(status, expected, tmp_path, monkeypatch, capsys):
    fake = ModuleType('research.data.openalex_snippets')
    fake.collect_snippets = lambda *args: {'status': status}
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    config = tmp_path / 'config.json'
    config.write_text('{}')
    assert main(['data', 'openalex-snippets', '--config', str(config), '--output', str(tmp_path / 'out')]) == expected
    assert json.loads(capsys.readouterr().out)['status'] == status


@pytest.mark.parametrize('status,expected', [('ready_for_annotation', 0), ('partial', 2), ('failed', 2)])
def test_supplemental_collection_status_exit_codes(status, expected, tmp_path, monkeypatch, capsys):
    fake = ModuleType('research.data.supplemental')
    fake.collect_supplemental = lambda *args: {'status': status, 'training_ready': False}
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    config, exposures = tmp_path / 'config.json', tmp_path / 'exposures'
    config.write_text('{}')
    exposures.mkdir()
    (exposures / 'manifest.json').write_text('{}')
    assert main(['data', 'supplemental', '--config', str(config), '--exposures', str(exposures),
                 '--output', str(tmp_path / 'out')]) == expected
    assert json.loads(capsys.readouterr().out)['status'] == status


def test_readiness_dispatches_explicit_snapshot_and_reports_not_fit_ready(tmp_path, monkeypatch, capsys):
    calls = []
    fake = ModuleType('research.data.supplemental_readiness')
    fake.assess_readiness = lambda *args: calls.append(args) or {'status': 'reported', 'fit_ready': False}
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    bundle, snapshot, output = tmp_path / 'bundle', tmp_path / 'snapshot', tmp_path / 'out'
    assert main(['data', 'supplemental-readiness', '--bundle', str(bundle),
                 '--snapshot', str(snapshot), '--output', str(output)]) == 0
    assert calls == [(bundle, snapshot, output)]
    assert json.loads(capsys.readouterr().out)['fit_ready'] is False


@pytest.mark.parametrize('error', [ValueError('invalid schema'), FileExistsError('existing destination')])
def test_data_workflow_validation_errors_are_nonzero(error, tmp_path, monkeypatch, capsys):
    fake = ModuleType('research.data.openalex_snippets')
    def fail(*args):
        raise error
    fake.collect_snippets = fail
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    config = tmp_path / 'config.json'
    config.write_text('{}')
    assert main(['data', 'openalex-snippets', '--config', str(config), '--output', str(tmp_path / 'out')]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'failed'
    assert report['error_type'] == type(error).__name__


@pytest.mark.parametrize('args', [
    ['--help'], ['data', '--help'], ['data', 'openalex-snippets', '--help'],
    ['data', 'exposures', '--help'], ['data', 'supplemental-tasks', '--help'],
    ['data', 'supplemental-readiness', '--help'], ['data', 'split', '--help'],
    ['data', 'supplemental', '--help'],
    ['detector', 'predict', '--help'], ['benchmark', '--help'], ['annotate', '--help'],
    ['comparison', '--help'], ['comparison', 'preflight', '--help'],
    ['comparison', 'run', '--help'], ['comparison', 'report', '--help'],
])
def test_workflow_help_never_uses_network(args, monkeypatch, capsys):
    def unexpected_request(*args, **kwargs):
        pytest.fail('help must not access the network')
    monkeypatch.setattr('requests.sessions.Session.request', unexpected_request)
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 0
    assert 'usage:' in capsys.readouterr().out


def test_supplemental_dispatch_freezes_explicit_exposures(tmp_path, monkeypatch, capsys):
    calls = []
    fake = ModuleType('research.data.supplemental')
    fake.collect_supplemental = lambda *args: calls.append(args) or {'status': 'ready_for_annotation'}
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    config, exposures, output = tmp_path / 'config.json', tmp_path / 'exposures', tmp_path / 'out'
    original = b'{"seed":42}\n'
    config.write_bytes(original)
    exposures.mkdir()
    manifest_bytes = b'{"schema_version":"exposures-1","files":[]}\n'
    (exposures / 'manifest.json').write_bytes(manifest_bytes)
    assert main(['data', 'supplemental', '--config', str(config), '--exposures', str(exposures),
                 '--output', str(output)]) == 0
    assert len(calls) == 1
    effective, destination = calls[0]
    assert destination == output
    assert effective['exposures'] == str(exposures.resolve())
    assert effective['exposures_manifest_sha256'] == hashlib.sha256(manifest_bytes).hexdigest()
    assert effective['config_source']['bytes_utf8'] == original.decode()
    assert config.read_bytes() == original
    assert json.loads(capsys.readouterr().out)['status'] == 'ready_for_annotation'


def test_exposure_dispatch_appends_diagnostic_without_mutating_config(tmp_path, monkeypatch, capsys):
    calls = []
    fake = ModuleType('research.data.exposure')
    fake.build_exposures = lambda *args: calls.append(args) or {'status': 'ready'}
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    config, diagnostic = tmp_path / 'config.json', tmp_path / 'diagnostic'
    config.write_bytes(b'{"inputs":[]}\n')
    diagnostic.mkdir()
    payload = b'{"files":[]}\n'
    (diagnostic / 'manifest.json').write_bytes(payload)
    assert main(['data', 'exposures', '--config', str(config), '--diagnostic', str(diagnostic),
                 '--output', str(tmp_path / 'out')]) == 0
    assert calls[0][0]['inputs'] == [{
        'kind': 'bundle_manifest', 'path': str((diagnostic / 'manifest.json').resolve()),
        'sha256': hashlib.sha256(payload).hexdigest(), 'role': 'diagnostic'}]
    assert config.read_bytes() == b'{"inputs":[]}\n'
    assert json.loads(capsys.readouterr().out)['status'] == 'ready'


@pytest.fixture
def split_case(tmp_path, monkeypatch):
    from research.contracts import text_revision
    from research.data import cli as data_cli
    from research.data.exposure import build_exposures
    roles = ['train_reserved', 'diagnostic', 'native_excluded', 'heldout_reserved', 'historical_exposed']
    documents = []
    inputs = []
    for name in [*roles, 'clean-a', 'clean-b']:
        text = f'{name} passage.'
        doc = {'document_id': name, 'source_record_id': name, 'text': text,
               'text_revision': text_revision(text), 'source': 'ecosystems',
               'source_ids': {'doi': f'10.1000/{name}'}, 'fulltext_eligible': True}
        documents.append(doc)
        if name in roles:
            source = tmp_path / f'{name}.jsonl'
            source.write_text(json.dumps(doc) + '\n')
            inputs.append({'kind': 'documents_jsonl', 'path': str(source),
                           'sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'role': name})
    exposures = tmp_path / 'exposures'
    build_exposures({'inputs': inputs}, exposures)
    original = deepcopy(documents)
    monkeypatch.setattr(data_cli, 'load_corpus', lambda path: deepcopy(documents))
    config = tmp_path / 'sampling.json'
    config.write_text(json.dumps({
        'seed': 42, 'quotas': {'ecosystems': {'train': 1, 'dev': 1, 'test': 1}},
        'historical_documents': [], 'training_reservations': [],
        'exposure_bundles': [{'path': str(exposures),
                              'sha256': hashlib.sha256((exposures / 'manifest.json').read_bytes()).hexdigest()}],
    }))
    output, private_output = tmp_path / 'split', tmp_path / 'private-test-fixture'
    argv = ['data', 'split', '--corpus', str(tmp_path / 'corpus'), '--config', str(config),
            '--output', str(output), '--private-output', str(private_output)]
    return config, exposures, output, private_output, argv, documents, original, inputs


def test_split_cli_applies_exposures_before_actual_assignment(split_case, capsys):
    from research.data.manifest import read_jsonl
    config, exposures, output, private_output, argv, documents, original, _ = split_case
    assert main(argv) == 0
    selected = {row['document_id']: role for role, path in [
        ('train', output / 'train'), ('dev', output / 'dev'), ('test', private_output)]
        for row in read_jsonl(path / 'documents.jsonl')}
    assert selected['train_reserved'] == 'train'
    assert set(selected) == {'train_reserved', 'clean-a', 'clean-b'}
    assert documents == original
    assert json.loads(capsys.readouterr().out)['status'] == 'ready'
    summary = json.loads((output / 'manifest.json').read_bytes())
    assert summary['exposure_bundles'] == json.loads(config.read_bytes())['exposure_bundles']


@pytest.mark.parametrize('problem', ['missing', 'stale', 'unpinned', 'invalid_shape'])
def test_split_rejects_missing_or_unpinned_exposure_bundles(split_case, problem):
    config, exposures, output, private_output, argv, _, _, _ = split_case
    settings = json.loads(config.read_bytes())
    if problem == 'missing':
        settings['exposure_bundles'][0]['path'] = str(exposures / 'absent')
    elif problem == 'stale':
        settings['exposure_bundles'][0]['sha256'] = '0' * 64
    elif problem == 'unpinned':
        del settings['exposure_bundles'][0]['sha256']
    else:
        settings['exposure_bundles'] = str(exposures)
    config.write_text(json.dumps(settings))
    with pytest.raises((ValueError, OSError)):
        main(argv)
    assert not output.exists()
    assert not private_output.exists()


@pytest.mark.parametrize('blocked', [False, True])
@pytest.mark.parametrize('target', ['manifest', 'companion', 'source'])
def test_split_rechecks_cached_exposures_before_any_publication(split_case, monkeypatch, blocked, target):
    from research.data import cli as data_cli
    from research.data import exposure
    from research.data.splits import SplitError
    _, exposures, output, private_output, argv, _, _, inputs = split_case
    original_assign = data_cli.assign_splits
    original_load = exposure.load_exposures
    calls = []
    def load(bundle):
        calls.append(bundle)
        return original_load(bundle)
    def mutate_then_assign(*args):
        path = {'manifest': exposures / 'manifest.json',
                'companion': exposures / 'identities.jsonl',
                'source': Path(inputs[0]['path'])}[target]
        path.write_bytes(path.read_bytes() + b'\n')
        if blocked:
            raise SplitError({'code': 'synthetic_shortage'})
        return original_assign(*args)
    monkeypatch.setattr(exposure, 'load_exposures', load)
    monkeypatch.setattr(data_cli, 'assign_splits', mutate_then_assign)
    with pytest.raises(ValueError):
        main(argv)
    assert calls == [exposures]
    assert not output.exists()
    assert not private_output.exists()


def test_split_blocked_report_retains_exposure_provenance(split_case, capsys):
    config, _, output, private_output, argv, _, _, _ = split_case
    settings = json.loads(config.read_bytes())
    settings['quotas']['ecosystems']['test'] = 2
    config.write_text(json.dumps(settings))
    assert main(argv) == 2
    report = json.loads((output / 'blocked.json').read_bytes())
    assert report['status'] == 'blocked'
    assert report['exposure_bundles'] == settings['exposure_bundles']
    assert not private_output.exists()
    assert json.loads(capsys.readouterr().out) == report


@pytest.mark.parametrize('existing', ['public', 'private', 'both'])
@pytest.mark.parametrize('kind', ['directory', 'dangling_symlink'])
def test_shortage_split_rejects_existing_destinations_before_assignment(split_case, monkeypatch, existing, kind):
    from research.data import cli as data_cli
    config, _, output, private_output, argv, _, _, _ = split_case
    settings = json.loads(config.read_bytes())
    settings['quotas']['ecosystems']['test'] = 2
    config.write_text(json.dumps(settings))
    destinations = {'public': [output], 'private': [private_output],
                    'both': [output, private_output]}[existing]
    for destination in destinations:
        if kind == 'directory':
            destination.mkdir()
            (destination / 'frozen.json').write_bytes(b'{"frozen":true}\n')
        else:
            destination.symlink_to(destination.with_name(destination.name + '-absent'), target_is_directory=True)
    def inventory():
        return {path: ('symlink', str(path.readlink())) if path.is_symlink()
                else path.read_bytes() if path.is_file() else 'directory'
                for path in config.parent.rglob('*')}
    frozen = inventory()
    assignments = []
    original_assign = data_cli.assign_splits
    def assign(*args):
        assignments.append(args)
        return original_assign(*args)
    monkeypatch.setattr(data_cli, 'assign_splits', assign)
    try:
        with pytest.raises(FileExistsError, match='split outputs must be new'):
            main(argv)
    finally:
        assert inventory() == frozen
        assert assignments == []


def test_split_pin_matches_manifest_bytes_actually_loaded(split_case, monkeypatch):
    from research.data import exposure
    from research.data import cli as data_cli
    _, exposures, output, private_output, argv, _, _, _ = split_case
    manifest = exposures / 'manifest.json'
    pinned = manifest.read_bytes()
    original_load = exposure.load_exposures
    def changed_during_load(bundle):
        manifest.write_bytes(pinned + b'\n')
        loaded = original_load(bundle)
        manifest.write_bytes(pinned)
        return loaded
    monkeypatch.setattr(exposure, 'load_exposures', changed_during_load)
    def no_grouping(*args):
        pytest.fail('mismatched loaded pin must reject before grouping')
    monkeypatch.setattr(data_cli, 'group_works', no_grouping)
    with pytest.raises(ValueError, match='manifest hash mismatch'):
        main(argv)
    assert not output.exists()
    assert not private_output.exists()


def test_legacy_split_without_exposures_preserves_outputs_on_collision(split_case, capsys):
    config, _, output, private_output, argv, _, _, _ = split_case
    settings = json.loads(config.read_bytes())
    del settings['exposure_bundles']
    config.write_text(json.dumps(settings))
    assert main(argv) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary['status'] == 'ready'
    assert 'exposure_bundles' not in summary
    frozen = {path: path.read_bytes() for root in (output, private_output)
              for path in root.rglob('*') if path.is_file()}
    with pytest.raises(FileExistsError):
        main(argv)
    assert {path: path.read_bytes() for path in frozen} == frozen


def test_legacy_benchmark_still_writes_timing_summary(tmp_path, monkeypatch, capsys):
    from research import cli
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    run = tmp_path / 'runs/fixture'
    run.mkdir(parents=True)
    (run / 'events.jsonl').write_text(
        '{"duration_seconds":1.0,"status":"success"}\n'
        '{"duration_seconds":3.0,"status":"failure"}\n')
    assert main(['benchmark', '--run-id', 'fixture']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['n_requests'] == 2
    assert report['total_seconds'] == 4.0
    assert report['mean_seconds'] == 2.0
    assert report['error_events'] == 1
    assert json.loads((run / 'benchmark.jsonl').read_text()) == report


def test_legacy_annotation_restore_command_keeps_its_dispatch(tmp_path, monkeypatch, capsys):
    from research.annotations import snapshots
    calls = []
    def restore(bundle, store):
        calls.append((bundle, store))
        return {'status': 'restored', 'fixture': True}
    monkeypatch.setattr(snapshots, 'restore_review_snapshot', restore)
    bundle, store = tmp_path / 'bundle', tmp_path / 'store.db'
    assert main(['annotate', 'restore', '--bundle', str(bundle), '--store', str(store)]) == 0
    assert calls == [(bundle, store)]
    assert json.loads(capsys.readouterr().out) == {'status': 'restored', 'fixture': True}


@pytest.mark.parametrize('inactive,expected', [('unsupported', 0), ('unavailable', 1)])
def test_comparison_commands_dispatch_without_network(tmp_path, monkeypatch, capsys, inactive, expected):
    from research.comparison import cli as comparison_cli
    from research.contracts import text_revision
    def no_network(*args, **kwargs):
        pytest.fail('workflow test must not access network')
    monkeypatch.setattr('requests.sessions.Session.request', no_network)
    config = tmp_path / 'comparison.json'
    config.write_text(json.dumps({'max_content_tokens': 4, 'overlap_tokens': 0,
        'arms': [{'arm_id': 'scibert-base', 'backend': 'base', 'config': {}},
                 {'arm_id': 'optional', 'config': {},
                  ('disabled_reason' if inactive == 'unsupported' else 'unavailable_reason'): 'fixture'}]}))
    source = tmp_path / 'input.jsonl'
    source.write_text(json.dumps({'document_id': 'fixture', 'text': 'Hi',
                                 'text_revision': text_revision('Hi')}) + '\n')
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {'input_ids': list(range(len(text))),
                    'offset_mapping': [(i, i + 1) for i in range(len(text))]}
    monkeypatch.setattr(comparison_cli, 'load_tokenizer', lambda config: (Tokenizer(), {'fixture': True}))
    for command in ['preflight', 'run']:
        args = ['comparison', command, '--config', str(config), '--output', str(tmp_path / command)]
        if command == 'run':
            args += ['--input', str(source)]
        assert main(args) == expected
        manifest = json.loads((tmp_path / command / 'manifest.json').read_bytes())
        assert manifest['status'] == ('complete' if expected == 0 else 'partial')
    assert main(['comparison', 'report', '--run', str(tmp_path / 'run'),
                 '--output', str(tmp_path / 'report')]) == 0
    assert (tmp_path / 'report' / 'report.json').is_file()
    assert json.loads(capsys.readouterr().out.splitlines()[-1])['status'] == 'reported'


@pytest.mark.parametrize('args', [
    ['--help'], ['data', 'openalex-snippets', '--help'], ['data', 'exposures', '--help'],
    ['data', 'supplemental', '--help'], ['data', 'supplemental-tasks', '--help'],
    ['data', 'supplemental-readiness', '--help'], ['comparison', '--help'],
    ['comparison', 'preflight', '--help'], ['comparison', 'run', '--help'],
    ['comparison', 'report', '--help'],
])
def test_help_in_fresh_process_imports_no_ml_and_uses_no_network(args):
    code = '''
import importlib.abc
import socket
import sys
class NoML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'transformers'):
            raise AssertionError('ML import during help: ' + fullname)
sys.meta_path.insert(0, NoML())
def no_network(*args, **kwargs):
    raise AssertionError('Network during help')
socket.create_connection = no_network
socket.socket.connect = no_network
socket.socket.connect_ex = no_network
from research.cli import main
main(sys.argv[1:])
'''
    result = subprocess.run([sys.executable, '-c', code, *args], cwd=ROOT,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout


def test_comparison_failed_requested_arm_keeps_partial_run_and_report(tmp_path, monkeypatch):
    from research.comparison import cli as comparison_cli
    from research.comparison.backends import capabilities
    from research.comparison.runner import run_comparison
    from research.contracts import text_revision
    calls = []
    class FailingBackend:
        def load(self, arm):
            calls.append(('load', arm['arm_id']))
            return {'status': 'ready', 'reason': None, 'capabilities': capabilities(True),
                    'identity': {'synthetic': True}}
        def predict(self, window):
            calls.append(('predict', window['text']))
            return {'window_id': window['window_id'], 'status': 'failure', 'spans': [],
                    'unresolved': [], 'raw': {'body': 'fixture error'}, 'error': 'fixture error'}
        def close(self):
            calls.append(('close',))
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {'input_ids': list(range(len(text))),
                    'offset_mapping': [(i, i + 1) for i in range(len(text))]}
    def run(*args):
        return run_comparison(*args, adapter_factory=lambda arm: FailingBackend())
    def no_network(*args, **kwargs):
        pytest.fail('failed comparison fixture must not access network')
    monkeypatch.setattr('requests.sessions.Session.request', no_network)
    monkeypatch.setattr(comparison_cli, 'run_comparison', run)
    monkeypatch.setattr(comparison_cli, 'load_tokenizer', lambda config: (Tokenizer(), {'fixture': True}))
    config, source = tmp_path / 'config.json', tmp_path / 'inputs.jsonl'
    config.write_text(json.dumps({'arms': [{'arm_id': 'requested', 'config': {}}]}))
    source.write_text(json.dumps({'document_id': 'fixture', 'text': 'Hi',
                                 'text_revision': text_revision('Hi')}) + '\n')
    assert main(['comparison', 'run', '--config', str(config), '--input', str(source),
                 '--output', str(tmp_path / 'run')]) == 1
    assert calls == [('load', 'requested'), ('predict', 'We used NumPy.'), ('predict', 'Hi'), ('close',)]
    manifest = json.loads((tmp_path / 'run/manifest.json').read_bytes())
    assert manifest['status'] == 'partial'
    assert main(['comparison', 'report', '--run', str(tmp_path / 'run'),
                 '--output', str(tmp_path / 'report')]) == 0
    report = json.loads((tmp_path / 'report/report.json').read_bytes())
    assert report['run_status'] == 'partial'
