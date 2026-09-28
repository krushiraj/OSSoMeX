import importlib
import json
from copy import deepcopy

import pytest

from research.contracts import text_revision
from research.comparison.backends import capabilities


def fixture_inputs():
    documents = [{'document_id': name, 'text': text, 'text_revision': text_revision(text)}
                 for name, text in [('../../doc', 'NumPy works.'), ('d2', 'We use R.')]]
    windows = [{'window_id': f'w{i}', 'document_id': d['document_id'],
                'text_revision': d['text_revision'], 'start': 0, 'end': len(d['text']),
                'text': d['text'], 'window_text_revision': d['text_revision'], 'content_tokens': 4}
               for i, d in enumerate(documents)]
    return documents, windows


class FakeBackend:
    def __init__(self, arm, events, mode='ready'):
        self.arm, self.events, self.mode = arm, events, mode

    def load(self, arm):
        assert arm['config_source']['bytes_utf8'] == '{"native":true}'
        self.events.append((self.arm['arm_id'], 'load'))
        return {'status': 'unavailable' if self.mode == 'unavailable' else 'ready',
                'reason': 'offline' if self.mode == 'unavailable' else None,
                'capabilities': capabilities(self.mode != 'unavailable'),
                'identity': {'verified': False, 'policy_differences': ['synthetic policy'],
                             'prompt_sources': {'system': 'Exact prompt\r\n'}}}

    def predict(self, window):
        self.events.append((self.arm['arm_id'], 'predict', deepcopy(window)))
        failed = self.mode == 'failing' and window['window_id'] == 'w0'
        return {'window_id': window['window_id'], 'status': 'failure' if failed else 'no_mentions',
                'spans': [], 'unresolved': [], 'raw': {'body': 'bad native body' if failed else 'native []'},
                'error': 'failed window' if failed else None}

    def close(self):
        self.events.append((self.arm['arm_id'], 'close'))


def execute(tmp_path, mode='ready', **kwargs):
    runner = importlib.import_module('research.comparison.runner')
    documents, windows = fixture_inputs()
    events = []
    arms = [{'arm_id': name, 'backend': 'fake', 'config': {'native': True}} for name in ('a', '../../b')]
    ticks = iter(range(1000))
    manifest = runner.run_comparison(documents, windows, arms, tmp_path / 'run',
                                    adapter_factory=lambda arm: FakeBackend(arm, events, mode if arm['arm_id'] != 'a' else 'ready'),
                                    clock=lambda: next(ticks), **kwargs)
    return manifest, events


@pytest.mark.parametrize('mode', ['ready', 'unavailable', 'failing'])
def test_identical_inputs_and_complete_result_population(tmp_path, mode):
    manifest, events = execute(tmp_path, mode)
    results = [json.loads(line) for line in (tmp_path / 'run/results.jsonl').read_text().splitlines()]
    assert len(results) == 4
    assert {r['status'] for r in results[:2]} == {'no_mentions'}
    assert [r['status'] for r in results[2:]] == {'ready': ['no_mentions', 'no_mentions'],
                                                'unavailable': ['unavailable', 'unavailable'],
                                                'failing': ['failure', 'no_mentions']}[mode]
    assert events.count(('a', 'load')) == events.count(('../../b', 'load')) == 1
    measured = [[e[2] for e in events if e[:2] == (arm, 'predict') and e[2]['window_id'].startswith('w')]
                for arm in ('a', '../../b')]
    assert measured[0] == fixture_inputs()[1]
    assert measured[1] == ([] if mode == 'unavailable' else measured[0])
    assert events.index(('a', 'close')) < events.index(('../../b', 'load'))
    warmup = [e[2] for e in events if e[:2] == ('a', 'predict') and e[2]['window_id'] not in ('w0', 'w1')]
    assert len(warmup) == 1 and warmup[0]['text'] not in [d['text'] for d in fixture_inputs()[0]]
    assert set(manifest['arms'][0]['timing']) >= {'load_seconds', 'warmup_seconds', 'measured_seconds'}
    assert manifest['status'] == ('complete' if mode == 'ready' else 'partial')
    assert manifest['exit_code'] == (0 if mode == 'ready' else 1)
    for result in results:
        assert result['raw_artifact'].startswith('raw/') and '..' not in result['raw_artifact']
        assert (tmp_path / 'run' / result['raw_artifact']).is_file()
    raw_text = ''.join(p.read_text() for p in (tmp_path / 'run/raw').glob('*.json'))
    assert 'native []' in raw_text
    if mode == 'failing':
        assert 'bad native body' in raw_text
    assert any((tmp_path / 'run' / f['path']).read_bytes() == b'Exact prompt\r\n' for f in manifest['files'])


def test_rejects_reused_root_and_empty_or_incomplete_population(tmp_path):
    runner = importlib.import_module('research.comparison.runner')
    docs, windows = fixture_inputs()
    arms = [{'arm_id': 'a', 'backend': 'base', 'config': {}}]
    for bad_docs, bad_windows, bad_arms in [([], [], arms), (docs, windows[:1], arms),
                                           (docs, windows + windows[:1], arms), (docs, windows, [])]:
        with pytest.raises(ValueError):
            runner.run_comparison(bad_docs, bad_windows, bad_arms, tmp_path / 'invalid')
        assert not (tmp_path / 'invalid').exists()
    (tmp_path / 'exists').mkdir()
    with pytest.raises(FileExistsError):
        runner.run_comparison(docs, windows, arms, tmp_path / 'exists')


@pytest.mark.parametrize('failure_path', ['results.jsonl', 'manifest.json', 'raw/'])
def test_disk_failure_never_publishes_terminal_manifest(tmp_path, monkeypatch, failure_path):
    runner = importlib.import_module('research.comparison.runner')
    real = runner.atomic_write_new
    def fail(path, payload):
        if failure_path in str(path):
            raise OSError('disk full')
        assert not (tmp_path / 'run/manifest.json').exists()
        real(path, payload)
    monkeypatch.setattr(runner, 'atomic_write_new', fail)
    with pytest.raises(OSError, match='disk full'):
        execute(tmp_path)
    assert not (tmp_path / 'run/manifest.json').exists()


@pytest.mark.parametrize('artifact', ['results.jsonl', 'raw'])
def test_report_refuses_tampered_evidence(tmp_path, artifact):
    execute(tmp_path)
    report = importlib.import_module('research.comparison.report')
    target = tmp_path / 'run/results.jsonl' if artifact == 'results.jsonl' else next((tmp_path / 'run/raw').glob('*.json'))
    target.write_bytes(b'{}\n')
    with pytest.raises(ValueError, match='hash'):
        report.build_report(tmp_path / 'run', tmp_path / 'report')
    assert not (tmp_path / 'report').exists()


def test_report_verified_agreement_and_separate_reference_kinds(tmp_path):
    execute(tmp_path, 'failing')
    report = importlib.import_module('research.comparison.report')
    result = report.build_report(tmp_path / 'run', tmp_path / 'agreement')
    assert result['mode'] == 'unlabelled_agreement'
    assert not {'precision', 'recall', 'f1'} & set(json.dumps(result).replace('"', ' ').split())
    assert result['documents'][0]['text'] == 'NumPy works.'
    assert len(result['documents'][0]['arms']) == 2
    markdown = (tmp_path / 'agreement/report.md').read_text()
    assert 'NumPy works.' in markdown and 'raw/' in markdown and 'synthetic policy' in markdown
    reference = tmp_path / 'refs.jsonl'
    doc = fixture_inputs()[0][0]
    reference.write_text(json.dumps({'document_id': doc['document_id'], 'text_revision': doc['text_revision'],
                                    'spans': [], 'coverage': [], 'provenance': {}}) + '\n')
    scored = report.build_report(tmp_path / 'run', tmp_path / 'reference', references=reference)
    assert set(scored['references']) == {'human_reviewed', 'agent_provisional'}
    assert scored['references']['human_reviewed']['mode'] == 'reference_diagnostic'


def test_interrupt_does_not_publish_manifest(tmp_path, monkeypatch):
    importlib.import_module('research.comparison.runner')
    def interrupt(self, window):
        raise KeyboardInterrupt()
    monkeypatch.setattr(FakeBackend, 'predict', interrupt)
    with pytest.raises(KeyboardInterrupt):
        execute(tmp_path)
    assert not (tmp_path / 'run/manifest.json').exists()


@pytest.mark.parametrize('malformation', ['foreign_id', 'missing_id', 'unhashable_id', 'missing_raw', 'exception', 'mutate_input'])
def test_bad_window_outputs_retain_failure_and_exact_frozen_envelope(tmp_path, monkeypatch, malformation):
    original = FakeBackend.predict
    def malformed(self, window):
        if window['window_id'] != 'w0':
            return original(self, window)
        result = original(self, window)
        if malformation == 'foreign_id':
            result['window_id'] = 'foreign'
        elif malformation == 'missing_id':
            result.pop('window_id')
        elif malformation == 'unhashable_id':
            result['window_id'] = []
        elif malformation == 'missing_raw':
            result.pop('raw')
        elif malformation == 'exception':
            raise RuntimeError()
        else:
            window['text'] = 'changed'
        return result
    monkeypatch.setattr(FakeBackend, 'predict', malformed)
    manifest, events = execute(tmp_path)
    rows = [json.loads(line) for line in (tmp_path / 'run/results.jsonl').read_text().splitlines()]
    assert rows[0]['status'] == ('no_mentions' if malformation == 'mutate_input' else 'failure')
    assert rows[0]['chunks'][0]['window']['text'] == 'NumPy works.'
    assert manifest['result_count'] == 4


def test_invalid_warmup_is_partial_even_when_all_measured_windows_complete(tmp_path, monkeypatch):
    original = FakeBackend.predict
    def malformed(self, window):
        result = original(self, window)
        if window['window_id'] == 'synthetic-warmup':
            result['unresolved'] = [{'text': 'missing', 'reason': 'not_found'}]
        return result
    monkeypatch.setattr(FakeBackend, 'predict', malformed)
    manifest, _ = execute(tmp_path)
    assert manifest['status'] == 'partial' and manifest['exit_code'] == 1
    assert manifest['arms'][0]['warmup']['status'] == 'failure'


def test_configuration_sources_and_context_are_rechecked_before_publication(tmp_path, monkeypatch):
    runner = importlib.import_module('research.comparison.runner')
    docs, windows = fixture_inputs()
    config = tmp_path / 'native.json'
    config.write_text('{"native":true}')
    original = FakeBackend.close
    def mutate(self):
        original(self)
        config.write_text('{"native":false}')
    monkeypatch.setattr(FakeBackend, 'close', mutate)
    with pytest.raises(ValueError, match='changed'):
        runner.run_comparison(docs, windows, [{'arm_id': 'a', 'config_path': str(config)}], tmp_path / 'run',
                              adapter_factory=lambda arm: FakeBackend(arm, []))
    assert not (tmp_path / 'run/manifest.json').exists()
    with pytest.raises(ValueError, match='identical'):
        runner.run_comparison(docs, windows, [{'arm_id': 'a', 'config': {}, 'comparison_context': {'warmup_text': 'one'}},
                                             {'arm_id': 'b', 'config': {}, 'comparison_context': {'warmup_text': 'two'}}],
                              tmp_path / 'different')
    assert not (tmp_path / 'different').exists()


def test_tokenizer_checkpoint_sources_are_rechecked_before_publication(tmp_path, monkeypatch):
    runner = importlib.import_module('research.comparison.runner')
    from research.comparison.backends import sha256
    docs, windows = fixture_inputs()
    checkpoint = tmp_path / 'tokenizer'
    checkpoint.mkdir()
    (checkpoint / 'tokenizer.json').write_text('{}')
    pinned = {'files': [{'path': 'tokenizer.json', 'sha256': sha256(b'{}')}]}
    data = json.dumps(pinned).encode()
    (checkpoint / 'manifest.json').write_bytes(data)
    identity = {'checkpoint': str(checkpoint), 'manifest_sha256': sha256(data), 'manifest': pinned, 'files': pinned['files']}
    def mutate(self):
        (checkpoint / 'tokenizer.json').write_text('changed')
    monkeypatch.setattr(FakeBackend, 'close', mutate)
    with pytest.raises(ValueError, match='tokenizer|artifact'):
        runner.run_comparison(docs, windows, [{'arm_id': 'a', 'config': {'native': True},
                                              'comparison_context': {'tokenizer': identity}}], tmp_path / 'run',
                              adapter_factory=lambda arm: FakeBackend(arm, []))
    assert not (tmp_path / 'run/manifest.json').exists()


@pytest.mark.parametrize('phase', ['load', 'close'])
def test_load_and_close_exceptions_produce_explicit_partial_status(tmp_path, monkeypatch, phase):
    def fail(*args):
        raise RuntimeError('broken backend')
    monkeypatch.setattr(FakeBackend, phase, fail)
    manifest, _ = execute(tmp_path)
    assert manifest['exit_code'] == 1
    assert manifest['result_count'] == 4
    if phase == 'load':
        assert all(arm['status'] == 'unavailable' for arm in manifest['arms'])
    else:
        assert all(arm['close_error'] == 'broken backend' for arm in manifest['arms'])


@pytest.mark.parametrize('damage', ['traversal', 'symlink', 'omitted_file', 'duplicate_result'])
def test_report_rejects_unsafe_or_incomplete_artifact_graph(tmp_path, damage):
    execute(tmp_path)
    report = importlib.import_module('research.comparison.report')
    from research.comparison.backends import sha256
    run = tmp_path / 'run'
    manifest = json.loads((run / 'manifest.json').read_text())
    if damage == 'traversal':
        (tmp_path / 'outside').write_text('outside')
        manifest['files'].append({'path': '../outside', 'sha256': sha256(b'outside')})
    elif damage == 'symlink':
        raw = next((run / 'raw').glob('*.json'))
        outside = tmp_path / 'outside'
        outside.write_bytes(raw.read_bytes())
        raw.unlink()
        raw.symlink_to(outside)
    elif damage == 'omitted_file':
        manifest['files'] = [f for f in manifest['files'] if f['path'] != 'results.jsonl']
    else:
        target = run / 'results.jsonl'
        rows = target.read_text().splitlines()
        target.write_text('\n'.join([rows[0], *rows[:-1]]) + '\n')
        next(f for f in manifest['files'] if f['path'] == 'results.jsonl')['sha256'] = sha256(target.read_bytes())
    (run / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        report.build_report(run, tmp_path / 'report')
    assert not (tmp_path / 'report').exists()


def test_report_reserves_output_outside_immutable_run(tmp_path):
    execute(tmp_path)
    report = importlib.import_module('research.comparison.report')
    with pytest.raises(ValueError, match='outside'):
        report.build_report(tmp_path / 'run', tmp_path / 'run/report')
    assert not (tmp_path / 'run/report').exists()


def test_report_preserves_complete_reference_jsonl_source_bytes(tmp_path):
    execute(tmp_path)
    report = importlib.import_module('research.comparison.report')
    doc = fixture_inputs()[0][0]
    source = json.dumps({'document_id': doc['document_id'], 'text_revision': doc['text_revision'],
                         'spans': [], 'coverage': [], 'provenance': {}}, indent=None) + '\r\n'
    reference = tmp_path / 'reference.jsonl'
    reference.write_bytes(source.encode())
    report.build_report(tmp_path / 'run', tmp_path / 'report', references=reference)
    manifest = json.loads((tmp_path / 'report/manifest.json').read_text())
    assert any((tmp_path / 'report' / file['path']).read_bytes() == source.encode() for file in manifest['files'])


@pytest.mark.parametrize('fail_second', [False, True])
def test_multichunk_offsets_and_failure_discard_whole_document(tmp_path, fail_second):
    runner = importlib.import_module('research.comparison.runner')
    docs, _ = fixture_inputs()
    document = docs[0]
    windows = [{'window_id': identifier, 'document_id': document['document_id'],
                'text_revision': document['text_revision'], 'start': start, 'end': end,
                'text': document['text'][start:end], 'window_text_revision': text_revision(document['text'][start:end]),
                'content_tokens': 2} for identifier, start, end in [('first', 0, 6), ('second', 6, 12)]]
    class SpanBackend(FakeBackend):
        def predict(self, window):
            result = super().predict(window)
            if window['window_id'] == 'first':
                result.update(status='success', spans=[{'label': 'SOFTWARE', 'start': 0, 'end': 5,
                    'text': 'NumPy', 'score': None, 'score_kind': None, 'alignment_method': 'native_codepoint'}])
            if window['window_id'] == 'second':
                if fail_second:
                    result.update(status='failure', error='broken second chunk')
                else:
                    result.update(status='success', spans=[{'label': 'SOFTWARE', 'start': 0, 'end': 5,
                        'text': 'works', 'score': 0.3, 'score_kind': 'native', 'alignment_method': 'native_codepoint'}])
            return result
    runner.run_comparison([document], windows, [{'arm_id': 'a', 'config': {'native': True}}], tmp_path / 'run',
                          adapter_factory=lambda arm: SpanBackend(arm, []))
    result = json.loads((tmp_path / 'run/results.jsonl').read_text())
    assert result['status'] == ('failure' if fail_second else 'success')
    assert [(s['start'], s['end'], s['text']) for s in result['spans']] == ([] if fail_second else [(0, 5, 'NumPy'), (6, 11, 'works')])
    assert len(result['chunks']) == 2
    report = importlib.import_module('research.comparison.report')
    assert report.build_report(tmp_path / 'run', tmp_path / 'report')['status'] == 'reported'


def test_base_only_run_has_null_reference_metrics_and_explicit_no_quality_claim(tmp_path):
    runner = importlib.import_module('research.comparison.runner')
    report = importlib.import_module('research.comparison.report')
    docs, windows = fixture_inputs()
    manifest = runner.run_comparison(docs, windows, [{'arm_id': 'scibert-base', 'backend': 'base', 'config': {}}], tmp_path / 'run')
    assert manifest['status'] == 'complete'
    reference = tmp_path / 'refs.jsonl'
    reference.write_text(json.dumps({'document_id': docs[0]['document_id'], 'text_revision': docs[0]['text_revision'],
                                    'spans': [{'label': 'SOFTWARE', 'start': 0, 'end': 5, 'text': 'NumPy'}],
                                    'coverage': [{'label': 'SOFTWARE', 'start': 0, 'end': 12, 'complete': True,
                                                  'review_kind': 'human_reviewed'}], 'provenance': {}}) + '\n')
    result = report.build_report(tmp_path / 'run', tmp_path / 'report', references=reference)
    labels = result['references']['human_reviewed']['arms']['scibert-base']['labels']
    assert labels['SOFTWARE']['operational'] == {'tp': None, 'fp': None, 'fn': None, 'precision': None, 'recall': None, 'f1': None}
    assert result['heldout_quality_evaluated'] is False


def test_original_input_must_match_effective_run(tmp_path):
    runner = importlib.import_module('research.comparison.runner')
    from research.comparison.backends import sha256
    docs, windows = fixture_inputs()
    fake_source = json.dumps([{'not': 'the documents'}])
    source = {'path': None, 'bytes_utf8': fake_source, 'sha256': sha256(fake_source.encode())}
    with pytest.raises(ValueError, match='input source'):
        runner.run_comparison(docs, windows, [{'arm_id': 'base', 'backend': 'base', 'config': {},
                                             'comparison_context': {'input_source': source}}], tmp_path / 'run')
    assert not (tmp_path / 'run').exists()


def test_reference_report_refuses_failed_partial_manifest_with_false_status(tmp_path):
    execute(tmp_path, 'failing')
    report = importlib.import_module('research.comparison.report')
    run = tmp_path / 'run'
    manifest = json.loads((run / 'manifest.json').read_text())
    manifest.update(status='complete', exit_code=0)
    (run / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='status'):
        report.build_report(run, tmp_path / 'report')


@pytest.mark.parametrize('change', ['add_auxiliary', 'mutate_auxiliary', 'replace_with_symlink'])
def test_tokenizer_auxiliary_changes_block_terminal_publication(tmp_path, monkeypatch, change):
    runner = importlib.import_module('research.comparison.runner')
    from research.comparison.backends import sha256
    docs, windows = fixture_inputs()
    checkpoint = tmp_path / 'tokenizer'
    checkpoint.mkdir()
    names = ['tokenizer.json', 'tokenizer_config.json', 'config.json']
    if change != 'add_auxiliary':
        names.append('special_tokens_map.json')
    files = []
    for name in names:
        (checkpoint / name).write_text('{}')
        files.append({'path': name, 'sha256': sha256(b'{}')})
    pinned = {'schema_version': 'detector-checkpoint-1', 'files': files}
    data = json.dumps(pinned).encode()
    (checkpoint / 'manifest.json').write_bytes(data)
    identity = {'checkpoint': str(checkpoint), 'manifest_sha256': sha256(data), 'manifest': pinned, 'files': files}
    external = tmp_path / 'external.json'
    external.write_text('{}')
    def mutate(self):
        target = checkpoint / 'special_tokens_map.json'
        if change == 'replace_with_symlink':
            target.unlink()
            target.symlink_to(external)
        else:
            target.write_text('{"pad_token": "CHANGED"}')
    monkeypatch.setattr(FakeBackend, 'close', mutate)
    with pytest.raises(ValueError, match='tokenizer|artifact'):
        runner.run_comparison(docs, windows, [{'arm_id': 'a', 'config': {'native': True},
                                              'comparison_context': {'tokenizer': identity}}], tmp_path / 'run',
                              adapter_factory=lambda arm: FakeBackend(arm, []))
    assert not (tmp_path / 'run/manifest.json').exists()
