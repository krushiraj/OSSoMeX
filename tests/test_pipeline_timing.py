import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from research.contracts import text_revision
from research.comparison.pipeline_timing import run_benchmark


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def fixture_run(tmp_path, *, repeats=2, full_status='success', soft_status='success', ready=True,
                mutate_config=False, mutate_checkpoint=False, warm_status='success'):
    source = tmp_path / 'input.jsonl'
    documents = [{'document_id': 'a', 'text': 'Alpha uses R.'},
                 {'document_id': 'b', 'text': 'Beta uses Python.'}]
    source.write_text(''.join(json.dumps(row) + '\n' for row in documents))
    checkpoint = tmp_path / 'model'
    checkpoint.mkdir()
    (checkpoint / 'manifest.json').write_text('{}')
    config = tmp_path / 'softcite.json'
    config.write_text('{}')
    clock = Clock()
    sync_calls = 0

    def synchronize(device):
        nonlocal sync_calls
        assert device == 'cpu'
        clock.advance(13 if sync_calls % 2 == 0 else 1)
        sync_calls += 1

    class Full:
        manifest = {'capabilities': {'software_spans': True, 'version_spans': True,
                                     'intent': False},
                    'stages': {'detector': {'status': 'available', 'manifest_sha256': 'stage-digest'}}}

        def __init__(self, path, device):
            assert path == checkpoint and device == 'cpu'
            if mutate_checkpoint:
                (checkpoint / 'manifest.json').write_text('{"changed":true}')
            self.identity = hashlib.sha256((checkpoint / 'manifest.json').read_bytes()).hexdigest()
            clock.advance(2)

        def predict(self, document):
            clock.advance(5)
            return {'document_id': document['document_id'],
                    'text_revision': document['text_revision'], 'status': full_status,
                    'native_full': document['text']}

    class Soft:
        def __init__(self):
            if mutate_config:
                config.write_text('{"changed":true}')

        def load(self, arm):
            assert 'config_path' not in arm
            assert arm['config'] == {}
            assert arm['config_source']['sha256'] == hashlib.sha256(b'{}').hexdigest()
            clock.advance(7)
            return {'status': 'ready' if ready else 'unavailable',
                    'reason': None if ready else 'service offline',
                    'capabilities': {'software_spans': True, 'version_spans': True,
                                     'intent': False}, 'identity': {'model_id': 'softcite-test'}}

        def predict(self, window):
            clock.advance(11)
            if window['window_id'] == 'synthetic-warmup':
                return {'window_id': window['window_id'], 'status': warm_status}
            if soft_status == 'raise':
                raise RuntimeError('transport down')
            return {'window_id': window['window_id'], 'status': soft_status,
                    'raw': {'native': {'text': window['text']}}, 'spans': [], 'unresolved': [],
                    'error': 'inference failed' if soft_status == 'failure' else None}

        def close(self):
            pass

    output = tmp_path / 'run'
    manifest = run_benchmark(source, {'full': checkpoint}, config, output, device='cpu',
                             repeats=repeats, full_factory=Full, soft_factory=Soft,
                             clock=clock, synchronize=synchronize)
    rows = [json.loads(line) for line in (output / 'requests.jsonl').read_text().splitlines()]
    first = [json.loads(line) for line in (output / 'first_pass.jsonl').read_text().splitlines()]
    return manifest, rows, first, documents, output


def test_published_input_hash_matches_frozen_file_and_preserves_source_hash(tmp_path):
    manifest, _, _, _, output = fixture_run(tmp_path)
    if manifest['hardware']['torch'] is not None:
        assert manifest['hardware']['torch_threads'] > 0
        assert manifest['hardware']['torch_interop_threads'] > 0
    else:
        assert manifest['hardware']['torch_threads'] is None
        assert manifest['hardware']['torch_interop_threads'] is None
    assert manifest['input_file_sha256'] == hashlib.sha256((output / 'inputs.jsonl').read_bytes()).hexdigest()
    assert manifest['source_input_file_sha256'] == hashlib.sha256((tmp_path / 'input.jsonl').read_bytes()).hexdigest()
    assert manifest['input_file_sha256'] != manifest['source_input_file_sha256']


def test_failed_warmup_blocks_warmed_measurements(tmp_path):
    manifest, rows, first, _, _ = fixture_run(tmp_path, warm_status='failure')
    soft = next(arm for arm in manifest['arms'] if arm['arm_id'] == 'softcite')
    assert soft['load_status'] == 'ready' and soft['warmup_status'] == 'failure'
    assert soft['summary']['attempted'] == 0 and soft['summary']['not_attempted'] == 4
    assert all(row['status'] == 'unavailable' and row['elapsed_seconds'] is None
               and 'warmup' in row['error'] for row in rows if row['arm_id'] == 'softcite')
    assert all(row['arm_id'] != 'softcite' for row in first)


def test_timing_excludes_load_warmup_and_serialization_and_pairs_exact_text(tmp_path):
    manifest, rows, first, documents, _ = fixture_run(tmp_path)
    by_arm = {arm['arm_id']: arm for arm in manifest['arms']}
    assert by_arm['full']['load_seconds'] == 3
    assert by_arm['full']['identity']['stages']['detector']['manifest_sha256'] == 'stage-digest'
    assert by_arm['softcite']['load_seconds'] == 8
    assert by_arm['full']['warmup_seconds'] == 6
    assert by_arm['softcite']['warmup_seconds'] == 12
    assert {row['elapsed_seconds'] for row in rows if row['arm_id'] == 'full'} == {6}
    assert {row['elapsed_seconds'] for row in rows if row['arm_id'] == 'softcite'} == {12}
    assert len(first) == 4
    for document in documents:
        outputs = [row for row in first if row['document_id'] == document['document_id']]
        assert {row['arm_id'] for row in outputs} == {'full', 'softcite'}
        assert outputs[0]['input_sha256'] == outputs[1]['input_sha256']
        assert next(row for row in outputs if row['arm_id'] == 'full')['native']['native_full'] == document['text']
        assert next(row for row in outputs if row['arm_id'] == 'softcite')['native']['raw']['native']['text'] == document['text']
    assert all(row['text_revision'] == text_revision(next(d['text'] for d in documents if d['document_id'] == row['document_id'])) for row in rows)


def test_checkpoint_change_during_load_is_unavailable_without_loaded_identity(tmp_path):
    manifest, rows, first, _, _ = fixture_run(tmp_path, mutate_checkpoint=True)
    full = next(arm for arm in manifest['arms'] if arm['arm_id'] == 'full')
    assert full['load_status'] == 'unavailable'
    assert 'source changed' in full['load_reason']
    assert full['identity'] == {}
    assert full['summary']['not_attempted'] == 4
    assert all(row['status'] == 'unavailable' for row in rows if row['arm_id'] == 'full')
    assert all(row['arm_id'] != 'full' for row in first)


def test_config_change_during_backend_creation_is_unavailable_without_loaded_identity(tmp_path):
    manifest, rows, first, _, output = fixture_run(tmp_path, mutate_config=True)
    soft = next(arm for arm in manifest['arms'] if arm['arm_id'] == 'softcite')
    assert soft['load_status'] == 'unavailable'
    assert 'source changed' in soft['load_reason']
    assert soft['identity'] == {}
    assert soft['summary']['not_attempted'] == 4
    assert all(row['status'] == 'unavailable' for row in rows if row['arm_id'] == 'softcite')
    assert all(row['arm_id'] != 'softcite' for row in first)
    assert (output / 'softcite-config.json').read_bytes() == b'{}'


def test_failed_native_results_remain_attempts_and_reduce_successful_throughput(tmp_path):
    manifest, rows, first, _, _ = fixture_run(tmp_path, soft_status='failure')
    soft = next(arm for arm in manifest['arms'] if arm['arm_id'] == 'softcite')
    assert soft['summary']['attempted'] == 4
    assert soft['summary']['successful'] == 0
    assert soft['summary']['failed'] == 4
    assert soft['summary']['attempted_per_second'] == pytest.approx(4 / 48)
    assert soft['summary']['failed_per_second'] == pytest.approx(4 / 48)
    assert soft['summary']['successful_per_second'] == 0
    assert all(row['status'] == 'failure' and row['error'] == 'inference failed' for row in rows if row['arm_id'] == 'softcite')
    assert all(row['native']['status'] == 'failure' for row in first if row['arm_id'] == 'softcite')


def test_unavailable_backend_has_no_inference_attempts_or_fabricated_capabilities(tmp_path):
    manifest, rows, first, _, _ = fixture_run(tmp_path, ready=False)
    soft = next(arm for arm in manifest['arms'] if arm['arm_id'] == 'softcite')
    assert soft['summary']['attempted'] == 0
    assert soft['summary']['not_attempted'] == 4
    assert soft['summary']['successful_per_second'] is None
    assert soft['capabilities']['intent'] is False
    assert soft['load_semantics'] == 'service_health_check_only'
    assert all(row['status'] == 'unavailable' and row['elapsed_seconds'] is None for row in rows if row['arm_id'] == 'softcite')
    assert all(row['arm_id'] != 'softcite' for row in first)


def test_predict_exception_keeps_duration_and_first_pass_error(tmp_path):
    manifest, rows, first, _, _ = fixture_run(tmp_path, soft_status='raise')
    soft = next(arm for arm in manifest['arms'] if arm['arm_id'] == 'softcite')
    assert soft['summary']['attempted'] == 4
    assert soft['summary']['failed'] == 4
    assert all(row['elapsed_seconds'] == 12 and 'transport down' in row['error'] for row in rows if row['arm_id'] == 'softcite')
    assert all(row['native'] is None and 'transport down' in row['measurement_error'] for row in first if row['arm_id'] == 'softcite')


def test_output_directory_is_immutable_and_rejects_invalid_bounds(tmp_path):
    _, _, _, _, output = fixture_run(tmp_path)
    source = tmp_path / 'input.jsonl'
    config = tmp_path / 'softcite.json'
    checkpoint = tmp_path / 'model'
    before = (output / 'manifest.json').read_bytes()
    with pytest.raises(FileExistsError):
        run_benchmark(source, {'full': checkpoint}, config, output, device='cpu')
    assert (output / 'manifest.json').read_bytes() == before
    with pytest.raises(ValueError, match='repeats'):
        run_benchmark(source, {'full': checkpoint}, config, tmp_path / 'invalid', device='cpu', repeats=0)
    assert not (tmp_path / 'invalid').exists()


def test_seeded_order_is_shared_and_request_exceptions_are_counted(tmp_path):
    manifest, rows, _, documents, _ = fixture_run(tmp_path)
    expected = manifest['order']
    assert sorted(expected) == sorted(row['document_id'] for row in documents)
    for arm_id in ('full', 'softcite'):
        for repeat in (1, 2):
            assert [row['document_id'] for row in rows if row['arm_id'] == arm_id and row['repeat'] == repeat] == expected


def test_softcite_only_uses_same_warmed_batch_one_schedule(tmp_path):
    source = tmp_path / 'input.jsonl'
    source.write_text(json.dumps({'document_id': 'a', 'text': 'We used NumPy.'}) + '\n')
    config = tmp_path / 'softcite.json'
    config.write_text('{}')

    class Soft:
        def load(self, arm):
            return {'status': 'ready', 'reason': None, 'capabilities': {}, 'identity': {}}

        def predict(self, window):
            return {'window_id': window['window_id'], 'status': 'success'}

        def close(self):
            pass

    manifest = run_benchmark(source, {}, config, tmp_path / 'run', device='cpu', repeats=3,
                             soft_factory=Soft, synchronize=lambda device: None)
    assert [arm['arm_id'] for arm in manifest['arms']] == ['softcite']
    assert manifest['arms'][0]['warmup_status'] == 'success'
    assert manifest['arms'][0]['summary']['scheduled'] == 3
    assert manifest['arms'][0]['summary']['successful'] == 3


def test_cli_requires_device_and_bounded_repeats(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'benchmark_pipeline.py'
    command = [sys.executable, str(script), '--input', str(tmp_path / 'input.jsonl'),
               '--checkpoint', 'full=' + str(tmp_path / 'model'), '--softcite-config',
               str(tmp_path / 'softcite.json'), '--output', str(tmp_path / 'out')]
    missing = subprocess.run(command, capture_output=True, text=True)
    assert missing.returncode != 0
    assert '--device' in missing.stderr
    invalid = subprocess.run([*command, '--device', 'cpu', '--repeats', '0'], capture_output=True, text=True)
    assert invalid.returncode != 0
    assert 'repeats' in invalid.stderr
    assert not (tmp_path / 'out').exists()
