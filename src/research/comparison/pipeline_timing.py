"""Warmed, document-level timing of the full-label and Softcite pipelines."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import platform
import random
from statistics import median
import sys
import time

from ..contracts import validate_document
from ..data.artifacts import atomic_write_new
from .backends import json_text, resolve_arm
from .runner import code_identity


WARMUP_TEXT = 'We used NumPy.'
COMPLETED = {'success', 'no_mentions'}


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def _jsonl(rows):
    return ''.join(json_text(row) + '\n' for row in rows).encode('utf-8')


def _default_synchronize(device):
    if device == 'mps':
        import torch
        torch.mps.synchronize()


def _default_full_factory(path, device):
    from ..training.full_label import FullLabelPipeline
    return FullLabelPipeline(path, device)


def _default_soft_factory():
    from .backend_softcite import SoftciteBackend
    return SoftciteBackend()


def _timed(operation, clock, synchronize, device):
    try:
        synchronize(device)
    except Exception as exc:
        return None, f'presynchronization {type(exc).__name__}: {exc}', None
    start = clock()
    native, error = None, None
    try:
        native = operation()
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    try:
        synchronize(device)
    except Exception as exc:
        error = f'synchronization {type(exc).__name__}: {exc}'
    elapsed = clock() - start
    return native, error, elapsed


def _summary(rows):
    elapsed = [row['elapsed_seconds'] for row in rows if row['elapsed_seconds'] is not None]
    total = sum(elapsed)
    attempted = len(elapsed)
    successful = sum(row['status'] in COMPLETED for row in rows)
    return {'scheduled': len(rows), 'attempted': attempted, 'successful': successful,
            'failed': attempted - successful, 'not_attempted': len(rows) - attempted,
            'measured_seconds': total,
            'latency_median_seconds': median(elapsed) if elapsed else None,
            'latency_p95_seconds': sorted(elapsed)[math.ceil(.95 * attempted) - 1] if elapsed else None,
            'attempted_per_second': attempted / total if total > 0 else None,
            'failed_per_second': (attempted - successful) / total if total > 0 else None,
            'successful_per_second': successful / total if total > 0 else None}


def _hardware(device):
    details = {'platform': platform.platform(), 'machine': platform.machine(),
               'processor': platform.processor(), 'python': sys.version.split()[0],
               'device': device, 'torch': None, 'mps_available': None,
               'torch_threads': None, 'torch_interop_threads': None}
    try:
        import torch
        details['torch'] = torch.__version__
        details['mps_available'] = torch.backends.mps.is_available()
        details['torch_threads'] = torch.get_num_threads()
        details['torch_interop_threads'] = torch.get_num_interop_threads()
    except ImportError:
        pass
    return details


def run_benchmark(input_path, checkpoints, softcite_config, output, *, device, repeats=3,
                  batch_size=1, full_factory=None, soft_factory=None,
                  clock=None, synchronize=None):
    """Run each arm on the same seeded order; publish a new immutable directory."""
    if device not in ('cpu', 'mps'):
        raise ValueError('explicit device must be cpu or mps')
    if type(repeats) is not int or not 1 <= repeats <= 20:
        raise ValueError('repeats must be between 1 and 20')
    if batch_size != 1:
        raise ValueError('full-pipeline comparison requires batch size 1')
    if not isinstance(checkpoints, dict) or len(checkpoints) > 8:
        raise ValueError('zero to eight named checkpoints required')
    if any(not isinstance(name, str) or not name.strip() or name == 'softcite' for name in checkpoints):
        raise ValueError('checkpoint names must be nonblank and distinct from softcite')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    hardware = _hardware(device)
    if device == 'mps' and not hardware['mps_available']:
        raise ValueError('MPS is unavailable')
    input_path, softcite_config = Path(input_path), Path(softcite_config)
    input_bytes, config_bytes = input_path.read_bytes(), softcite_config.read_bytes()
    documents = [validate_document(json.loads(line)) for line in input_bytes.decode('utf-8').splitlines()]
    if not documents or len(documents) > 1000 or sum(len(row['text']) for row in documents) > 5_000_000:
        raise ValueError('input must contain 1-1000 documents and at most 5 million characters')
    identities = [(row['document_id'], row['text_revision']) for row in documents]
    if len(set(identities)) != len(identities) or len({row['document_id'] for row in documents}) != len(documents):
        raise ValueError('duplicate document identity')
    config = json.loads(config_bytes)
    if not isinstance(config, dict):
        raise ValueError('Softcite config must be an object')
    checkpoints = {name: Path(path) for name, path in checkpoints.items()}
    checkpoint_hashes = {name: _sha256((path / 'manifest.json').read_bytes()) for name, path in checkpoints.items()}
    order = list(range(len(documents)))
    random.Random(42).shuffle(order)
    clock = clock or time.perf_counter
    synchronize = synchronize or _default_synchronize
    full_factory = full_factory or _default_full_factory
    soft_factory = soft_factory or _default_soft_factory
    output.mkdir(parents=True, exist_ok=False)
    requests, first_pass, arms = [], [], []
    warmup = validate_document({'document_id': 'synthetic-warmup', 'text': WARMUP_TEXT})
    for arm_id in [*checkpoints, 'softcite']:
        backend = None
        loaded, load_error = None, None
        if arm_id == 'softcite':
            arm = {'arm_id': 'softcite', 'backend': 'softcite', 'config': config,
                   'config_source': {'path': str(softcite_config.resolve()),
                                     'sha256': _sha256(config_bytes),
                                     'bytes_utf8': config_bytes.decode('utf-8')}}
            def load():
                nonlocal backend
                backend = soft_factory()
                frozen = resolve_arm(arm)
                result = backend.load(frozen)
                resolve_arm(frozen)
                return result
            load_semantics = 'service_health_check_only'
        else:
            path = checkpoints[arm_id]
            def load():
                nonlocal backend
                if _sha256((path / 'manifest.json').read_bytes()) != checkpoint_hashes[arm_id]:
                    raise ValueError('checkpoint source changed before load')
                backend = full_factory(path, device)
                if (backend.identity != checkpoint_hashes[arm_id]
                        or _sha256((path / 'manifest.json').read_bytes()) != checkpoint_hashes[arm_id]):
                    raise ValueError('checkpoint source changed during load')
                return {'status': 'ready', 'reason': None, 'capabilities': backend.manifest['capabilities'],
                        'identity': {'checkpoint_sha256': backend.identity,
                                     'stages': {stage: {'status': entry['status'],
                                                        'manifest_sha256': entry.get('manifest_sha256'),
                                                        'unavailable_reasons': entry.get('unavailable_reasons')}
                                                for stage, entry in backend.manifest.get('stages', {}).items()}}}
            load_semantics = 'local_checkpoint_initialization'
        loaded, load_error, load_seconds = _timed(load, clock, synchronize, device)
        if load_error:
            loaded = {'status': 'unavailable', 'reason': load_error,
                      'capabilities': None, 'identity': {}}
        if not isinstance(loaded, dict) or loaded.get('status') not in ('ready', 'unavailable', 'unsupported'):
            loaded = {'status': 'unavailable', 'reason': 'invalid backend load result',
                      'capabilities': None, 'identity': {}}
        arm_record = {'arm_id': arm_id, 'load_seconds': load_seconds,
                      'load_semantics': load_semantics, 'load_status': loaded['status'],
                      'load_reason': loaded.get('reason'), 'identity': loaded.get('identity'),
                      'capabilities': loaded.get('capabilities'),
                      'checkpoint_manifest_sha256': checkpoint_hashes.get(arm_id),
                      'warmup_seconds': None, 'warmup_status': None, 'warmup_error': None}
        try:
            if loaded['status'] == 'ready':
                def predict(document):
                    if arm_id == 'softcite':
                        return backend.predict({'window_id': document['document_id'],
                            'text': document['text'], 'window_text_revision': document['text_revision']})
                    return backend.predict(deepcopy(document))
                warm_native, warm_error, warm_seconds = _timed(lambda: predict(warmup), clock, synchronize, device)
                arm_record.update(warmup_seconds=warm_seconds,
                                  warmup_status=warm_native.get('status') if isinstance(warm_native, dict) else 'failure',
                                  warmup_error=warm_error)
            measurement_ready = (loaded['status'] == 'ready' and arm_record['warmup_error'] is None
                                 and arm_record['warmup_status'] in COMPLETED)
            arm_rows = []
            for repeat in range(repeats):
                for position in order:
                    document = documents[position]
                    if measurement_ready:
                        native, error, elapsed = _timed(lambda: predict(document), clock, synchronize, device)
                        expected = document['document_id']
                        actual = native.get('window_id' if arm_id == 'softcite' else 'document_id') if isinstance(native, dict) else None
                        status = native.get('status') if isinstance(native, dict) else None
                        if error is None and (actual != expected or status not in COMPLETED | {'failure', 'partial'}):
                            error = 'invalid native result identity or status'
                        if error is None and arm_id != 'softcite' and native.get('text_revision') != document['text_revision']:
                            error = 'native text revision mismatch'
                        status = status if error is None else 'failure'
                        if status not in COMPLETED:
                            error = error or native.get('error') or native.get('reason') or status
                        if repeat == 0:
                            first_pass.append({'arm_id': arm_id, 'document_id': expected,
                                'input_sha256': document['text_revision'], 'native': native,
                                'measurement_error': error})
                    else:
                        native, elapsed = None, None
                        status = loaded['status'] if loaded['status'] != 'ready' else 'unavailable'
                        error = loaded.get('reason') or 'synthetic warmup did not complete successfully'
                    arm_rows.append({'arm_id': arm_id, 'repeat': repeat + 1,
                        'document_id': document['document_id'], 'text_revision': document['text_revision'],
                        'input_sha256': document['text_revision'], 'status': status,
                        'elapsed_seconds': elapsed, 'error': error})
            requests.extend(arm_rows)
            arm_record['summary'] = _summary(arm_rows)
        finally:
            if backend is not None and hasattr(backend, 'close'):
                try:
                    backend.close()
                except Exception as exc:
                    arm_record['close_error'] = f'{type(exc).__name__}: {exc}'
        arms.append(arm_record)
    files = []
    for name, payload in [('inputs.jsonl', _jsonl(documents)),
                          ('requests.jsonl', _jsonl(requests)),
                          ('first_pass.jsonl', _jsonl(first_pass)),
                          ('softcite-config.json', config_bytes)]:
        atomic_write_new(output / name, payload)
        files.append({'path': name, 'sha256': _sha256(payload)})
    manifest = {'schema_version': 'pipeline-timing-1', 'seed': 42,
                'timing_policy': 'full document predict including tokenization, extraction, linking and output construction; accelerator synchronized; load, synthetic warmup and publication excluded',
                'repeats': repeats, 'batch_size': batch_size, 'order': [documents[i]['document_id'] for i in order],
                'document_count': len(documents), 'input_file_sha256': _sha256(_jsonl(documents)),
                'source_input_file_sha256': _sha256(input_bytes),
                'softcite_config_sha256': _sha256(config_bytes), 'input_path': str(input_path.resolve()),
                'hardware': hardware, 'code': code_identity(), 'arms': arms, 'files': files}
    atomic_write_new(output / 'manifest.json', (json_text(manifest) + '\n').encode('utf-8'))
    return manifest
