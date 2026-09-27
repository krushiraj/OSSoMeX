"""Sequential comparisons with immutable, complete frozen evidence."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import time

from ..contracts import text_revision
from ..data.artifacts import atomic_write_new
from ..data.manifest import verified_path
from .alignment import reduce_windows
from .backends import REPO_ROOT, capabilities, json_text, make_backend, resolve_arm, sha256, window_result
from .contracts import CAPABILITY_FIELDS, OFFSET_UNIT, SCHEMA_VERSION, validate_input, validate_result


WARMUP_TEXT = 'We used NumPy.'


def json_bytes(value):
    return (json_text(value) + '\n').encode('utf-8')


def jsonl_bytes(rows):
    return b''.join(json_bytes(row) for row in rows)


def publish(output, files, path, payload):
    atomic_write_new(output / path, payload)
    files.append({'path': path, 'sha256': sha256(payload)})


def artifact_name(*identities):
    return sha256(json_text(identities).encode('utf-8'))


def validate_population(documents, windows):
    if not isinstance(documents, list) or not documents:
        raise ValueError('nonempty documents required; empty inputs provide no comparison evidence')
    documents = [validate_input(document) for document in documents]
    by_id = {document['document_id']: document for document in documents}
    if len(by_id) != len(documents):
        raise ValueError('duplicate document identity')
    if not isinstance(windows, list):
        raise ValueError('windows must be a list')
    groups, seen = {key: [] for key in by_id}, set()
    for window in windows:
        if not isinstance(window, dict):
            raise ValueError('invalid frozen window')
        document = by_id.get(window.get('document_id'))
        identifier, start, end = window.get('window_id'), window.get('start'), window.get('end')
        if (document is None or not isinstance(identifier, str) or not identifier.strip() or identifier in seen
                or type(start) is not int or type(end) is not int or not 0 <= start < end <= len(document['text'])
                or window.get('text_revision') != document['text_revision']
                or window.get('text') != document['text'][start:end]
                or window.get('window_text_revision') != text_revision(window['text'])
                or type(window.get('content_tokens')) is not int or not 0 <= window['content_tokens'] <= 480):
            raise ValueError('invalid, duplicate, or mismatched frozen window')
        seen.add(identifier)
        groups[document['document_id']].append(window)
    for document_id, group in groups.items():
        end = 0
        for window in sorted(group, key=lambda item: (item['start'], item['end'])):
            if window['start'] > end:
                raise ValueError('incomplete frozen window coverage')
            end = max(end, window['end'])
        if not group or end != len(by_id[document_id]['text']):
            raise ValueError('incomplete frozen window population')
    return documents, deepcopy(windows)


def prepare_arms(arms):
    if not isinstance(arms, list) or not arms:
        raise ValueError('nonempty arms required')
    resolved = [resolve_arm(arm) for arm in arms]
    ids = [arm.get('arm_id') for arm in resolved]
    if any(not isinstance(value, str) or not value.strip() for value in ids) or len(set(ids)) != len(ids):
        raise ValueError('unique nonblank arm_id required')
    context = resolved[0].get('comparison_context', {})
    if not isinstance(context, dict) or any(json_text(arm.get('comparison_context', {})) != json_text(context) for arm in resolved):
        raise ValueError('comparison_context must be identical across arms')
    warmup = context.get('warmup_text', WARMUP_TEXT)
    if not isinstance(warmup, str) or not warmup.strip():
        raise ValueError('nonempty synthetic warmup_text required')
    for key in ('config_source', 'input_source'):
        source = context.get(key)
        if source is not None:
            verify_source(source)
    verify_tokenizer(context.get('tokenizer'))
    return resolved, deepcopy(context)


def verify_source(source):
    data = source['bytes_utf8'].encode('utf-8')
    if sha256(data) != source['sha256']:
        raise ValueError('source hash mismatch')
    if source.get('path') is not None and Path(source['path']).read_bytes() != data:
        raise ValueError('source changed since freezing')


def verify_tokenizer(identity):
    if identity is None or 'checkpoint' not in identity:
        return
    root = Path(identity['checkpoint'])
    data = (root / 'manifest.json').read_bytes()
    if sha256(data) != identity['manifest_sha256'] or json.loads(data) != identity['manifest']:
        raise ValueError('tokenizer manifest changed since freezing')
    if identity['files'] != identity['manifest']['files']:
        raise ValueError('tokenizer file identity mismatch')
    for record in identity['files']:
        verified_path(root, record)


def freeze_configuration(output, files, arms, context):
    publish(output, files, 'arms.json', json_bytes(arms))
    for arm in arms:
        name = artifact_name(arm['arm_id'])
        publish(output, files, f'configs/{name}.json', arm['config_source']['bytes_utf8'].encode('utf-8'))
    for key in ('config_source', 'input_source'):
        if context.get(key) is not None:
            publish(output, files, f'sources/{key}.txt', context[key]['bytes_utf8'].encode('utf-8'))
    publish(output, files, 'context.json', json_bytes(context))


def code_identity():
    def git(*args):
        return subprocess.check_output(['git', '-C', str(REPO_ROOT), *args], text=True).strip()
    try:
        return {'commit': git('rev-parse', 'HEAD'), 'dirty': bool(git('status', '--porcelain')),
                'status_porcelain': git('status', '--porcelain')}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {'commit': None, 'dirty': None, 'error': str(exc)}


def validate_loaded(loaded):
    if not isinstance(loaded, dict) or loaded.get('status') not in ('ready', 'unavailable', 'unsupported'):
        raise ValueError('invalid backend load status')
    caps = loaded.get('capabilities')
    if (not isinstance(caps, dict) or set(caps) != set(CAPABILITY_FIELDS)
            or any(type(value) is not bool for value in caps.values())
            or any(caps[key] for key in CAPABILITY_FIELDS if key not in ('software_spans', 'version_spans'))):
        raise ValueError('invalid backend capabilities')
    if not isinstance(loaded.get('identity'), dict) or loaded['identity'].get('config_verified') is False:
        raise ValueError('unverified backend configuration')
    if loaded['status'] != 'ready' and (not isinstance(loaded.get('reason'), str) or not loaded['reason'].strip()):
        raise ValueError('inactive backend requires reason')
    if loaded['status'] == 'unsupported' and any(caps.values()):
        raise ValueError('unsupported backend cannot claim capabilities')
    return deepcopy(loaded)


def load_backend(arm, adapter_factory, clock):
    start = clock()
    backend = None
    try:
        backend = adapter_factory(deepcopy(arm))
        loaded = validate_loaded(backend.load(deepcopy(arm)))
    except Exception as exc:
        loaded = {'status': 'unavailable', 'reason': str(exc) or type(exc).__name__,
                  'capabilities': capabilities(False), 'identity': {'load_exception': type(exc).__name__}}
    return backend, loaded, clock() - start


def predict_window(backend, window):
    try:
        native = backend.predict(deepcopy(window))
    except Exception as exc:
        native = window_result(window, error=str(exc) or type(exc).__name__,
                               raw={'exception': type(exc).__name__, 'detail': str(exc)})
    envelope = deepcopy(native) if isinstance(native, dict) else {'native_result': native}
    envelope['window'] = deepcopy(window)
    return envelope


def freeze_prompts(output, files, arm_id, identity):
    for key, source in identity.get('prompt_sources', {}).items():
        if not isinstance(source, str):
            raise ValueError('prompt source must preserve exact UTF-8 text')
        payload = source.encode('utf-8')
        expected = identity.get('prompt_source_sha256', {}).get(key)
        if expected is not None and expected != sha256(payload):
            raise ValueError('prompt source hash mismatch')
        publish(output, files, f'prompts/{artifact_name(arm_id, key)}.txt', payload)


def inactive_result(document, arm_id, loaded, raw_path):
    return {'schema_version': SCHEMA_VERSION, 'arm_id': arm_id,
            'document_id': document['document_id'], 'text_revision': document['text_revision'],
            'status': loaded['status'], 'reason': loaded['reason'], 'offset_unit': OFFSET_UNIT,
            'capabilities': deepcopy(loaded['capabilities']), 'scores_calibrated': False,
            'spans': [], 'unresolved': [], 'chunks': [], 'raw_artifact': raw_path,
            'provenance': {'preflight': True}}


def measure_arm(backend, arm, loaded, documents, windows, output, files, clock, warmup_text):
    arm_id = arm['arm_id']
    timing = {'warmup_seconds': None, 'measured_seconds': None, 'windows_per_second': None,
              'documents_per_second': None}
    warmup, predictions, paths = None, [], {}
    if loaded['status'] == 'ready':
        warm = {'window_id': 'synthetic-warmup', 'document_id': 'synthetic-warmup',
                'text': warmup_text, 'text_revision': text_revision(warmup_text),
                'window_text_revision': text_revision(warmup_text), 'start': 0, 'end': len(warmup_text),
                'content_tokens': 0, 'synthetic': True}
        start = clock()
        warmup = predict_window(backend, warm)
        timing['warmup_seconds'] = clock() - start
        warmup = reduce_windows({'document_id': warm['document_id'], 'text': warmup_text,
                                 'text_revision': warm['text_revision']}, arm_id, [warmup], loaded['capabilities'])['chunks'][0]
        publish(output, files, f'raw/{artifact_name(arm_id, "warmup")}.json', json_bytes(warmup))
        start = clock()
        for window in windows:
            result = predict_window(backend, window)
            predictions.append(result)
            path = f'raw/{artifact_name(arm_id, "window", window["window_id"])}.json'
            publish(output, files, path, json_bytes(result))
            paths[window['window_id']] = path
        timing['measured_seconds'] = clock() - start
        if timing['measured_seconds'] > 0:
            timing['windows_per_second'] = len(windows) / timing['measured_seconds']
            timing['documents_per_second'] = len(documents) / timing['measured_seconds']
    results = []
    for document in documents:
        expected = [w['window_id'] for w in windows if w['document_id'] == document['document_id']]
        chunks = [p for p in predictions if p['window']['document_id'] == document['document_id']]
        index = f'raw/{artifact_name(arm_id, "document", document["document_id"])}.json'
        publish(output, files, index, json_bytes({'arm_id': arm_id, 'document_id': document['document_id'],
                'expected_window_ids': expected, 'windows': [{'window_id': key, 'path': paths[key]} for key in expected if key in paths],
                'preflight': loaded}))
        if loaded['status'] != 'ready':
            result = inactive_result(document, arm_id, loaded, index)
        else:
            result = reduce_windows(document, arm_id, chunks, loaded['capabilities'])
            actual = [chunk.get('window_id') for chunk in chunks]
            if actual != expected:
                result.update(status='failure', spans=[], reason='expected_window_population_mismatch')
            result['raw_artifact'] = index
        results.append(validate_result(document, result))
    return results, timing, warmup


def verify_files(output, files):
    for record in files:
        if sha256((output / record['path']).read_bytes()) != record['sha256']:
            raise ValueError('artifact hash changed before publication')


def run_is_partial(results, arm_records):
    return any(row['status'] in ('failure', 'unavailable') for row in results) or any(
        row.get('close_error') or (row.get('warmup') is not None and row['warmup'].get('status') not in ('success', 'no_mentions'))
        for row in arm_records)


def run_comparison(documents: list[dict], windows: list[dict], arms: list[dict], output: Path,
                   *, adapter_factory=make_backend, clock=time.monotonic) -> dict:
    documents, windows = validate_population(documents, windows)
    arms, context = prepare_arms(arms)
    if context.get('input_source') is not None:
        original = [json.loads(line) for line in context['input_source']['bytes_utf8'].splitlines()]
        if json_text(original) != json_text(documents):
            raise ValueError('input source differs from frozen documents')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    files, results, arm_records = [], [], []
    code = code_identity()
    publish(output, files, 'inputs.jsonl', jsonl_bytes(documents))
    publish(output, files, 'windows.jsonl', jsonl_bytes(windows))
    freeze_configuration(output, files, arms, context)
    warmup_text = context.get('warmup_text', WARMUP_TEXT)
    for arm in arms:
        backend, loaded, load_seconds = load_backend(arm, adapter_factory, clock)
        record = {'arm_id': arm['arm_id'], **loaded}
        try:
            freeze_prompts(output, files, arm['arm_id'], loaded['identity'])
            rows, timing, warmup = measure_arm(backend, arm, loaded, documents, windows, output, files, clock, warmup_text)
            record.update(timing={'load_seconds': load_seconds, **timing}, warmup=warmup)
            results.extend(rows)
        finally:
            if backend is not None:
                try:
                    backend.close()
                except Exception as exc:
                    record['close_error'] = str(exc) or type(exc).__name__
        arm_records.append(record)
    expected = {(arm['arm_id'], document['document_id']) for arm in arms for document in documents}
    if len(results) != len(expected) or {(row['arm_id'], row['document_id']) for row in results} != expected:
        raise ValueError('incomplete arm/document result population')
    publish(output, files, 'preflight.json', json_bytes(arm_records))
    publish(output, files, 'results.jsonl', jsonl_bytes(results))
    partial = run_is_partial(results, arm_records)
    manifest = {'schema_version': 'comparison-run-1', 'status': 'partial' if partial else 'complete',
                'exit_code': int(partial), 'document_count': len(documents), 'window_count': len(windows),
                'result_count': len(results), 'arm_order': [arm['arm_id'] for arm in arms], 'arms': arm_records,
                'code': code, 'warmup': {'text': warmup_text, 'synthetic': True, 'included_in_measured': False},
                'timing_policy': 'sequential arms; measured includes inference and raw publication, excludes load/warmup/reduction',
                'provenance': context, 'files': files}
    for arm in arms:
        resolve_arm(arm)
    for key in ('config_source', 'input_source'):
        if context.get(key) is not None:
            verify_source(context[key])
    verify_tokenizer(context.get('tokenizer'))
    verify_files(output, files)
    atomic_write_new(output / 'manifest.json', json_bytes(manifest))
    return manifest
