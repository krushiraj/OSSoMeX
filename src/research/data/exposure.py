"""Verified, role-aware reservations for acquisition and new split allocation."""

from copy import deepcopy
import json
from pathlib import Path
import re

from ..contracts import validate_document
from .artifacts import atomic_write_new
from .ecosystems import ExclusionIndex
from .manifest import digest, json_bytes
from .splits import identifiers


ROLES = ('train_reserved', 'diagnostic', 'native_excluded', 'heldout_reserved', 'historical_exposed')
PURPOSE_ROLES = {'new_acquisition': ROLES, 'heldout': ROLES, 'training': ROLES[1:]}
SCHEMA = 'exposure-reservations-1'


def _verified_bytes(path: Path, sha256: str) -> bytes:
    if not isinstance(sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', sha256):
        raise ValueError('INVALID_EXPOSURE_SHA256')
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f'STALE_EXPOSURE_SOURCE: {path}') from exc
    if digest(payload) != sha256:
        raise ValueError(f'STALE_EXPOSURE_SOURCE: {path}')
    return payload


def _companion_bytes(root: Path, files: list[dict]) -> dict:
    if not isinstance(files, list):
        raise ValueError('INVALID_EXPOSURE_FILES')
    result, resolved = {}, set()
    for record in files:
        if not isinstance(record, dict) or not isinstance(record.get('path'), str):
            raise ValueError('INVALID_EXPOSURE_FILE')
        relative = Path(record['path'])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root.resolve()) or path in resolved:
            raise ValueError('UNSAFE_EXPOSURE_COMPANION')
        resolved.add(path)
        result[record['path']] = (path, _verified_bytes(path, record.get('sha256')))
    return result


def _input_bytes(spec: dict) -> tuple[dict, dict]:
    if not isinstance(spec, dict) or spec.get('role') not in ROLES:
        raise ValueError('INVALID_EXPOSURE_ROLE')
    if spec.get('kind') not in ('bundle_manifest', 'documents_jsonl', 'diagnostic_inputs'):
        raise ValueError('INVALID_EXPOSURE_KIND')
    if spec['kind'] == 'diagnostic_inputs' and spec['role'] != 'diagnostic':
        raise ValueError('INVALID_DIAGNOSTIC_EXPOSURE_ROLE')
    if not isinstance(spec.get('path'), str) or not spec['path'].strip():
        raise ValueError('INVALID_EXPOSURE_PATH')
    path = Path(spec['path']).resolve()
    payload = _verified_bytes(path, spec.get('sha256'))
    normalized = {**deepcopy(spec), 'path': str(path)}
    if spec['kind'] != 'bundle_manifest':
        return normalized, {path.name: (path, payload)}
    manifest = json.loads(payload)
    if not isinstance(manifest, dict):
        raise ValueError('INVALID_EXPOSURE_SOURCE_MANIFEST')
    companions = _companion_bytes(path.parent, manifest.get('files'))
    required = ('inputs.jsonl', 'identities.jsonl') if spec['role'] == 'diagnostic' else ('documents.jsonl',)
    if any(name not in companions for name in required):
        raise ValueError('MISSING_EXPOSURE_COMPANION')
    if spec['role'] == 'train_reserved' and manifest.get('role') != 'train':
        raise ValueError('INVALID_TRAIN_RESERVED_BUNDLE')
    return normalized, {name: companions[name] for name in required}


def _verify_config_source(config: dict, root: Path) -> None:
    if 'config_source' not in config:
        return
    source = config['config_source']
    if not isinstance(source, dict) or not isinstance(source.get('path'), str):
        raise ValueError('INVALID_EXPOSURE_CONFIG_SOURCE')
    payload = _verified_bytes(root / source['path'], source.get('sha256'))
    if source.get('bytes_utf8') != payload.decode('utf-8'):
        raise ValueError('STALE_EXPOSURE_SOURCE: config bytes_utf8')


def _jsonl(payload: bytes, path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(payload.decode('utf-8').splitlines(), 1):
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError('object required')
        except (TypeError, ValueError) as exc:
            raise ValueError(f'INVALID_EXPOSURE_ROW: {path}:{number}') from exc
        rows.append(row)
    return rows


def _source_ids(row: dict) -> dict:
    source_ids = row.get('source_ids') or {}
    if not isinstance(source_ids, dict):
        raise ValueError('INVALID_EXPOSURE_IDENTIFIERS')
    if any(not isinstance(key, str) or (value is not None and type(value) not in (str, int))
           for key, value in source_ids.items()):
        raise ValueError('INVALID_EXPOSURE_IDENTIFIERS')
    return deepcopy(source_ids)


def _source_keys(row: dict) -> set[str]:
    keys = identifiers({'source_ids': _source_ids(row)})
    metadata = row.get('metadata') or {}
    if not isinstance(metadata, dict):
        raise ValueError('INVALID_EXPOSURE_METADATA')
    for value in (row, metadata):
        keys.update(identifiers({'source_ids': _source_ids({
            'source_ids': {'openalex': value.get('work_id'), 'doi': value.get('doi')}})}))
    return keys


def _association_keys(row: dict) -> list[set[str]]:
    groups = [_source_keys(row)]
    associations = row.get('source_associations', [])
    if not isinstance(associations, list) or any(not isinstance(value, dict) for value in associations):
        raise ValueError('INVALID_EXPOSURE_ASSOCIATIONS')
    for association in associations:
        groups.extend(_association_keys(association))
    return groups


def _identity_keys(row: dict) -> set[str]:
    keys = set().union(*_association_keys(row))
    for field in ('document_id', 'work_group_id'):
        if row.get(field) is not None:
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError('INVALID_EXPOSURE_IDENTITY')
            keys.add(field + ':' + row[field])
    return keys


def _normalize(row: dict) -> dict:
    row = deepcopy(row)
    row['source_ids'] = _source_ids(row)
    metadata = row.get('metadata') or {}
    if not isinstance(metadata, dict):
        raise ValueError('INVALID_EXPOSURE_METADATA')
    for key, value in (('openalex', metadata.get('work_id')), ('doi', metadata.get('doi'))):
        if not row['source_ids'].get(key) and value:
            row['source_ids'][key] = value
    keys = _identity_keys(row)
    text = row.get('text')
    if text is not None and not isinstance(text, str):
        raise ValueError('INVALID_EXPOSURE_TEXT')
    if text and text.strip():
        row = validate_document(row)
    elif not keys:
        raise ValueError('EMPTY_EXPOSURE_IDENTITY')
    row['identity_keys'] = sorted(keys)
    return row


def _jsonl_bytes(rows: list[dict]) -> bytes:
    return ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows).encode()


def _identity_aliases(rows: list[dict]) -> dict:
    parents = {}

    def root(key):
        parents.setdefault(key, key)
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    for row in rows:
        # Only a recorded parent association links aliases; shared snippet text does not.
        for group in _association_keys(row):
            keys = sorted(group)
            for key in keys[1:]:
                parents[root(key)] = root(keys[0])
            if keys:
                root(keys[0])
    groups = {}
    for key in parents:
        groups.setdefault(root(key), set()).add(key)
    return {key: groups[root(key)] for key in parents}


def _expanded_keys(row: dict, aliases: dict) -> set[str]:
    keys = set(row['identity_keys'])
    for key in row['identity_keys']:
        keys.update(aliases.get(key, ()))
    return keys


def _issues(rows: list[dict]) -> list[dict]:
    result, identities = [], {}
    aliases = _identity_aliases(rows)
    for row in rows:
        keys = row['identity_keys']
        source_keys = [key for key in keys if not key.startswith(('document_id:', 'work_group_id:'))]
        if not source_keys:
            result.append({'code': 'MISSING_SOURCE_IDENTIFIERS', 'exposure_id': row['exposure_id']})
        unresolved = sorted({key for group in _association_keys(row)
                             if not any(key.startswith('doi:') for key in group)
                             for key in group if key.startswith('openalex:')})
        if unresolved:
            result.append({'code': 'MISSING_DOI', 'exposure_id': row['exposure_id'],
                           'openalex_ids': unresolved,
                           'limitation': 'Cross-provider identity resolution is incomplete without a known shared identifier.'})
        conflict_keys = sorted(_expanded_keys(row, aliases))
        if (row.get('text') or '').strip():
            conflict_keys.append('text:' + digest(re.sub(r'\s+', ' ', row['text']).strip().encode()))
        for key in conflict_keys:
            identities.setdefault(key, []).append(row)
    conflicts = set()
    for key, matches in identities.items():
        roles = sorted({row['role'] for row in matches})
        exposure_ids = tuple(sorted({row['exposure_id'] for row in matches}))
        if len(roles) > 1 and exposure_ids not in conflicts:
            result.append({'code': 'EXPOSURE_ROLE_CONFLICT', 'identity': key, 'roles': roles,
                           'exposure_ids': list(exposure_ids), 'decision': 'most_restrictive_by_purpose'})
            conflicts.add(exposure_ids)
    return result


def build_exposures(config: dict, output: Path) -> dict:
    """Publish a new immutable bundle; no existing output is resumed or replaced."""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if not isinstance(config, dict) or not isinstance(config.get('inputs'), list) or not config['inputs']:
        raise ValueError('MISSING_EXPOSURE_INPUTS')
    source_root = Path.cwd().resolve()
    _verify_config_source(config, source_root)
    inputs, documents, identities = [], {}, {}
    for spec in config['inputs']:
        normalized_spec, sources = _input_bytes(spec)
        inputs.append(normalized_spec)
        for path, payload in sources.values():
            source_hash = digest(payload)
            for number, source_row in enumerate(_jsonl(payload, path), 1):
                row = _normalize(source_row)
                if 'role' in row:
                    row['source_role'] = row['role']
                row['role'] = spec['role']
                text_bearing = bool((row.get('text') or '').strip())
                target = documents if text_bearing else identities
                key = (row['role'], row['document_id'], row['text_revision']) if text_bearing else (
                    row['role'], tuple(row['identity_keys']))
                source = {'path': str(path), 'sha256': source_hash, 'row': number,
                          'input_path': normalized_spec['path']}
                if key not in target:
                    row['exposure_id'] = 'exposure:' + digest(json_bytes(key))[:32]
                    row['exposure_sources'] = [source]
                    target[key] = row
                else:
                    existing = target[key]
                    existing['exposure_sources'].append(source)
                    existing['identity_keys'] = sorted(set(existing['identity_keys']) | set(row['identity_keys']))
                    if row.get('source_associations'):
                        existing.setdefault('source_associations', []).extend(row['source_associations'])
    docs, reserved = list(documents.values()), list(identities.values())
    issues = _issues(docs + reserved)
    payloads = {'config.json': json_bytes(config), 'documents.jsonl': _jsonl_bytes(docs),
                'identities.jsonl': _jsonl_bytes(reserved), 'issues.jsonl': _jsonl_bytes(issues)}
    manifest = {'schema_version': SCHEMA, 'status': 'complete', 'inputs': inputs,
                'source_root': str(source_root),
                'document_count': len(docs), 'identity_count': len(reserved), 'issue_count': len(issues),
                'heldout_freeze_complete': False,
                'limitations': ['The overall 50-paper heldout freeze is incomplete.',
                                'Known identities and text matches do not establish complete cross-provider identity resolution.'],
                'files': [{'path': name, 'sha256': digest(payload)} for name, payload in payloads.items()]}
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        atomic_write_new(output / name, payload)
    for spec in inputs:
        _input_bytes(spec)
    _verify_config_source(config, source_root)
    atomic_write_new(output / 'manifest.json', json_bytes(manifest))
    return manifest


def load_exposures(bundle: Path) -> dict:
    """Verify frozen artifacts and sources, then return rows with lazy role indexes."""
    bundle = Path(bundle).resolve()
    payload = (bundle / 'manifest.json').read_bytes()
    manifest = json.loads(payload)
    if not isinstance(manifest, dict) or manifest.get('schema_version') != SCHEMA:
        raise ValueError('INVALID_EXPOSURE_MANIFEST')
    files = _companion_bytes(bundle, manifest.get('files'))
    if set(files) != {'config.json', 'documents.jsonl', 'identities.jsonl', 'issues.jsonl'}:
        raise ValueError('MISSING_EXPOSURE_COMPANION')
    config = json.loads(files['config.json'][1])
    if not isinstance(config, dict) or not isinstance(manifest.get('inputs'), list) or not manifest['inputs']:
        raise ValueError('INVALID_EXPOSURE_CONFIG')
    source_root = Path(manifest['source_root'])
    if not source_root.is_absolute():
        raise ValueError('INVALID_EXPOSURE_SOURCE_ROOT')
    expected_inputs = [{**spec, 'path': str((source_root / spec['path']).resolve())} for spec in config.get('inputs', [])]
    if expected_inputs != manifest['inputs']:
        raise ValueError('EXPOSURE_INPUT_MANIFEST_MISMATCH')
    for spec in manifest['inputs']:
        _input_bytes(spec)
    _verify_config_source(config, source_root)
    result = {'manifest': manifest, 'config': config,
              'documents': _jsonl(files['documents.jsonl'][1], files['documents.jsonl'][0]),
              'identities': _jsonl(files['identities.jsonl'][1], files['identities.jsonl'][0]),
              'issues': _jsonl(files['issues.jsonl'][1], files['issues.jsonl'][0]),
              '_bundle': bundle, '_manifest_sha256': digest(payload), '_indexes': {}, '_identity_index': {}}
    for category in ('documents', 'identities'):
        count_key = 'document_count' if category == 'documents' else 'identity_count'
        if manifest[count_key] != len(result[category]):
            raise ValueError('EXPOSURE_COUNT_MISMATCH')
        for row in result[category]:
            if row.get('role') not in ROLES or not isinstance(row.get('exposure_id'), str):
                raise ValueError('INVALID_EXPOSURE_ROW')
            normalized = _normalize(row)
            if not set(normalized['identity_keys']).issubset(set(row['identity_keys'])):
                raise ValueError('INVALID_EXPOSURE_IDENTITIES')
            if (category == 'documents') != bool((row.get('text') or '').strip()):
                raise ValueError('INVALID_EXPOSURE_TEXT_ROLE')
    rows = result['documents'] + result['identities']
    aliases = _identity_aliases(rows)
    for row in rows:
        for key in _expanded_keys(row, aliases):
            result['_identity_index'].setdefault(key, []).append(row)
    return result


def verify_exposure_sources(exposures: dict) -> None:
    """Recheck bundle, all companions and original inputs immediately before publishing."""
    bundle = exposures['_bundle']
    payload = _verified_bytes(bundle / 'manifest.json', exposures['_manifest_sha256'])
    manifest = json.loads(payload)
    files = _companion_bytes(bundle, manifest['files'])
    for spec in manifest['inputs']:
        _input_bytes(spec)
    _verify_config_source(json.loads(files['config.json'][1]), Path(manifest['source_root']))


def exposure_reasons(document: dict, exposures: dict, *, purpose: str) -> list[dict]:
    """Return blocking matches, preserving their roles; matching does not rehash disk inputs."""
    if purpose not in PURPOSE_ROLES:
        raise ValueError('INVALID_EXPOSURE_PURPOSE')
    candidate = _normalize(document)
    roles = PURPOSE_ROLES[purpose]
    matched = {}
    for key in candidate['identity_keys']:
        for row in exposures['_identity_index'].get(key, []):
            if row['role'] in roles:
                matched[row['exposure_id']] = (row, 'identifier_overlap')
    if (candidate.get('text') or '').strip():
        for role in roles:
            if role not in exposures['_indexes']:
                rows = [row for row in exposures['documents'] if row['role'] == role]
                by_id = {row['exposure_id']: row for row in rows}
                index = ExclusionIndex([{**row, 'document_id': row['exposure_id']} for row in rows])
                exposures['_indexes'][role] = (index, by_id)
            index, by_id = exposures['_indexes'][role]
            for reason in index.reasons(candidate):
                ident = reason['document_id']
                matched.setdefault(ident, (by_id[ident], reason['reason']))
    return [{'exposure_id': ident, 'document_id': row.get('document_id'), 'role': row['role'], 'reason': reason}
            for ident, (row, reason) in sorted(matched.items())]


def apply_split_exposures(documents: list[dict], exposures: dict) -> list[dict]:
    """Copy records and set existing allocation flags without rewriting native/split fields."""
    result = deepcopy(documents)
    for row in result:
        reasons = exposure_reasons(row, exposures, purpose='heldout')
        if reasons:
            row['development_exposed'] = True
            if any(reason['role'] != 'train_reserved' for reason in reasons):
                row['historical'] = True
            row.setdefault('exposure_reasons', []).extend(reasons)
    return result
