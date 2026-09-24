"""Validated split bundles with held-out text omitted from train/dev."""

from pathlib import Path

from ..contracts import validate_document
from .manifest import digest, json_bytes, read_jsonl, verified_path, write_jsonl, write_once
from .splits import check_training_manifest


def materialize_bundle(manifest: dict, destination: Path, *, resume=False) -> dict:
    if destination.exists() and not resume:
        raise FileExistsError(destination)
    if destination.exists() and any(p.name not in {'documents.jsonl','manifest.json'} for p in destination.iterdir()):
        raise ValueError('UNEXPECTED_BUNDLE_ARTIFACT')
    role = manifest['role']
    if role not in ('train', 'dev', 'test'):
        raise ValueError('UNKNOWN_SPLIT')
    docs = [validate_document(d) for d in manifest['documents'] if d.get('split') == role]
    if not docs:
        raise ValueError('EMPTY_SPLIT')
    if role == 'train':
        issues = check_training_manifest({'documents': docs}, manifest.get('heldout', {}))
        if issues:
            raise ValueError(issues)
    if len({d['document_id'] for d in docs}) != len(docs):
        raise ValueError('DUPLICATE_DOCUMENT_ID')
    destination.mkdir(parents=True,exist_ok=resume)
    write_jsonl(destination / 'documents.jsonl', docs)
    result = {'schema_version':'2.0', 'role':role, 'document_count':len(docs),
              'split_digest':manifest.get('split_digest'), 'heldout':manifest.get('heldout', {}),
              'files':[{'path':'documents.jsonl', 'sha256':digest((destination/'documents.jsonl').read_bytes())}]}
    write_once(destination / 'manifest.json', json_bytes(result))
    return result


def load_bundle(bundle: Path, roles=('train', 'dev')) -> tuple[dict, list[dict]]:
    import json
    manifest = json.loads((bundle/'manifest.json').read_bytes())
    if manifest['role'] not in roles:
        raise ValueError('SPLIT_NOT_PERMITTED')
    for record in manifest['files']:
        verified_path(bundle, record)
    docs = [validate_document(d) for d in read_jsonl(bundle/'documents.jsonl')]
    if manifest['role'] == 'train' and check_training_manifest({'documents':docs}, manifest['heldout']):
        raise ValueError('HELDOUT_TRAINING_DATA')
    return manifest, docs
