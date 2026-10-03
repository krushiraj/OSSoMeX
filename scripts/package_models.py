"""Package only registry-pinned pipeline manifests and their verified stage files."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def collect(root, registry, selected):
    checkpoint_root = (root / 'checkpoints').resolve()
    files = {}
    known = {row['id']: row for row in registry['models']}
    if not selected or set(selected) - known.keys():
        raise ValueError('select one or more model IDs from models/registry.json')

    def add(path, expected=None):
        path = path.resolve()
        if not path.is_relative_to(checkpoint_root) or not path.is_file():
            raise ValueError(f'missing or unsafe checkpoint file: {path.name}')
        checksum = sha256(path)
        if expected and checksum != expected:
            raise ValueError(f'checkpoint hash mismatch: {path.name}')
        name = 'checkpoints/' + path.relative_to(checkpoint_root).as_posix()
        if name in files and files[name]['sha256'] != checksum:
            raise ValueError(f'conflicting checkpoint file: {name}')
        files[name] = {'source': path, 'sha256': checksum, 'bytes': path.stat().st_size}

    for identifier in selected:
        record = known[identifier]
        pipeline_dir = checkpoint_root / identifier
        pipeline_path = pipeline_dir / 'manifest.json'
        add(pipeline_path, record['manifest_sha256'])
        pipeline = json.loads(pipeline_path.read_bytes())
        if pipeline.get('schema_version') != 'full-label-checkpoint-1' or not pipeline.get('pipeline_complete'):
            raise ValueError(f'incomplete pipeline: {identifier}')
        for stage, entry in pipeline['stages'].items():
            if entry.get('status') != 'available':
                raise ValueError(f'unavailable stage: {stage}')
            if {k: entry[k] for k in ('path', 'manifest_sha256')} != record['stages'][stage]:
                raise ValueError(f'pipeline differs from registry: {stage}')
            directory = (pipeline_dir / entry['path']).resolve()
            manifest_path = directory / 'manifest.json'
            add(manifest_path, entry['manifest_sha256'])
            manifest = json.loads(manifest_path.read_bytes())
            for item in manifest['files']:
                relative = Path(item['path'])
                path = (directory / relative).resolve()
                if relative.is_absolute() or not path.is_relative_to(directory):
                    raise ValueError(f'unsafe stage file: {relative}')
                add(path, item['sha256'])
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', action='append', required=True, help='Repeat for multiple registered pipelines')
    parser.add_argument('--output', required=True, type=Path, help='New .tar path; never overwrite a release asset')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or Path(str(output) + '.sha256').exists():
        raise FileExistsError(output)
    registry = json.loads((ROOT / 'models/registry.json').read_bytes())
    files = collect(ROOT, registry, args.model)
    metadata = {'schema_version': 'ossomex-model-bundle-1', 'release': registry['release'],
                'models': args.model, 'public_distribution': 'pending',
                'files': {name: {k: row[k] for k in ('sha256', 'bytes')} for name, row in files.items()}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream, tarfile.open(fileobj=stream, mode='w') as archive:
        for name, row in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = row['bytes']
            info.mode = 0o644
            with row['source'].open('rb') as source:
                archive.addfile(info, source)
        notes = {'MODEL_BUNDLE.json': (json.dumps(metadata, indent=2) + '\n').encode()}
        for name in ('LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md', 'docs/models/README.md'):
            notes['bundle-notices/' + Path(name).name] = (ROOT / name).read_bytes()
        for name, payload in notes.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(payload), 0o644
            archive.addfile(info, io.BytesIO(payload))
    # Check the actual archived bytes, not only the source files before packaging.
    with tarfile.open(output, 'r') as archive:
        for name, record in metadata['files'].items():
            with archive.extractfile(name) as source:
                if hashlib.file_digest(source, 'sha256').hexdigest() != record['sha256']:
                    raise ValueError(f'archive verification failed: {name}')
    checksum = sha256(output)
    Path(str(output) + '.sha256').write_text(f'{checksum}  {output.name}\n')
    print(json.dumps({'output': str(output), 'sha256': checksum, 'bytes': output.stat().st_size,
                      'models': args.model, 'verified_model_files': len(files)}, indent=2))


if __name__ == '__main__':
    main()
