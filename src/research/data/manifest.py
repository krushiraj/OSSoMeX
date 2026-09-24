"""Immutable JSON artifacts and acquisition provenance checks."""

import hashlib
import json
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()


def write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f'artifact conflict: {path}')
        return
    with path.open('xb') as stream:
        stream.write(payload)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    write_once(path, ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows).encode())


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError('object required')
            rows.append(row)
        except (ValueError, TypeError) as exc:
            raise ValueError(f'{path}:{number}: invalid JSON object') from exc
    return rows


def verified_path(root: Path, record: dict) -> Path:
    relative = Path(record['path'])
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError('unsafe artifact path')
    if not path.is_file() or digest(path.read_bytes()) != record['sha256']:
        raise ValueError(f'missing or changed artifact: {relative}')
    return path


def fulltext_eligible(record: dict) -> bool:
    return (record.get('public') is True and record.get('supplied_text_scope') == 'fulltext'
            and record.get('language') == 'en' and bool(record.get('access_basis'))
            and record.get('text_license') in {'CC-BY-4.0', 'CC-BY-3.0', 'CC-BY-2.0',
                                              'CC-BY-SA-4.0', 'CC0-1.0', 'public-domain'})
