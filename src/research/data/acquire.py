"""Bounded public acquisition. No metadata association is a negative label."""

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import io
import ipaddress
import json
from pathlib import Path, PurePosixPath
import socket
import stat
import time
from urllib.parse import parse_qsl, quote, urlencode, urljoin, urlsplit
import zipfile

import requests

from .manifest import digest, fulltext_eligible, json_bytes, read_jsonl, verified_path, write_jsonl, write_once

QUOTAS = {'sofair': 80, 'somesci': 40, 'ecosystems': 20, 'openalex': 20}


class AcquisitionError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _safe_url(url: str, resolve: bool = False) -> None:
    p = urlsplit(url)
    if (p.scheme != 'https' or not p.hostname or p.username or p.password or p.fragment
            or p.port not in (None, 443)
            or any(any(secret in k.lower() for secret in ('token', 'key', 'secret', 'password'))
                   for k, _ in parse_qsl(p.query))):
        raise AcquisitionError('unsafe_url')
    if p.hostname == 'localhost' or p.hostname.endswith(('.localhost', '.local')):
        raise AcquisitionError('unsafe_url')
    try:
        addresses = [ipaddress.ip_address(p.hostname)]
    except ValueError:
        addresses = [ipaddress.ip_address(a[4][0]) for a in socket.getaddrinfo(p.hostname, 443)] if resolve else []
    if any(not address.is_global for address in addresses):
        raise AcquisitionError('unsafe_url')


def component_url(base: str, identifier: str, suffix: str = '') -> str:
    return base.rstrip('/') + '/' + quote(identifier, safe='') + ('/' + suffix if suffix else '')


def _check_payload(data: bytes, policy: dict) -> None:
    if policy.get('format') == 'json':
        try:
            json.loads(data)
        except (ValueError, UnicodeError) as exc:
            raise AcquisitionError('invalid_json') from exc
    if policy.get('format') == 'zip':
        zip_members(data, skip_symlinks=policy.get('skip_symlinks', False))
    if policy.get('sha256') and digest(data) != policy['sha256']:
        raise AcquisitionError('checksum_mismatch')
    if policy.get('md5') and hashlib.md5(data).hexdigest() != policy['md5']:
        raise AcquisitionError('checksum_mismatch')


def fetch_public(url: str, destination: Path, policy: dict) -> dict:
    _safe_url(url)
    sidecar = destination.with_name(destination.name + '.json')
    if destination.exists() or sidecar.exists():
        if not destination.exists() or not sidecar.exists():
            raise AcquisitionError('incomplete_cache')
        record = json.loads(sidecar.read_bytes())
        if record['url'] != url:
            raise AcquisitionError('cache_conflict')
        data = destination.read_bytes()
        if digest(data) != record['sha256']:
            raise AcquisitionError('checksum_mismatch')
        _check_payload(data, policy)
        return record
    session = policy.get('session') or requests.Session()
    sleep = policy.get('sleep', time.sleep)
    max_bytes = min(policy.get('max_bytes', 64 * 1024 * 1024), 128 * 1024 * 1024)
    events = []
    last = 'network_failed'
    for attempt in range(1, 6):
        retry_after = None
        response = None
        try:
            current = url
            for redirect in range(6):
                _safe_url(current, resolve='session' not in policy)
                response = session.get(current, timeout=(10, 60), allow_redirects=False, stream=True,
                                       headers={'User-Agent': 'scibert-research/2.0'})
                if response.status_code not in (301, 302, 303, 307, 308):
                    break
                target = urljoin(current, response.headers.get('Location', ''))
                if hasattr(response, 'close'):
                    response.close()
                _safe_url(target)
                current = target
            else:
                raise AcquisitionError('redirect_limit')
            status = response.status_code
            events.append({'attempt': attempt, 'status': status})
            if status == 200:
                if hasattr(response, 'iter_content'):
                    chunks, size = [], 0
                    for chunk in response.iter_content(65536):
                        size += len(chunk)
                        if size > max_bytes:
                            raise AcquisitionError('response_too_large')
                        chunks.append(chunk)
                    data = b''.join(chunks)
                else:
                    data = response.content
                if len(data) > max_bytes:
                    raise AcquisitionError('response_too_large')
                _check_payload(data, policy)
                record = {'url': url, 'resolved_url': current, 'status': 'success', 'http_status': status,
                          'sha256': digest(data), 'bytes': len(data), 'attempts': attempt, 'events': events,
                          'content_type': response.headers.get('Content-Type'),
                          'retrieved_at_utc': datetime.now(timezone.utc).isoformat()}
                write_once(destination, data)
                write_once(sidecar, json_bytes(record))
                return record
            if status not in (429, 500, 502, 503, 504):
                raise AcquisitionError(f'http_{status}')
            last = 'rate_limited' if status == 429 else 'server_failed'
            retry_after = response.headers.get('Retry-After')
        except (requests.RequestException, OSError):
            events.append({'attempt': attempt, 'status': 'network_failed'})
            last = 'network_failed'
        finally:
            if response is not None and hasattr(response, 'close'):
                response.close()
        if attempt < 5:
            delay = 2 ** (attempt - 1)
            if retry_after:
                try:
                    after = float(retry_after)
                except ValueError:
                    try:
                        after = (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds()
                    except (ValueError, TypeError):
                        after = 0
                if after > 60:
                    raise AcquisitionError('retry_after_exceeds_wait_budget')
                delay = max(delay, after)
            sleep(min(delay, 60))
    raise AcquisitionError(last)


def zip_members(data: bytes, skip_symlinks: bool = False) -> dict[str, bytes]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            total, result = 0, {}
            for entry in archive.infolist():
                name = entry.filename
                path = PurePosixPath(name)
                mode = entry.external_attr >> 16
                if skip_symlinks and stat.S_ISLNK(mode):
                    continue
                total += entry.file_size
                if (path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name
                        or stat.S_ISLNK(mode) or name in result or total > 256 * 1024 * 1024
                        or len(archive.infolist()) > 20000):
                    raise AcquisitionError('unsafe_archive')
                if not entry.is_dir():
                    result[name] = archive.read(entry)
            return result
    except (zipfile.BadZipFile, RuntimeError) as exc:
        raise AcquisitionError('unsafe_archive') from exc


def collect_candidates(source_config: dict, cache_dir: Path) -> list[dict]:
    config = source_config
    rows, pages, seen = [], [], set()
    cap = min(config.get('max_pages', 20), 20)
    size = min(config.get('page_size', 100), 100)
    if cap < 1 or size < 1:
        raise AcquisitionError('invalid_pagination')
    for page in range(1, cap + 1):
        query = {**config.get('query', {}), 'page': page, config.get('page_size_key', 'per_page'): size}
        url = config['url'] + '?' + urlencode(query)
        dest = cache_dir / 'responses' / digest(url.encode())
        record = fetch_public(url, dest, {**config.get('policy', {}), 'format': 'json'})
        body = json.loads(dest.read_bytes())
        records = body.get(config['results_key']) if config.get('results_key') and isinstance(body, dict) else body
        if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
            raise AcquisitionError('invalid_page_shape')
        pages.append({**record, 'page': page, 'requested_count': size, 'returned_count': len(records)})
        for item in records:
            ident = item.get(config.get('id_field', 'id'))
            if ident is None or not str(ident).strip():
                raise AcquisitionError('missing_identifier')
            key = str(ident).lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append({'source': config['name'], 'source_record_id': str(ident), 'status': 'success',
                         'role': 'weak_candidate', 'fulltext_eligible': False, 'metadata': item,
                         'metadata_license': config.get('metadata_license'),
                         'path': str(dest.relative_to(cache_dir)), 'sha256': record['sha256'],
                         'query_url': url, 'retrieved_at_utc': record['retrieved_at_utc']})
        if len(records) < size or len(rows) >= config.get('target_candidates', 3 * QUOTAS[config['name']]):
            break
    write_jsonl(cache_dir / 'pages.jsonl', pages)
    return rows


def acquire_text(source: dict, cache_dir: Path) -> dict:
    if not fulltext_eligible(source):
        raise AcquisitionError('text_access_unverified')
    format_name = source.get('format')
    if format_name not in ('text', 'tei'):
        raise AcquisitionError('unsupported_text_format')
    dest = cache_dir / 'texts' / digest(source['url'].encode())
    record = fetch_public(source['url'], dest, source.get('policy', {}))
    raw = dest.read_bytes()
    if format_name == 'text':
        if b'<html' in raw[:1000].lower() or b'<!doctype html' in raw[:1000].lower():
            raise AcquisitionError('unexpected_html')
        text = raw.decode('utf-8')
    else:
        from .tei import read_tei
        text = read_tei(raw)['text']
    if not text.strip():
        raise AcquisitionError('empty_text')
    return {**{k:v for k,v in source.items() if k != 'policy'}, **record, 'role':'paper_text',
            'fulltext_eligible':True, 'extractor_version':'plain-utf8-v1' if format_name == 'text' else 'tei-v2',
            'path':str(dest.relative_to(cache_dir))}


def acquire_sources(config_path: Path, output: Path) -> dict:
    config = json.loads(config_path.read_bytes())
    write_once(output / 'config.json', json_bytes(config))
    all_rows, failures = [], []
    for source in config['sources']:
        root = output / source['name']
        try:
            if source['kind'] == 'metadata':
                rows = collect_candidates(source, root)
            elif source['kind'] == 'file':
                path = root / 'metadata.json'
                record = fetch_public(source['url'], path, {'format':'json'})
                rows = [{**record, **source, 'source':source['name'], 'path':'metadata.json', 'role':'source_metadata'}]
            elif source['kind'] == 'paper_text':
                rows = [acquire_text(source, root)]
            else:
                path = root / 'archive.zip'
                if source['kind'] == 'local_archive':
                    data = Path(source['path']).read_bytes()
                    _check_payload(data, source)
                    write_once(path, data)
                    record = {'sha256': digest(data), 'url': source['url'], 'status': 'success'}
                else:
                    record = fetch_public(source['url'], path, {**source, 'format': 'zip'})
                rows = [{**record, **{k: v for k, v in source.items() if k != 'path'},
                         'source': source['name'], 'path': 'archive.zip', 'role': 'native_archive',
                         'fulltext_eligible': False}]
            write_jsonl(root / 'candidates.jsonl', rows)
            all_rows.extend({**r, 'path': str(Path(source['name']) / r['path'])} for r in rows)
        except (AcquisitionError, OSError, ValueError) as exc:
            failure = {'source': source['name'], 'status': 'failed',
                       'code': exc.code if isinstance(exc, AcquisitionError) else type(exc).__name__}
            failures.append(failure)
    # Failures are separate attempts, so a resumed successful acquisition can complete the immutable ledger.
    if failures:
        attempt = digest(json_bytes(failures))
        write_jsonl(output / 'failures' / f'{attempt}.jsonl', failures)
        write_jsonl(output / 'partial' / f'{digest(json_bytes(all_rows))}.jsonl', all_rows)
        return {'status': 'blocked', 'failures': failures, 'candidate_records': len(all_rows)}
    write_jsonl(output / 'acquisitions.jsonl', all_rows)
    return {'status': 'success', 'candidate_records': len(all_rows)}


def audit_sources(source_manifest: Path, output_dir: Path) -> dict:
    rows = read_jsonl(source_manifest)
    sources = {name: {'required': count, 'records': 0, 'eligible_fulltext': 0} for name, count in QUOTAS.items()}
    for row in rows:
        source = sources.setdefault(row['source'], {'required': 0, 'records': 0, 'eligible_fulltext': 0})
        source['records'] += 1
        if 'path' in row:
            verified_path(source_manifest.parent, row)
        source['eligible_fulltext'] += int(fulltext_eligible(row))
        if row.get('role') == 'native_archive':
            members = zip_members(verified_path(source_manifest.parent, row).read_bytes(), row.get('skip_symlinks', False))
            source['archive_members'] = len(members)
            source['tei_documents'] = sum(n.endswith('.tei.xml') for n in members)
            source['brat_documents'] = sum(n.endswith('.txt') and n[:-4]+'.ann' in members for n in members)
            source['native_split_files'] = [n for n in members if n.endswith('train_devel_test_split.json')]
            source['note'] = 'Archive availability is not verified full-text/license/native-split eligibility. Build first.'
    for source in sources.values():
        source['shortage'] = max(0, source['required'] - source['eligible_fulltext'])
    report = {'status': 'blocked' if any(s['shortage'] for s in sources.values()) else 'ready',
              'sources': sources, 'source_manifest_sha256': digest(source_manifest.read_bytes())}
    write_once(output_dir / 'audit.json', json_bytes(report))
    return report
