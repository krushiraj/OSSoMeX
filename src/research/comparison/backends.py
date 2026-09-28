"""Local comparison backend protocol, configuration and transport boundaries."""

from copy import deepcopy
import base64
import hashlib
import ipaddress
import json
import math
from pathlib import Path
import socket
from urllib.parse import urlsplit, urlunsplit

import requests

from ..contracts import text_revision
from .contracts import CAPABILITY_FIELDS


REPO_ROOT = Path(__file__).resolve().parents[3]


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def capabilities(enabled=True):
    return {key: enabled and key in ('software_spans', 'version_spans') for key in CAPABILITY_FIELDS}


def resolve_arm(arm: dict, *, repo_root=None) -> dict:
    """Freeze exact config bytes; revalidate both bytes and parsed config on reuse."""
    if not isinstance(arm, dict):
        raise ValueError('arm must be an object')
    result = deepcopy(arm)
    if 'config' in result and 'config_path' in result:
        raise ValueError('cannot supply both config and config_path')
    root = Path(repo_root or REPO_ROOT).resolve()
    source = result.get('config_source')
    if source is not None:
        if 'config_path' in result or not isinstance(source, dict):
            raise ValueError('invalid resolved config_source')
        data = source['bytes_utf8'].encode('utf-8')
        if sha256(data) != source['sha256']:
            raise ValueError('config_source hash mismatch')
        if source['path'] is not None and Path(source['path']).read_bytes() != data:
            raise ValueError('config source changed since resolution')
        if json_text(json.loads(data)) != json_text(result.get('config')):
            raise ValueError('resolved config differs from frozen bytes')
    elif 'config_path' in result:
        path = Path(result.pop('config_path'))
        path = (root / path).resolve() if not path.is_absolute() else path.resolve()
        data = path.read_bytes()
        result['config'] = json.loads(data)
        source = {'path': str(path), 'sha256': sha256(data), 'bytes_utf8': data.decode('utf-8')}
    else:
        result.setdefault('config', {})
        data = json_text(result['config']).encode('utf-8')
        source = {'path': None, 'sha256': sha256(data), 'bytes_utf8': data.decode('utf-8')}
    if not isinstance(result['config'], dict):
        raise ValueError('config must be an object')
    result.setdefault('request_options', {})
    if not isinstance(result['request_options'], dict):
        raise ValueError('request_options must be an object')
    json_text(result['request_options'])
    result['config_source'] = source
    return result


def local_url(url: str) -> str:
    """Reject non-loopback endpoints and pin localhost to a validated literal."""
    if not isinstance(url, str) or any(c.isspace() or ord(c) < 32 for c in url):
        raise ValueError('invalid local endpoint')
    parsed = urlsplit(url)
    if (parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or parsed.fragment):
        raise ValueError('endpoint requires local HTTP without credentials or fragment')
    host, port = parsed.hostname, parsed.port
    if host == 'localhost':
        addresses = socket.getaddrinfo(host, port or (443 if parsed.scheme == 'https' else 80),
                                       type=socket.SOCK_STREAM)
        ips = [ipaddress.ip_address(row[4][0]) for row in addresses]
        if not ips or any(not ip.is_loopback for ip in ips):
            raise ValueError('localhost must resolve only to loopback')
        host = str(min(ips, key=lambda ip: ip.version))
    else:
        try:
            address = ipaddress.ip_address(host)
        except ValueError as exc:
            raise ValueError('endpoint host must be literal loopback or localhost') from exc
        if not address.is_loopback:
            raise ValueError('endpoint must be loopback')
    authority = f'[{host}]' if ':' in host else host
    if port is not None:
        authority += f':{port}'
    return urlunsplit((parsed.scheme, authority, parsed.path, parsed.query, ''))


def verify_evidence(record: dict) -> dict:
    """Read explicitly pinned local provenance; no remote metadata acquisition."""
    if not isinstance(record, dict) or not isinstance(record.get('sha256'), str):
        raise ValueError('local evidence requires path and sha256')
    path = (REPO_ROOT / record['path']).resolve()
    data = path.read_bytes()
    if sha256(data) != record['sha256']:
        raise ValueError('local evidence hash mismatch')
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError('local evidence must be an object')
    return value


def validate_window(window):
    if (not isinstance(window, dict) or not isinstance(window.get('window_id'), str)
            or not window['window_id'] or not isinstance(window.get('text'), str)
            or window.get('window_text_revision') != text_revision(window['text'])):
        raise ValueError('invalid frozen window text/revision')


def score_fields(score, kind):
    if score is None:
        return {'score': None, 'score_kind': None}
    if type(score) not in (int, float) or not math.isfinite(score):
        raise ValueError('invalid native extraction score')
    return {'score': score, 'score_kind': kind}


def window_result(window, *, spans=None, unresolved=None, raw=None, error=None, status=None):
    spans, unresolved = spans or [], unresolved or []
    if error is not None and not str(error).strip():
        error = 'backend_failure_without_detail'
    status = status or ('failure' if error is not None or unresolved else ('success' if spans else 'no_mentions'))
    return {'window_id': window['window_id'], 'status': status,
            'spans': spans if status in ('success', 'no_mentions') else [],
            'unresolved': unresolved, 'raw': raw if raw is not None else {}, 'error': error}


class ComparisonBackend:
    def __init__(self):
        self.arm = None
        self.loaded = None

    def _prepare(self, arm):
        self.arm = resolve_arm(arm)
        return {'config_sha256': self.arm['config_source']['sha256'],
                'request_options_sha256': sha256(json_text(self.arm['request_options']).encode('utf-8')),
                'request_options': deepcopy(self.arm['request_options'])}

    def _loaded(self, status, reason, identity=None, *, supported=True):
        if status != 'ready' and (reason is None or not str(reason).strip()):
            reason = 'backend_unavailable_without_detail'
        self.loaded = {'status': status, 'reason': reason, 'capabilities': capabilities(supported),
                       'identity': identity or {}}
        return deepcopy(self.loaded)

    def _require_ready(self, window):
        validate_window(window)
        if self.loaded is None or self.loaded['status'] != 'ready':
            raise ValueError('backend is not ready')

    def load(self, config: dict) -> dict:
        raise NotImplementedError

    def predict(self, window: dict) -> dict:
        raise NotImplementedError

    def close(self) -> None:
        self.loaded = None


class InactiveBackend(ComparisonBackend):
    def __init__(self, status, reason, *, disabled=False):
        super().__init__()
        self.status, self.reason = status, reason
        self.disabled = disabled

    def load(self, config):
        identity = {'backend': config.get('backend'), 'disabled': self.disabled, 'config_verified': False}
        try:
            identity.update(self._prepare(config))
            identity.update({'config': deepcopy(self.arm['config']), 'config_verified': True})
        except Exception as exc:
            # Base remains unsupported/no_task_head even when provenance is incomplete.
            identity['config_error'] = str(exc) or type(exc).__name__
        return self._loaded(self.status, self.reason, identity, supported=False)

    def predict(self, window):
        return window_result(window, status=self.status, error=self.reason)


class HTTPBackend(ComparisonBackend):
    def __init__(self, transport=None):
        super().__init__()
        self.transport = transport if transport is not None else requests.Session()
        self.transport.trust_env = False

    def _request(self, method, url, raw, **kwargs):
        url = local_url(url)
        raw.update({'method': method, 'url': url})
        try:
            response = self.transport.request(method, url, allow_redirects=False, **kwargs)
        except requests.RequestException as exc:
            raw.update({'exception': type(exc).__name__, 'detail': str(exc)})
            raise
        content = response.content
        encoding = response.encoding or response.apparent_encoding
        body = response.text
        raw.update({'status_code': response.status_code, 'headers': dict(response.headers),
                    'body': body, 'body_encoding': encoding,
                    'requests_body': body, 'requests_encoding': encoding,
                    'body_bytes_base64': base64.b64encode(content).decode('ascii'),
                    'body_sha256': sha256(content)})
        return response

    def close(self):
        self.transport.close()
        super().close()


def make_backend(arm: dict) -> ComparisonBackend:
    backend = arm.get('backend')
    if backend == 'base' or arm.get('arm_id') == 'scibert-base':
        return InactiveBackend('unsupported', 'no_task_head')
    if arm.get('disabled_reason') or arm.get('arm_id') == 'scibert-frozen-encoder-probe':
        return InactiveBackend('unsupported', arm.get('disabled_reason', 'approval_required'), disabled=True)
    if arm.get('unsupported_reason'):
        return InactiveBackend('unsupported', arm['unsupported_reason'])
    if arm.get('unavailable_reason'):
        return InactiveBackend('unavailable', arm['unavailable_reason'])
    if backend == 'modernbert':
        return InactiveBackend('unsupported', 'no_trained_compatible_checkpoint')
    if backend == 'scibert':
        from .backend_scibert import SciBertBackend
        return SciBertBackend()
    if backend == 'softcite':
        from .backend_softcite import SoftciteBackend
        return SoftciteBackend()
    if backend == 'ollama':
        from .backend_ollama import OllamaBackend
        return OllamaBackend()
    return InactiveBackend('unsupported', 'no_compatible_backend')


def preflight_arms(arms: list[dict]) -> list[dict]:
    results = []
    for arm in arms:
        backend = make_backend(arm)
        try:
            results.append({'arm_id': arm['arm_id'], **backend.load(arm)})
        finally:
            backend.close()
    return results
