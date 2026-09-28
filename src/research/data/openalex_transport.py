"""Anonymous, bounded transport for the diagnostic OpenAlex endpoint.

Each destination is a new directory containing every attempt body and a final
transport record. Redirects are never followed, including same-host redirects.
"""

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
from pathlib import Path
import time
from urllib.parse import parse_qs, urlsplit

import requests

from .artifacts import atomic_write_new
from .manifest import digest, json_bytes


ENDPOINT = 'https://api.openalex.org/funder-search'
QUERIES = ('java', 'Python', 'R', 'ImageJ', 'scikit-learn', 'NumPy',
           'GROMACS', 'MATLAB', 'SPSS', 'BLAST')
POLICY_BOUNDS = {'timeout_seconds': (1, 35), 'max_retries': (0, 2),
                 'max_bytes': (1, 16777216)}


def validate_policy(policy: dict) -> dict:
    if not isinstance(policy, dict) or set(policy) != set(POLICY_BOUNDS):
        raise ValueError('invalid transport policy keys')
    for key, (minimum, maximum) in POLICY_BOUNDS.items():
        if type(policy[key]) is not int or not minimum <= policy[key] <= maximum:
            raise ValueError(f'invalid transport policy: {key}')
    return dict(policy)


def _validate_url(url: str) -> None:
    parts = urlsplit(url)
    if (parts.scheme != 'https' or parts.netloc != 'api.openalex.org'
            or parts.path != '/funder-search' or parts.fragment):
        raise ValueError('disallowed OpenAlex endpoint')
    params = parse_qs(parts.query, keep_blank_values=True, strict_parsing=True)
    if set(params) != {'search', 'page', 'per_page'} or any(len(v) != 1 for v in params.values()):
        raise ValueError('disallowed OpenAlex request parameters')
    if (params['search'][0] not in QUERIES or params['page'] != ['1']
            or params['per_page'][0] not in {'1', '2', '3', '4', '5'}):
        raise ValueError('disallowed OpenAlex request bounds')


def _retry_after(value: str | None, current: datetime) -> float | None:
    if value is None:
        return None
    decimal = value.strip()
    if decimal.isascii() and decimal.isdecimal():
        significant = decimal.lstrip('0') or '0'
        # 61 is a capped over-budget marker; the original header stays in evidence.
        if len(significant) > 2 or (len(significant) == 2 and significant > '60'):
            return 61
        return int(significant)
    try:
        date = parsedate_to_datetime(value)
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        return max(0, (date - current).total_seconds())
    except (ValueError, TypeError, OverflowError):
        return None


def _quota_exhausted(status: int, body: bytes) -> bool:
    if status == 429:
        return True
    if status < 400:
        return False
    text = body.decode('utf-8', errors='replace').lower()
    return (('quota' in text or 'rate limit' in text or 'credit' in text)
            and any(word in text for word in ('exhaust', 'exceed', 'insufficient', 'limit reached')))


def fetch_snippet_response(url: str, destination: Path, *, policy: dict,
                           session=None, sleep=time.sleep, now=None) -> dict:
    """Fetch once plus at most two retries; preserve complete/partial raw bodies.

``now`` is an optional callable returning an aware datetime. Injected sessions
are made anonymous but remain caller-owned; internally created sessions close.
Filesystem publication errors propagate and never create a completed record.
"""
    _validate_url(url)
    policy = validate_policy(policy)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    now = now or (lambda: datetime.now(timezone.utc))
    owned_session = session is None
    session = requests.Session() if owned_session else session
    session.trust_env = False
    session.auth = None
    session.cert = None
    session.params.clear()
    session.headers.clear()
    session.proxies.clear()
    session.cookies.clear()
    record = {'url': url, 'status': 'failed', 'reason': None,
              'stop_collection': False, 'attempts': [], 'raw_path': None,
              'response_sha256': None, 'retrieved_at': None}
    try:
        for number in range(1, policy['max_retries'] + 2):
            attempt = {'attempt': number, 'http_status': None, 'headers': {},
                       'retrieved_at': now().isoformat(), 'raw_path': None,
                       'response_sha256': None, 'body_complete': False,
                       'error_type': None, 'error': None, 'retry_after_seconds': None,
                       'retry_wait_seconds': None}
            response = None
            body = bytearray()
            transient = False
            reason = None
            try:
                # A response may set cookies; no cookie is sent on a later attempt.
                session.cookies.clear()
                response = session.get(url, timeout=policy['timeout_seconds'],
                                       allow_redirects=False, stream=True)
                attempt['http_status'] = response.status_code
                attempt['headers'] = dict(response.headers)
                for chunk in response.iter_content(chunk_size=65536):
                    room = policy['max_bytes'] - len(body)
                    body.extend(chunk[:room])
                    if len(chunk) > room:
                        reason = 'response_too_large'
                        break
                if reason is None:
                    attempt['body_complete'] = True
            except (requests.Timeout, requests.ConnectionError,
                    requests.exceptions.ChunkedEncodingError) as exc:
                attempt.update(error_type=type(exc).__name__, error=str(exc))
                transient = True
                reason = 'transient_network_error'
            except requests.RequestException as exc:
                attempt.update(error_type=type(exc).__name__, error=str(exc))
                reason = 'request_error'
            finally:
                if response is not None:
                    response.close()
            if response is not None or body:
                raw_path = f'attempt-{number:02d}.body'
                atomic_write_new(destination / raw_path, bytes(body))
                attempt.update(raw_path=raw_path, response_sha256=digest(bytes(body)),
                               received_bytes=len(body))
            record['attempts'].append(attempt)
            status = attempt['http_status']
            retry_header = next((v for k, v in attempt['headers'].items()
                                 if k.lower() == 'retry-after'), None)
            delay = _retry_after(retry_header, now())
            attempt['retry_after_seconds'] = delay
            attempt['retry_after_seconds_capped'] = (
                delay == 61 and retry_header is not None
                and retry_header.strip().isascii() and retry_header.strip().isdecimal())
            if status is not None and _quota_exhausted(status, bytes(body)):
                reason = 'quota_exhausted'
                record['stop_collection'] = True
            elif delay is not None and delay > 60:
                reason = 'retry_after_exceeds_budget'
                record['stop_collection'] = True
            elif reason is None:
                if 300 <= status < 400:
                    reason = 'redirect_rejected'
                elif 500 <= status < 600:
                    transient, reason = True, 'http_server_error'
                elif status != 200:
                    reason = 'http_error'
                else:
                    try:
                        json.loads(bytes(body))
                    except (ValueError, UnicodeError):
                        reason = 'invalid_json'
                    else:
                        record.update(status='completed', raw_path=attempt['raw_path'],
                                      response_sha256=attempt['response_sha256'],
                                      retrieved_at=attempt['retrieved_at'])
            attempt['reason'] = reason
            record['reason'] = reason
            if record['status'] == 'completed' or record['stop_collection']:
                break
            if not transient or number > policy['max_retries']:
                break
            wait = delay if delay is not None else min(2 ** (number - 1), 60)
            attempt['retry_wait_seconds'] = wait
            sleep(wait)
        atomic_write_new(destination / 'record.json', json_bytes(record))
        return record
    finally:
        if owned_session:
            session.close()
