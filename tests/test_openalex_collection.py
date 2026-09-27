import importlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest
import requests

from research.data.manifest import digest, read_jsonl


URL = 'https://api.openalex.org/funder-search?search=NumPy&page=1&per_page=5'
POLICY = {'timeout_seconds': 35, 'max_retries': 2, 'max_bytes': 16777216}
CONFIG = {
    'endpoint': 'https://api.openalex.org/funder-search',
    'queries': ['java', 'Python', 'R', 'ImageJ', 'scikit-learn', 'NumPy',
                'GROMACS', 'MATLAB', 'SPSS', 'BLAST'],
    'page': 1, 'per_page': 5, 'snippets_per_work': 2, 'max_candidates': 100,
    **POLICY, 'max_concurrency': 1,
}


def work(number=1, snippets=None):
    return {'id': f'https://openalex.org/W{number}', 'doi': None,
            'relevance_score': 1.0, 'snippets': ['NumPy'] if snippets is None else snippets}


class Response:
    def __init__(self, status=200, payload=None, body=None, headers=None, interruption=None):
        self.status_code = status
        self.headers = headers or {}
        self.body = body if body is not None else json.dumps(
            {'results': [work()]} if payload is None else payload).encode()
        self.interruption = interruption
        self.closed = False

    def iter_content(self, chunk_size):
        yield self.body
        if self.interruption:
            raise self.interruption

    def close(self):
        self.closed = True


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.trust_env = True
        self.auth = ('must', 'clear')
        self.headers = {'Authorization': 'must-clear'}
        self.proxies = {'https': 'must-clear'}
        self.params = {'api_key': 'must-clear'}
        self.cert = 'must-clear-client-certificate'
        self.cookies = requests.cookies.RequestsCookieJar()
        self.closed = False

    def get(self, url, **kwargs):
        assert self.trust_env is False
        assert self.auth is None
        assert not self.headers and not self.proxies and not self.cookies
        assert not self.params and self.cert is None
        assert kwargs['allow_redirects'] is False
        assert kwargs['stream'] is True
        assert kwargs['timeout'] <= 35
        self.calls.append((url, kwargs))
        item = next(self.responses)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        self.closed = True


def transport():
    return importlib.import_module('research.data.openalex_transport')


def collector():
    module = importlib.import_module('research.data.openalex_snippets')
    assert hasattr(module, 'collect_snippets'), 'bounded collector is absent'
    return module


@pytest.mark.parametrize('events,expected,statuses', [
    ([requests.Timeout('timeout'), Response()], 'completed', [None, 200]),
    ([Response(500), Response(503), Response()], 'completed', [500, 503, 200]),
    ([Response(500), Response(503), Response(502)], 'failed', [500, 503, 502]),
    ([Response(400)], 'failed', [400]),
    ([Response(body=b'{bad')], 'failed', [200]),
    ([Response(429)], 'failed', [429]),
])
def test_retry_budget_and_quota_stop(tmp_path, events, expected, statuses):
    session = Session(events)
    record = transport().fetch_snippet_response(URL, tmp_path / 'raw', policy=POLICY,
                                               session=session, sleep=lambda _: None)
    assert record['status'] == expected
    assert [row['http_status'] for row in record['attempts']] == statuses
    assert len(record['attempts']) <= 3
    assert all(call[1]['timeout'] == 35 for call in session.calls)
    assert all(call[0] == URL for call in session.calls)
    assert record['stop_collection'] is (statuses == [429])
    for attempt in record['attempts']:
        if attempt['raw_path']:
            assert digest((tmp_path / 'raw' / attempt['raw_path']).read_bytes()) == attempt['response_sha256']


@pytest.mark.parametrize('header,seconds,stop', [
    ('12', 12, False), ('60', 60, False), ('61', None, True),
    ('Sun, 27 Sep 2026 00:00:20 GMT', 20, False),
    ('Sun, 27 Sep 2026 00:01:01 GMT', None, True),
])
def test_retry_after_waits_are_bounded(tmp_path, header, seconds, stop):
    waits = []
    session = Session([Response(503, headers={'Retry-After': header}), Response()])
    result = transport().fetch_snippet_response(
        URL, tmp_path / 'raw', policy=POLICY, session=session, sleep=waits.append,
        now=lambda: datetime(2026, 9, 27, tzinfo=timezone.utc))
    assert result['stop_collection'] is stop
    assert waits == ([] if stop else [seconds])
    assert len(session.calls) == (1 if stop else 2)


@pytest.mark.parametrize('url', [
    'http://api.openalex.org/funder-search?search=NumPy&page=1&per_page=5',
    URL + '&api_key=secret', URL + '&mailto=person@example.org', URL + '&page=2',
    URL.replace('page=1', 'page=2'), URL.replace('api.openalex.org', 'evil.example'),
    URL.replace('api.openalex.org', 'user:password@api.openalex.org'),
    URL.replace('funder-search', 'works'), URL + '#fragment',
])
def test_disallowed_urls_never_request(tmp_path, url):
    session = Session([])
    with pytest.raises(ValueError):
        transport().fetch_snippet_response(url, tmp_path / 'raw', policy=POLICY, session=session)
    assert session.calls == []
    assert not (tmp_path / 'raw').exists()


def test_redirect_and_explicit_quota_do_not_retry(tmp_path):
    for number, response in enumerate([
        Response(302, headers={'Location': 'https://evil.example'}),
        Response(403, payload={'error': 'Daily quota exhausted'}),
    ]):
        session = Session([response])
        result = transport().fetch_snippet_response(URL, tmp_path / str(number), policy=POLICY,
                                                   session=session)
        assert result['status'] == 'failed'
        assert len(session.calls) == 1
        assert result['stop_collection'] is (number == 1)


def test_lower_budgets_and_oversized_stream(tmp_path):
    session = Session([Response(503)])
    result = transport().fetch_snippet_response(
        URL, tmp_path / 'retry', policy={**POLICY, 'timeout_seconds': 2, 'max_retries': 0},
        session=session)
    assert len(result['attempts']) == 1
    assert session.calls[0][1]['timeout'] == 2
    result = transport().fetch_snippet_response(
        URL, tmp_path / 'large', policy={**POLICY, 'max_bytes': 4}, session=Session([Response()]))
    assert result['reason'] == 'response_too_large'
    assert len((tmp_path / 'large' / result['attempts'][0]['raw_path']).read_bytes()) == 4
    assert result['attempts'][0]['body_complete'] is False


def test_interrupted_stream_preserves_partial_response_and_status(tmp_path):
    session = Session([Response(body=b'partial', interruption=requests.ConnectionError('broken')),
                       Response()])
    result = transport().fetch_snippet_response(URL, tmp_path / 'raw', policy=POLICY,
                                               session=session, sleep=lambda _: None)
    first = result['attempts'][0]
    assert first['http_status'] == 200 and first['body_complete'] is False
    assert (tmp_path / 'raw' / first['raw_path']).read_bytes() == b'partial'
    assert first['error_type'] == 'ConnectionError'
    assert result['status'] == 'completed'


def test_destination_reuse_and_interrupted_atomic_write(tmp_path, monkeypatch):
    module = transport()
    target = tmp_path / 'raw'
    target.mkdir()
    with pytest.raises(FileExistsError):
        module.fetch_snippet_response(URL, target, policy=POLICY, session=Session([]))
    def interrupted(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(module, 'atomic_write_new', interrupted)
    with pytest.raises(KeyboardInterrupt):
        module.fetch_snippet_response(URL, tmp_path / 'interrupt', policy=POLICY, session=Session([Response()]))
    assert not (tmp_path / 'interrupt' / 'record.json').exists()


def collect_fake(monkeypatch, tmp_path, responses, config=None):
    module = collector()
    sessions = []
    events = iter(responses)
    def make_session():
        assert all(session.closed for session in sessions)
        session = Session([next(events)])
        sessions.append(session)
        return session
    monkeypatch.setattr(transport().requests, 'Session', make_session)
    result = module.collect_snippets(config or CONFIG, tmp_path / 'bundle')
    return result, sessions


def test_bundle_dedup_blank_budget_and_empty_parent_reservations(tmp_path, monkeypatch):
    config = {**CONFIG, 'max_candidates': 3}
    payload = {'results': [work(1, ['  ', 'NumPy']), work(2, []), work(3, ['NumPy', 'must skip'])]}
    result, sessions = collect_fake(monkeypatch, tmp_path, [Response(payload=payload)], config)
    assert result['status'] == 'partial'
    assert sum(len(session.calls) for session in sessions) == 1
    root = tmp_path / 'bundle'
    documents = read_jsonl(root / 'inputs.jsonl')
    identities = read_jsonl(root / 'identities.jsonl')
    assert [row['text'] for row in documents] == ['NumPy']
    assert len(documents[0]['source_associations']) == 2
    assert {row['source_ids']['openalex'] for row in identities} == {'W1', 'W2', 'W3'}
    empty = next(row for row in identities if row['source_ids']['openalex'] == 'W2')
    assert empty['reason'] == 'no_snippets'
    assert 'document_id' not in empty and 'text' not in empty
    assert empty['source_associations'][0]['work_rank'] == 1
    assert 'snippet_rank' not in empty['source_associations'][0]
    assert result['counts']['candidates'] == 3
    assert result['counts']['rejected'] == 1
    assert len(read_jsonl(root / 'requests.jsonl')) == 10
    assert any(row['reason'] == 'candidate_budget' for row in read_jsonl(root / 'issues.jsonl'))
    assert result['role'] == 'diagnostic'
    assert result['training_eligible'] is result['future_untouched_test_eligible'] is False
    for file in result['files']:
        assert digest((root / file['path']).read_bytes()) == file['sha256']
    assert {'inputs.jsonl', 'identities.jsonl', 'requests.jsonl', 'issues.jsonl', 'config.json'} <= {
        file['path'] for file in result['files']}


@pytest.mark.parametrize('responses,expected', [
    ([Response(payload={'results': []}) for _ in range(10)], 'completed'),
    ([Response(payload={'wrong': []}) for _ in range(10)], 'failed'),
    ([Response(400)] + [Response() for _ in range(9)], 'partial'),
    ([Response(429)], 'failed'),
])
def test_collection_status_and_stop(tmp_path, monkeypatch, responses, expected):
    result, sessions = collect_fake(monkeypatch, tmp_path, responses)
    assert result['status'] == expected
    assert len(sessions) == (1 if responses[0].status_code == 429 else 10)
    assert all(session.closed for session in sessions)
    if expected == 'failed':
        assert read_jsonl(tmp_path / 'bundle' / 'issues.jsonl')


def test_effective_config_frozen_before_request_and_source_reverified(tmp_path, monkeypatch):
    config_path = tmp_path / 'source.json'
    original = json.dumps(CONFIG, indent=4).encode()
    config_path.write_bytes(original)
    config = {**deepcopy(CONFIG), 'config_source': {
        'path': str(config_path), 'sha256': digest(original), 'bytes_utf8': original.decode()}}
    module = collector()
    real_fetch = transport().fetch_snippet_response
    def fetch(url, destination, *, policy):
        assert json.loads((tmp_path / 'bundle' / 'config.json').read_bytes()) == config
        assert (tmp_path / 'bundle' / 'request-plan.jsonl').exists()
        record = real_fetch(url, destination, policy=policy, session=Session([Response(429)]))
        config_path.write_text('{}')
        return record
    monkeypatch.setattr(module, 'fetch_snippet_response', fetch)
    with pytest.raises(ValueError, match='config_source'):
        module.collect_snippets(config, tmp_path / 'bundle')
    assert not (tmp_path / 'bundle' / 'manifest.json').exists()


def test_mismatched_raw_hash_prevents_manifest(tmp_path, monkeypatch):
    module = collector()
    real_fetch = transport().fetch_snippet_response
    def fetch(url, destination, *, policy):
        record = real_fetch(url, destination, policy=policy, session=Session([Response(429)]))
        (destination / record['attempts'][0]['raw_path']).write_bytes(b'changed')
        return record
    monkeypatch.setattr(module, 'fetch_snippet_response', fetch)
    with pytest.raises(ValueError, match='hash'):
        module.collect_snippets(CONFIG, tmp_path / 'bundle')
    assert not (tmp_path / 'bundle' / 'manifest.json').exists()


@pytest.mark.parametrize('key,value', [
    ('page', 2), ('per_page', 6), ('snippets_per_work', 3), ('max_candidates', 101),
    ('timeout_seconds', 36), ('max_retries', 3), ('max_concurrency', 3),
    ('max_bytes', 16777217), ('per_page', True), ('api_key', 'secret'),
    ('endpoint', 'https://api.openalex.org/works'),
])
def test_invalid_configuration_never_creates_destination(tmp_path, key, value):
    with pytest.raises(ValueError):
        collector().collect_snippets({**CONFIG, key: value}, tmp_path / 'bundle')
    assert not (tmp_path / 'bundle').exists()


def test_full_candidate_limit_precedes_deduplication(tmp_path, monkeypatch):
    payload = {'results': [work(number, ['same text', 'same text']) for number in range(1, 6)]}
    result, sessions = collect_fake(monkeypatch, tmp_path, [Response(payload=payload) for _ in range(10)])
    assert result['status'] == 'completed'
    assert result['counts']['candidates'] == 100
    assert result['counts']['documents'] == 1
    assert len(read_jsonl(tmp_path / 'bundle' / 'inputs.jsonl')[0]['source_associations']) == 100
    assert len(sessions) == 10


def test_budget_skipped_candidate_parent_is_still_reserved(tmp_path, monkeypatch):
    payload = {'results': [work(1, ['selected']), work(2, ['budget skipped'])]}
    result, sessions = collect_fake(monkeypatch, tmp_path, [Response(payload=payload)],
                                    {**CONFIG, 'max_candidates': 1})
    root = tmp_path / 'bundle'
    assert result['counts']['candidates'] == result['counts']['documents'] == 1
    assert [row['text'] for row in read_jsonl(root / 'inputs.jsonl')] == ['selected']
    identities = read_jsonl(root / 'identities.jsonl')
    assert {row['source_ids']['openalex'] for row in identities} == {'W1', 'W2'}
    skipped = next(row for row in identities if row['source_ids']['openalex'] == 'W2')
    assert 'text' not in skipped and 'document_id' not in skipped
    association = skipped['source_associations'][0]
    assert association['reason'] == 'candidate_budget'
    assert association['query'] == 'java'
    assert association['work_rank'] == 1 and association['snippet_rank'] == 0
    assert association['raw_snippet'] == 'budget skipped'
    assert association['request']['query_rank'] == 0
    assert digest((root / association['request']['raw_path']).read_bytes()) == association['response_sha256']
    assert len(sessions) == 1


def test_lower_work_and_snippet_limits_preserve_original_ranks(tmp_path, monkeypatch):
    config = {**CONFIG, 'per_page': 1, 'snippets_per_work': 1}
    payload = {'results': [work(1, ['   ', 'do not refill']), work(2, ['do not select'])]}
    result, sessions = collect_fake(monkeypatch, tmp_path, [Response(payload=payload) for _ in range(10)], config)
    assert result['counts']['candidates'] == result['counts']['rejected'] == 10
    assert result['counts']['documents'] == 0
    assert all('per_page=1' in session.calls[0][0] for session in sessions)
    identities = read_jsonl(tmp_path / 'bundle' / 'identities.jsonl')
    assert [row['source_ids']['openalex'] for row in identities] == ['W1']
    assert all(item['work_rank'] == item['snippet_rank'] == 0 for item in identities[0]['source_associations'])


def test_reused_bundle_rejected_without_network(tmp_path):
    root = tmp_path / 'bundle'
    root.mkdir()
    (root / 'existing').write_bytes(b'preserve')
    with pytest.raises(FileExistsError):
        collector().collect_snippets(CONFIG, root)
    assert list(root.iterdir()) == [root / 'existing']


@pytest.mark.parametrize('key,value', [
    ('timeout_seconds', 36), ('max_retries', 3), ('max_bytes', 16777217),
    ('timeout_seconds', True), ('max_bytes', 0), ('max_retries', -1),
])
def test_transport_policy_bounds_rejected_before_request(tmp_path, key, value):
    session = Session([])
    with pytest.raises(ValueError):
        transport().fetch_snippet_response(URL, tmp_path / 'raw', policy={**POLICY, key: value}, session=session)
    assert not (tmp_path / 'raw').exists()


def test_original_config_provenance_preserved(tmp_path, monkeypatch):
    source = tmp_path / 'source.json'
    original = json.dumps(CONFIG, indent=4) + '\n\n'
    source.write_text(original)
    provenance = {'path': str(source), 'bytes_utf8': original, 'sha256': digest(original.encode())}
    config = {**CONFIG, 'config_source': provenance}
    collect_fake(monkeypatch, tmp_path, [Response(429)], config)
    assert json.loads((tmp_path / 'bundle' / 'config.json').read_bytes())['config_source'] == provenance
    assert source.read_text() == original


def test_manifest_not_published_after_interrupted_companion_write(tmp_path, monkeypatch):
    module = collector()
    writer = module.atomic_write_new
    def publish(path, payload):
        if path.name == 'inputs.jsonl':
            raise OSError('disk full')
        writer(path, payload)
    monkeypatch.setattr(module, 'atomic_write_new', publish)
    with pytest.raises(OSError, match='disk full'):
        collect_fake(monkeypatch, tmp_path, [Response(429)])
    assert not (tmp_path / 'bundle' / 'manifest.json').exists()
    assert (tmp_path / 'bundle' / 'requests.jsonl').exists()


@pytest.mark.parametrize('failure', [Response(429), Response(503, headers={'Retry-After': '61'})])
def test_stop_preserves_prior_work_and_never_starts_next_query(tmp_path, monkeypatch, failure):
    result, sessions = collect_fake(monkeypatch, tmp_path, [Response(), failure])
    assert result['status'] == 'partial'
    assert len(sessions) == 2
    assert result['counts']['documents'] == 1
    assert result['counts']['skipped_queries'] == 8
    requests_log = read_jsonl(tmp_path / 'bundle' / 'requests.jsonl')
    assert all(row['status'] == 'skipped' for row in requests_log[2:])
