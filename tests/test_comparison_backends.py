"""Offline boundary tests: no model construction or live service calls."""

from copy import deepcopy
import hashlib
import importlib
import json
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest
import requests

from research.contracts import text_revision


def module(name='backends'):
    return importlib.import_module('research.comparison.' + name)


def arm(backend, config=None, **extra):
    return {'arm_id': backend + '-fixture', 'backend': backend, 'config': config or {},
            'request_options': {}, **extra}


def window(text='NumPy 1.2', name='w1'):
    return {'window_id': name, 'document_id': 'd1', 'text_revision': text_revision('prefix ' + text),
            'start': 7, 'end': 7 + len(text), 'text': text,
            'window_text_revision': text_revision(text), 'content_tokens': 4}


class Response:
    def __init__(self, payload=None, status=200, body=None):
        self.status_code = status
        self.text = json.dumps(payload) if body is None else body
        self.headers = {'Content-Type': 'application/json'}


class Transport:
    def __init__(self, responses=(), health=None):
        self.responses = iter(responses)
        self.health = health or Response({'models': [{'name': 'fixture:1', 'digest': 'sha256:abc'}]})
        self.calls = []
        self.trust_env = True
        self.closed = False

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        value = self.health if method == 'GET' else next(self.responses)
        if isinstance(value, Exception):
            raise value
        return value

    def close(self):
        self.closed = True


def softcite(responses, config=None, **extra):
    config = {'endpoint': 'http://127.0.0.1:8060/service/processSoftwareText', **(config or {})}
    transport = Transport(responses)
    backend = module('backend_softcite').SoftciteBackend(transport=transport)
    loaded = backend.load(arm('softcite', config, **extra))
    assert loaded['status'] == 'ready'
    return backend, transport, loaded


def ollama(contents, config=None, **extra):
    responses = [c if isinstance(c, (Response, Exception)) else Response({'message': {'content': c}})
                 for c in contents]
    transport = Transport(responses)
    backend = module('backend_ollama').OllamaBackend(transport=transport)
    loaded = backend.load(arm('ollama', {'base_url': 'http://127.0.0.1:11434',
                                       'model_id': 'fixture:1', **(config or {})}, **extra))
    assert loaded['status'] == 'ready'
    return backend, transport, loaded


def record(name='NumPy', version='1.2', context='NumPy 1.2'):
    return {'name': name, 'version': version, 'context_sentence': context,
            'intents': ['used'], 'sentiment': 'not_expressed'}


def native(name, start, end):
    return {'rawForm': name, 'offsetStart': start, 'offsetEnd': end}


def test_base_scibert_never_constructs_task_head(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('task model constructor called')
    monkeypatch.setattr(module('backend_scibert'), 'create_detector', forbidden)
    backend = module().make_backend(arm('base', arm_id='scibert-base'))
    loaded = backend.load(arm('base', arm_id='scibert-base'))
    assert loaded['status'] == 'unsupported'
    assert loaded['reason'] == 'no_task_head'
    assert set(loaded['capabilities'].values()) == {False}
    assert backend.predict(window())['status'] == 'unsupported'


def test_unavailable_roster_never_constructs_or_disappears(monkeypatch):
    monkeypatch.setattr(module('backend_scibert'), 'create_detector', lambda *a, **k: pytest.fail('constructed'))
    rows = module().preflight_arms([arm('base', arm_id='scibert-base'),
                                   arm('scibert', unavailable_reason='approval_required'),
                                   arm('modernbert')])
    assert [(r['arm_id'], r['status']) for r in rows] == [
        ('scibert-base', 'unsupported'), ('scibert-fixture', 'unavailable'), ('modernbert-fixture', 'unsupported')]


def test_inactive_rows_preserve_frozen_identity_without_model_construction(monkeypatch):
    monkeypatch.setattr(module('backend_scibert'), 'create_detector', lambda *a, **k: pytest.fail('constructed'))
    requests_ = [
        arm('base', {'revision': 'pinned-base-revision'}, arm_id='scibert-base'),
        arm('modernbert', {'checkpoint': None}),
        arm('scibert', {'checkpoint': 'must-not-load'}, arm_id='scibert-frozen-encoder-probe'),
        arm('scibert', {'checkpoint': 'must-not-load'}, disabled_reason='approval_required'),
        arm('scibert', {'checkpoint': 'must-not-load'}, unsupported_reason='no_compatible_head'),
        arm('fastdict', {'dictionary': 'unverified'}, unavailable_reason='missing_dictionary_provenance'),
    ]
    rows = module().preflight_arms(requests_)
    assert [(row['status'], row['reason']) for row in rows] == [
        ('unsupported', 'no_task_head'), ('unsupported', 'no_trained_compatible_checkpoint'),
        ('unsupported', 'approval_required'), ('unsupported', 'approval_required'),
        ('unsupported', 'no_compatible_head'), ('unavailable', 'missing_dictionary_provenance')]
    assert rows[0]['identity']['config'] == {'revision': 'pinned-base-revision'}
    assert rows[2]['identity']['disabled'] is True
    for row in rows:
        assert row['identity']['config_sha256'] and row['identity']['request_options_sha256']
        assert set(row['capabilities'].values()) == {False}


def test_inactive_config_tampering_fails_with_declared_reason_retained():
    resolved = module().resolve_arm(arm('base', {'revision': 'pinned'}, arm_id='scibert-base'))
    resolved['config']['revision'] = 'changed'
    loaded = module().make_backend(resolved).load(resolved)
    assert loaded['status'] == 'unsupported' and loaded['reason'] == 'no_task_head'
    assert loaded['identity']['config_verified'] is False
    assert 'frozen bytes' in loaded['identity']['config_error']


def test_config_resolution_freezes_bytes_and_detects_mutation(tmp_path):
    path = tmp_path / 'native.json'
    body = b'{ "endpoint": "http://127.0.0.1:8060/process" }\n'
    path.write_bytes(body)
    request = {'arm_id': 's', 'backend': 'softcite', 'config_path': 'native.json',
               'request_options': {'disambiguate': '0'}}
    resolved = module().resolve_arm(request, repo_root=tmp_path)
    assert resolved['config_source']['bytes_utf8'] == body.decode()
    assert resolved['config_source']['sha256'] == hashlib.sha256(body).hexdigest()
    assert resolved['request_options'] == {'disambiguate': '0'}
    assert 'config_path' not in resolved
    assert module().resolve_arm(resolved) == resolved
    tampered = deepcopy(resolved)
    tampered['config']['endpoint'] = 'http://127.0.0.1:9999/changed'
    with pytest.raises(ValueError, match='frozen bytes'):
        module().resolve_arm(tampered)
    path.write_text('{}')
    backend = module('backend_softcite').SoftciteBackend(transport=Transport())
    loaded = backend.load(resolved)
    assert loaded['status'] == 'unavailable'
    assert 'config' in loaded['reason']
    with pytest.raises(ValueError, match='both'):
        module().resolve_arm({**request, 'config': {}})


def test_config_integrity_distinguishes_json_boolean_from_number():
    resolved = module().resolve_arm(arm('ollama', {'seed': 1}))
    resolved['config']['seed'] = True
    with pytest.raises(ValueError, match='frozen bytes'):
        module().resolve_arm(resolved)


@pytest.mark.parametrize('endpoint', ['https://example.com/a', 'http://192.168.1.2/a',
                                    'http://user:password@localhost/a', 'file:///tmp/a',
                                    'http://localhost.evil.test/a', 'http://127.0.0.1/a#fragment'])
def test_rejects_nonlocal_credentials_and_unsupported_urls(endpoint):
    transport = Transport()
    backend = module('backend_softcite').SoftciteBackend(transport=transport)
    loaded = backend.load(arm('softcite', {'endpoint': endpoint}))
    assert loaded['status'] == 'unavailable'
    assert not transport.calls


def test_localhost_requires_only_loopback_and_requests_disable_proxies_redirects(monkeypatch):
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *a, **k: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('127.0.0.1', 8060)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('203.0.113.1', 8060))])
    transport = Transport()
    backend = module('backend_softcite').SoftciteBackend(transport=transport)
    assert backend.load(arm('softcite', {'endpoint': 'http://localhost:8060/process'}))['status'] == 'unavailable'
    assert not transport.calls
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *a, **k: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('127.0.0.1', 8060))])
    backend, transport, _ = softcite([Response({'software': []})], {'endpoint': 'http://localhost:8060/process'})
    assert backend.predict(window())['status'] == 'no_mentions'
    assert transport.trust_env is False
    assert all(call[2]['allow_redirects'] is False for call in transport.calls)
    assert all('127.0.0.1' in call[1] for call in transport.calls)
    assert transport.calls[0][2]['timeout'] == 10
    assert transport.calls[1][2]['timeout'] == 180
    backend.close()
    assert transport.closed


def test_literal_ipv6_is_local_and_localhost_prefers_ipv4_when_available(monkeypatch):
    assert module().local_url('http://[::1]:8060/process') == 'http://[::1]:8060/process'
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *a, **k: [
        (socket.AF_INET6, socket.SOCK_STREAM, 6, '', ('::1', 8060, 0, 0)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('127.0.0.1', 8060))])
    assert module().local_url('http://localhost:8060/process') == 'http://127.0.0.1:8060/process'


@pytest.mark.parametrize('response', [requests.ConnectionError('refused'), requests.Timeout('slow'),
                                      requests.Timeout(),
                                      Response({}, status=302), Response({}, status=500),
                                      Response(body='{"software": ['), Response({}),
                                      Response({'software': {}}), Response({'software': [None]}),
                                      Response(status=204, body='')])
def test_softcite_failure_retains_each_window_and_raw(response):
    backend, _, _ = softcite([response, Response({'software': []})])
    first, second = backend.predict(window()), backend.predict(window(name='w2'))
    assert first['status'] == 'failure'
    assert first['spans'] == [] and first['error']
    assert set(first) == {'window_id', 'status', 'spans', 'unresolved', 'raw', 'error'}
    if isinstance(response, Response):
        assert first['raw']['body'] == response.text
        assert first['raw']['status_code'] == response.status_code
    else:
        assert first['raw']['exception'] == type(response).__name__
    assert second['window_id'] == 'w2' and second['status'] == 'no_mentions'


def test_softcite_exact_offsets_multiple_versions_and_missing_scores():
    payload = {'software': [
        {'software-name': native('NumPy', 0, 5), 'version': native('1.2', 6, 9), 'confidence': .99},
        {'software-name': native('NumPy', 14, 19), 'version': native('2.0', 20, 23)}]}
    backend, transport, loaded = softcite([Response(payload)], request_options={'disambiguate': '0'})
    result = backend.predict(window('NumPy 1.2 and NumPy 2.0.'))
    assert result['status'] == 'success'
    assert [(s['label'], s['start'], s['end']) for s in result['spans']] == [
        ('SOFTWARE', 0, 5), ('VERSION', 6, 9), ('SOFTWARE', 14, 19), ('VERSION', 20, 23)]
    assert all(s['score'] is None and s['score_kind'] is None for s in result['spans'])
    assert transport.calls[1][2]['data'] == {'text': 'NumPy 1.2 and NumPy 2.0.', 'disambiguate': '0'}
    assert loaded['identity']['verified'] is False


def test_softcite_081_mentions_warmup_envelope_is_explicit_empty_success():
    body = '{ "application": "software-mentions", "version": "0.8.1", "date": "2026-09-28T07:46+0000", "mentions": [], "runtime": 2}'
    backend, _, loaded = softcite([Response(body=body)])
    result = backend.predict(window('We used NumPy.'))
    assert result['status'] == 'no_mentions'
    assert result['spans'] == [] and result['error'] is None
    assert result['raw']['body'] == body
    assert result['raw']['native'] == json.loads(body)
    assert loaded['identity']['offset_unit'] == 'unverified'


def test_softcite_081_nonempty_mentions_preserve_repeated_native_offsets():
    text = ('ImageJ is an NIHfunded collaboration between several institutions, groups and individuals, '
            'including Rasband.The ImageJ2 collaboration hopes to create more extensibility, modularity '
            'and interoperability as well as extend ImageJ community resources.ImageJ2 retains the '
            'interface of ImageJ but adds new')
    attributes = {'created': {'score': 0.009934842586517334, 'value': False},
                  'shared': {'score': 0.0001615285873413086, 'value': False},
                  'used': {'score': 0.00048232078552246094, 'value': False}}
    payload = {'application': 'software-mentions', 'version': '0.8.1',
               'date': '2026-09-28T07:46+0000', 'runtime': 1885,
               'mentions': [{'context': text, 'documentContextAttributes': attributes,
                             'mentionContextAttributes': attributes, 'software-type': 'software',
                             'type': 'software', 'software-name': {
                                 'normalizedForm': 'ImageJ', **native('ImageJ', start, end)}}
                            for start, end in [(0, 6), (113, 119), (221, 227), (248, 254), (281, 287)]]}
    backend, _, _ = softcite([Response(payload)])
    result = backend.predict(window(text))
    assert result['status'] == 'success'
    assert [(s['label'], s['start'], s['end']) for s in result['spans']] == [
        ('SOFTWARE', 0, 6), ('SOFTWARE', 113, 119), ('SOFTWARE', 221, 227),
        ('SOFTWARE', 248, 254), ('SOFTWARE', 281, 287)]
    assert all(s['score'] is None and s['score_kind'] is None for s in result['spans'])
    assert result['raw']['native'] == payload


@pytest.mark.parametrize('envelope', ['software', 'mentions'])
def test_softcite_envelopes_share_version_and_nullable_score_validation(envelope):
    payload = {envelope: [
        {'software-name': {**native('NumPy', 0, 5), 'confidence': None},
         'version': {**native('1.2', 6, 9), 'confidence': None}, 'confidence': .99},
        {'software-name': native('NumPy', 14, 19), 'version': native('2.0', 20, 23)},
        {'software-name': native('Tool', 25, 29), 'version': None}]}
    backend, _, _ = softcite([Response(payload)])
    result = backend.predict(window('NumPy 1.2 and NumPy 2.0. Tool'))
    assert result['status'] == 'success'
    assert [(s['label'], s['text'], s['start'], s['end']) for s in result['spans']] == [
        ('SOFTWARE', 'NumPy', 0, 5), ('VERSION', '1.2', 6, 9),
        ('SOFTWARE', 'NumPy', 14, 19), ('VERSION', '2.0', 20, 23), ('SOFTWARE', 'Tool', 25, 29)]
    assert all(s['score'] is None and s['score_kind'] is None for s in result['spans'])
    assert result['raw']['native'] == payload


@pytest.mark.parametrize('payload', [
    {}, [], {'mentions': None}, {'mentions': {}}, {'mentions': '[]'}, {'mentions': [None]},
    {'software': [], 'mentions': None}, {'software': None, 'mentions': []},
    {'software': [], 'mentions': [{'software-name': native('NumPy', 0, 5)}]},
    {'software': [{'software-name': native('NumPy', 0, 5)}], 'mentions': []},
    {'software': [{'software-name': native('NumPy', 0, 5)}],
     'mentions': [{'software-name': native('NumPy', 10, 15)}]},
])
def test_softcite_malformed_or_conflicting_envelopes_remain_failures(payload):
    backend, _, _ = softcite([Response(payload)])
    result = backend.predict(window('NumPy and NumPy'))
    assert result['status'] == 'failure' and result['spans'] == [] and result['error']
    assert result['raw']['native'] == payload
    assert result['raw']['body'] == json.dumps(payload)


@pytest.mark.parametrize('items', [[], [{'software-name': native('NumPy', 0, 5)}]])
def test_softcite_identical_dual_envelopes_do_not_duplicate_spans(items):
    backend, _, _ = softcite([Response({'software': items, 'mentions': deepcopy(items)})])
    result = backend.predict(window('NumPy'))
    assert result['status'] == ('success' if items else 'no_mentions')
    assert len(result['spans']) == len(items)


@pytest.mark.parametrize('envelope', ['software', 'mentions'])
def test_softcite_envelopes_do_not_repair_invalid_version_offsets(envelope):
    payload = {envelope: [{'software-name': native('NumPy', 0, 5), 'version': native('1.2', 0, 3)}]}
    backend, _, _ = softcite([Response(payload)])
    result = backend.predict(window('NumPy 1.2 and NumPy 1.2'))
    assert result['status'] == 'failure' and result['spans'] == []
    assert result['raw']['native'] == payload


@pytest.mark.parametrize('name', [native('NumPy', 1, 6), {'rawForm': 'NumPy'},
                                 {'normalizedForm': 'NumPy', 'offsetStart': 0, 'offsetEnd': 5}])
def test_softcite_never_searches_to_repair_offsets(name):
    backend, _, _ = softcite([Response({'software': [{'software-name': name}]})])
    result = backend.predict(window('NumPy and NumPy'))
    assert result['status'] == 'failure' and result['spans'] == []


@pytest.mark.parametrize('envelope', ['software', 'mentions'])
def test_unknown_unicode_contract_is_not_certified_by_ascii_success(envelope):
    backend, _, loaded = softcite([Response({envelope: [{'software-name': native('NumPy', 0, 5)}]}),
                                  Response({envelope: [{'software-name': native('NumPy', 2, 7)}]})])
    assert backend.predict(window('NumPy'))['status'] == 'success'
    assert loaded['identity']['offset_unit'] == 'unverified'
    result = backend.predict(window('😀 NumPy'))
    assert result['status'] == 'failure'
    assert 'offset' in result['error']


def test_softcite_verified_utf16_evidence_converts_nonbmp(tmp_path):
    evidence = tmp_path / 'offset-evidence.json'
    evidence.write_text(json.dumps({'backend_id': 'softcite-pinned', 'offset_unit': 'utf16',
                                    'method': 'pinned_implementation', 'implementation_revision': 'fixture-revision',
                                    'evidence': 'Pinned implementation uses Java String offsets.'}))
    metadata = {'path': str(evidence), 'sha256': hashlib.sha256(evidence.read_bytes()).hexdigest()}
    backend, _, loaded = softcite([Response({'software': [{'software-name': native('NumPy', 3, 8)}]})],
                                  {'model_id': 'softcite-pinned', 'offset_contract': metadata})
    result = backend.predict(window('😀 NumPy'))
    assert result['status'] == 'success'
    assert result['spans'][0]['start'] == 2 and result['spans'][0]['end'] == 7
    assert result['spans'][0]['alignment_method'] == 'native_utf16'
    assert loaded['identity']['offset_unit'] == 'utf16'


@pytest.mark.parametrize('bad_contract', ['wrong_hash', 'wrong_backend', 'ascii_probe'])
def test_softcite_rejects_unverified_offset_evidence(tmp_path, bad_contract):
    evidence = tmp_path / 'evidence.json'
    data = {'backend_id': 'pinned', 'offset_unit': 'utf16', 'evidence': 'Implementation evidence.',
            'method': 'pinned_implementation', 'implementation_revision': 'fixture-revision'}
    if bad_contract == 'wrong_backend':
        data['backend_id'] = 'other'
    if bad_contract == 'ascii_probe':
        data = {'backend_id': 'pinned', 'offset_unit': 'utf16', 'evidence': 'NumPy offsets 0:5',
                'method': 'synthetic_probe', 'text': 'NumPy', 'native_start': 0, 'native_end': 5, 'span_text': 'NumPy'}
    evidence.write_text(json.dumps(data))
    metadata = {'path': str(evidence), 'sha256': hashlib.sha256(evidence.read_bytes()).hexdigest()}
    if bad_contract == 'wrong_hash':
        metadata['sha256'] = '0' * 64
    backend = module('backend_softcite').SoftciteBackend(transport=Transport())
    loaded = backend.load(arm('softcite', {'endpoint': 'http://127.0.0.1/process', 'model_id': 'pinned',
                                         'offset_contract': metadata}))
    assert loaded['status'] == 'unavailable'


@pytest.mark.parametrize('unit,start,end', [('utf16', 3, 8), ('codepoint', 2, 7)])
def test_softcite_positive_nonbmp_probe_evidence(unit, start, end, tmp_path):
    evidence = tmp_path / 'probe.json'
    evidence.write_text(json.dumps({'backend_id': 'pinned', 'offset_unit': unit,
        'evidence': 'Captured synthetic probe.', 'method': 'synthetic_probe',
        'text': '😀 NumPy', 'native_start': start, 'native_end': end, 'span_text': 'NumPy'}))
    contract = {'path': str(evidence), 'sha256': hashlib.sha256(evidence.read_bytes()).hexdigest()}
    backend, _, _ = softcite([Response({'software': [{'software-name': native('NumPy', start, end)}]})],
                             {'model_id': 'pinned', 'offset_contract': contract})
    result = backend.predict(window('😀 NumPy'))
    assert result['status'] == 'success' and result['spans'][0]['start'] == 2


def test_softcite_204_requires_pinned_evidence(tmp_path):
    evidence = tmp_path / '204.json'
    evidence.write_text(json.dumps({'backend_id': 'pinned', 'http_204_means_no_mentions': True,
                                    'evidence': 'Pinned response contract.'}))
    contract = {'path': str(evidence), 'sha256': hashlib.sha256(evidence.read_bytes()).hexdigest()}
    backend, _, _ = softcite([Response(status=204, body='')], {'model_id': 'pinned', 'empty_204_contract': contract})
    assert backend.predict(window())['status'] == 'no_mentions'


@pytest.mark.parametrize('health', [requests.ConnectionError('refused'), requests.Timeout('slow'), Response(status=302), Response(status=500)])
def test_softcite_unavailable_health_retains_transport_evidence(health):
    backend = module('backend_softcite').SoftciteBackend(transport=Transport(health=health))
    loaded = backend.load(arm('softcite', {'endpoint': 'http://127.0.0.1/process'}))
    assert loaded['status'] == 'unavailable' and loaded['identity']['health']


def test_softcite_request_options_cannot_override_frozen_text():
    transport = Transport()
    backend = module('backend_softcite').SoftciteBackend(transport=transport)
    loaded = backend.load(arm('softcite', {'endpoint': 'http://127.0.0.1/process'}, request_options={'text': 'replacement'}))
    assert loaded['status'] == 'unavailable' and not transport.calls


@pytest.mark.parametrize('content', ['[', '[] trailing', 'Here: []', '{"mentions": []}',
                                    '```json\n[]', '```json\n[]\n``` trailing',
                                    '[{"name": "NumPy"}]'])
def test_ollama_requires_complete_schema_valid_json(content):
    backend, _, _ = ollama([content, '[]'])
    result = backend.predict(window())
    assert result['status'] == 'failure'
    assert result['raw']['content'] == content
    assert backend.predict(window(name='w2'))['status'] == 'no_mentions'


def test_ollama_exact_context_disambiguates_without_version_links():
    text = '😀 NumPy 1.2. Then NumPy 2.0.'
    records = [record(context='😀 NumPy 1.2.'), record(version='2.0', context='Then NumPy 2.0.')]
    backend, transport, loaded = ollama(['```json\n' + json.dumps(records) + '\n```'], variant='L1')
    result = backend.predict(window(text))
    assert result['status'] == 'success'
    assert [(s['label'], s['start'], s['end']) for s in result['spans']] == [
        ('SOFTWARE', 2, 7), ('VERSION', 8, 11), ('SOFTWARE', 18, 23), ('VERSION', 24, 27)]
    assert all(s['score'] is None for s in result['spans'])
    assert loaded['capabilities']['version_linking'] is False
    assert loaded['identity']['model_digest'] == 'sha256:abc'
    assert 'Python' in loaded['identity']['policy_differences'][0]
    payload = transport.calls[1][2]['json']
    assert payload['stream'] is False
    assert payload['options'] == {'temperature': 0, 'seed': 42, 'num_ctx': 8192, 'num_predict': 1024}
    assert payload['format'] == 'json'
    assert transport.calls[1][2]['timeout'] == 240
    assert '"output": []' in payload['messages'][1]['content']


def test_ollama_unresolved_candidates_fail_whole_window():
    backend, _, _ = ollama([json.dumps([record(version=None, context='NumPy and NumPy')])])
    result = backend.predict(window('NumPy and NumPy'))
    assert result['status'] == 'failure' and result['spans'] == []
    assert result['unresolved'][0]['reason'] == 'mention_not_unique_in_context'


def test_ollama_named_request_options_preserved_and_hashed():
    backend, transport, loaded = ollama(['[]'], {'use_format_json': False}, request_options={'options': {'top_k': 7}})
    assert backend.predict(window())['status'] == 'no_mentions'
    payload = transport.calls[1][2]['json']
    assert 'format' not in payload
    assert payload['options']['top_k'] == 7
    assert loaded['identity']['request_options_sha256']
    assert loaded['identity']['prompt_sha256']


@pytest.mark.parametrize('response', [requests.ConnectionError('refused'), requests.Timeout('slow'), Response(status=302),
                                      Response(status=500), Response(body='{'), Response({}),
                                      Response({'message': {'content': '[]'}, 'done': False}),
                                      Response({'message': {'content': '[]'}, 'done': True, 'done_reason': 'length'})])
def test_ollama_transport_or_incomplete_response_never_becomes_empty_success(response):
    backend, _, _ = ollama([response])
    result = backend.predict(window())
    assert result['status'] == 'failure' and result['spans'] == []
    if isinstance(response, Response):
        assert result['raw']['body'] == response.text


def test_ollama_rejects_invalid_nested_record_types_with_raw_retention():
    backend, _, _ = ollama([json.dumps([{**record(), 'intents': [['used']]}])])
    result = backend.predict(window())
    assert result['status'] == 'failure' and result['raw']['content']


@pytest.mark.parametrize('options', [{'stream': True}, {'model': 'different'}, {'messages': []}])
def test_ollama_options_cannot_override_identity_or_frozen_prompt(options):
    transport = Transport()
    backend = module('backend_ollama').OllamaBackend(transport=transport)
    result = backend.load(arm('ollama', {'base_url': 'http://127.0.0.1', 'model_id': 'fixture:1'}, request_options=options))
    assert result['status'] == 'unavailable' and not transport.calls


@pytest.mark.parametrize('health', [requests.ConnectionError('refused'), requests.Timeout('slow'),
                                   Response({}, status=503), Response(body='{'), Response({}),
                                   Response({'models': []})])
def test_ollama_preflight_requires_available_model(health):
    backend = module('backend_ollama').OllamaBackend(transport=Transport(health=health))
    loaded = backend.load(arm('ollama', {'base_url': 'http://127.0.0.1:11434', 'model_id': 'fixture:1'}))
    assert loaded['status'] == 'unavailable' and loaded['reason']


def test_scibert_one_resident_detector_and_minidocument_revision():
    calls = []
    class Detector:
        identity = 'checkpoint-hash'
        manifest = {'capabilities': {k: k in ('software_spans', 'version_spans')
                                     for k in module().CAPABILITY_FIELDS}}
        def predict(self, doc):
            calls.append(doc)
            return {'document_id': doc['document_id'], 'text_revision': doc['text_revision'],
                    'checkpoint_sha256': self.identity, 'capabilities': self.manifest['capabilities'],
                    'offset_unit': 'unicode_codepoint_half_open', 'status': 'success',
                    'chunks': [{'window': 0, 'status': 'success'}],
                    'spans': [{'label': 'SOFTWARE', 'text': 'NumPy', 'start': 0, 'end': 5, 'score': .7}]}
    constructed = []
    def factory(checkpoint, device):
        constructed.append((checkpoint, device))
        return Detector()
    backend = module('backend_scibert').SciBertBackend(detector_factory=factory)
    loaded = backend.load(arm('scibert', {'checkpoint': '/tmp/checkpoint-fixture', 'device': 'cpu'}))
    assert loaded['status'] == 'ready'
    for name in ('w1', 'w2'):
        result = backend.predict(window(name=name))
        assert result['status'] == 'success'
        assert result['raw']['native']['checkpoint_sha256'] == 'checkpoint-hash'
        assert result['spans'][0]['score'] == .7
        assert result['spans'][0]['score_kind'] == 'token_probability_geometric_mean_uncalibrated'
    assert len(constructed) == 1
    assert all(doc['text_revision'] == text_revision('NumPy 1.2') for doc in calls)
    backend.close()
    assert backend.predict(window())['status'] == 'failure'


def test_scibert_rejects_internal_rechunking_and_retains_native():
    detector = SimpleNamespace(identity='x', manifest={'capabilities': {
        k: k in ('software_spans', 'version_spans') for k in module().CAPABILITY_FIELDS}},
        predict=lambda doc: {'chunks': [{'status': 'success'}, {'status': 'success'}], 'spans': [], 'status': 'no_mentions'})
    backend = module('backend_scibert').SciBertBackend(detector_factory=lambda *a, **k: detector)
    assert backend.load(arm('scibert', {'checkpoint': '/tmp/fixture'}))['status'] == 'ready'
    result = backend.predict(window())
    assert result['status'] == 'failure'
    assert len(result['raw']['native']['chunks']) == 2
    assert 'chunk' in result['error']


def test_empty_native_exception_is_still_a_failed_window():
    class Detector:
        identity = 'x'
        manifest = {'capabilities': {k: k in ('software_spans', 'version_spans') for k in module().CAPABILITY_FIELDS}}
        def predict(self, document):
            raise RuntimeError()
    backend = module('backend_scibert').SciBertBackend(detector_factory=lambda *a, **k: Detector())
    assert backend.load(arm('scibert', {'checkpoint': '/tmp/fixture'}))['status'] == 'ready'
    result = backend.predict(window())
    assert result['status'] == 'failure' and result['error']


def test_http_imports_do_not_import_torch():
    code = ('import sys; from research.comparison.backends import make_backend; '
            'make_backend({"backend":"ollama"}); make_backend({"backend":"softcite"}); '
            'assert "torch" not in sys.modules')
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
