import hashlib
import io
import json
import zipfile

import pytest

from research.data import acquire, manifest


class Response:
    def __init__(self, data=b'[]', status=200, headers=None):
        self.content = data
        self.status_code = status
        self.headers = headers or {'Content-Type': 'application/json'}


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.urls = []

    def get(self, url, **kwargs):
        assert kwargs['timeout'] == (10, 60)
        assert kwargs['allow_redirects'] is False
        self.urls.append(url)
        return next(self.responses)


def test_cache_never_replaced_and_retry_exhaustion_is_not_empty(tmp_path):
    session = Session([Response(b'[{"id":1}]')] + [Response(status=429)] * 5)
    waits = []
    policy = {'session': session, 'sleep': waits.append, 'format': 'json'}
    dest = tmp_path / 'cached'
    first = acquire.fetch_public('https://example.org/data', dest, policy)
    assert acquire.fetch_public('https://example.org/data', dest, policy)['sha256'] == first['sha256']
    assert len(session.urls) == 1
    with pytest.raises(acquire.AcquisitionError, match='rate_limited'):
        acquire.fetch_public('https://example.org/next', tmp_path / 'next', policy)
    assert dest.read_bytes() == b'[{"id":1}]'
    assert waits == [1, 2, 4, 8]
    assert not (tmp_path / 'next').exists()


def test_retry_after_is_respected(tmp_path):
    waits = []
    policy = {'session': Session([Response(status=503, headers={'Retry-After': '12'}), Response()]),
              'sleep': waits.append, 'format': 'json'}
    assert acquire.fetch_public('https://example.org/a', tmp_path / 'a', policy)['attempts'] == 2
    assert waits == [12]


@pytest.mark.parametrize('body,expected', [(b'<html>error</html>', 'invalid_json'), (b'{}', 'checksum_mismatch')])
def test_bad_json_and_checksum_never_cached(tmp_path, body, expected):
    with pytest.raises(acquire.AcquisitionError, match=expected):
        acquire.fetch_public('https://example.org/a', tmp_path / 'a',
                             {'session': Session([Response(body)]), 'format': 'json', 'sha256': '0' * 64})
    assert not (tmp_path / 'a').exists()


@pytest.mark.parametrize('url', ['http://example.org', 'https://user:secret@example.org',
                                'https://127.0.0.1/a', 'https://example.org/?api_key=secret'])
def test_unsafe_or_credential_urls_rejected(tmp_path, url):
    with pytest.raises(acquire.AcquisitionError, match='unsafe_url'):
        acquire.fetch_public(url, tmp_path / 'a', {})


def test_redirect_is_validated_before_request(tmp_path):
    session = Session([Response(status=302, headers={'Location': 'http://127.0.0.1/secret'})])
    with pytest.raises(acquire.AcquisitionError, match='unsafe_url'):
        acquire.fetch_public('https://example.org/a', tmp_path / 'a', {'session': session})
    assert len(session.urls) == 1


def test_changed_cache_or_query_is_rejected(tmp_path):
    dest = tmp_path / 'a'
    policy = {'session': Session([Response()]), 'format': 'json'}
    acquire.fetch_public('https://example.org/a', dest, policy)
    with pytest.raises(acquire.AcquisitionError, match='cache_conflict'):
        acquire.fetch_public('https://example.org/b', dest, policy)
    dest.write_bytes(b'corrupt')
    with pytest.raises(acquire.AcquisitionError, match='checksum_mismatch'):
        acquire.fetch_public('https://example.org/a', dest, policy)


def test_exact_pagination_deduplicates_ids_without_making_negatives(tmp_path):
    session = Session([Response(b'[{"doi":"10.1/X"},{"doi":"10.1/Y"}]'),
                       Response(b'[{"doi":"10.1/X"},{"doi":"10.1/Z"}]')])
    config = {'name': 'ecosystems', 'kind': 'metadata', 'url': 'https://example.org/papers',
              'id_field': 'doi', 'page_size': 2, 'max_pages': 2, 'target_candidates': 3,
              'policy': {'session': session}, 'metadata_license': 'CC-BY-SA-4.0'}
    rows = acquire.collect_candidates(config, tmp_path)
    assert len(rows) == 3
    assert session.urls == ['https://example.org/papers?page=1&per_page=2', 'https://example.org/papers?page=2&per_page=2']
    assert all(r['role'] == 'weak_candidate' and not r['fulltext_eligible'] for r in rows)
    pages = manifest.read_jsonl(tmp_path / 'pages.jsonl')
    assert [p['returned_count'] for p in pages] == [2, 2]
    assert acquire.component_url('https://example.org/papers', '10.1/a/b', 'mentions') == 'https://example.org/papers/10.1%2Fa%2Fb/mentions'


@pytest.mark.parametrize('member', ['../outside', '/absolute', 'C:/absolute', 'a/../../bad', 'a\\..\\bad'])
def test_zip_traversal_rejected_without_extracting(tmp_path, member):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        z.writestr(member, 'bad')
    with pytest.raises(acquire.AcquisitionError, match='unsafe_archive'):
        acquire.zip_members(buf.getvalue())
    assert list(tmp_path.iterdir()) == []


def test_access_basis_and_scope_are_not_inferred_from_metadata_license():
    assert not manifest.fulltext_eligible({'metadata_license': 'CC0', 'supplied_text_scope': 'fulltext'})
    assert not manifest.fulltext_eligible({'text_license': 'CC-BY-4.0', 'public': True, 'supplied_text_scope': 'methods'})
    assert manifest.fulltext_eligible({'text_license': 'CC-BY-4.0', 'access_basis': 'source_license',
                                      'public': True, 'supplied_text_scope': 'fulltext', 'language': 'en'})


def test_manifest_path_and_hash_are_verified(tmp_path):
    path = tmp_path / 'blob'
    path.write_bytes(b'ok')
    row = {'path': 'blob', 'sha256': hashlib.sha256(b'ok').hexdigest()}
    assert manifest.verified_path(tmp_path, row) == path
    for bad in [{**row, 'path': '../outside'}, {**row, 'sha256': '0'*64}]:
        with pytest.raises(ValueError):
            manifest.verified_path(tmp_path, bad)


def test_symlinks_can_be_excluded_but_never_followed():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        entry = zipfile.ZipInfo('Label/annotation.conf')
        entry.create_system = 3
        entry.external_attr = 0o120777 << 16
        z.writestr(entry, '../../conf/annotation.conf')
        z.writestr('Label/paper.txt', 'Paper text')
    with pytest.raises(acquire.AcquisitionError, match='unsafe_archive'):
        acquire.zip_members(buf.getvalue())
    assert acquire.zip_members(buf.getvalue(), skip_symlinks=True) == {'Label/paper.txt': b'Paper text'}


def test_audit_reports_infeasible_fulltext_quota_and_preserves_input(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    rows = [{'source': 'ecosystems', 'status': 'success', 'role': 'weak_candidate',
             'source_record_id': 'a', 'fulltext_eligible': False}]
    manifest.write_jsonl(raw / 'acquisitions.jsonl', rows)
    report = acquire.audit_sources(raw / 'acquisitions.jsonl', tmp_path / 'audit')
    assert report['sources']['ecosystems']['eligible_fulltext'] == 0
    assert report['sources']['ecosystems']['required'] == 20
    assert report['sources']['ecosystems']['shortage'] == 20
    assert report['status'] == 'blocked'
    assert manifest.read_jsonl(raw / 'acquisitions.jsonl') == rows


def test_acquire_text_requires_explicit_access_and_rejects_error_pages(tmp_path):
    source = {'url':'https://example.org/paper', 'format':'text', 'source':'openalex',
              'source_record_id':'a', 'public':True, 'supplied_text_scope':'fulltext', 'language':'en'}
    with pytest.raises(acquire.AcquisitionError, match='text_access_unverified'):
        acquire.acquire_text(source, tmp_path)
    source.update(text_license='CC-BY-4.0', access_basis='https://example.org/license')
    source['policy'] = {'session':Session([Response(b'<html>Access denied</html>')])}
    with pytest.raises(acquire.AcquisitionError, match='unexpected_html'):
        acquire.acquire_text(source, tmp_path)


def test_acquired_plaintext_keeps_extractor_hash_and_provenance(tmp_path):
    source = {'url':'https://example.org/paper', 'format':'text', 'source':'openalex',
              'source_record_id':'a', 'public':True, 'supplied_text_scope':'fulltext', 'language':'en',
              'text_license':'CC-BY-4.0', 'access_basis':'https://example.org/license',
              'policy':{'session':Session([Response(b'We used X.')])}}
    row = acquire.acquire_text(source, tmp_path)
    assert row['fulltext_eligible'] is True
    assert row['extractor_version'] == 'plain-utf8-v1'
    assert manifest.verified_path(tmp_path, row).read_text() == 'We used X.'


def test_configured_paper_text_acquisition_audit_build_roundtrip(tmp_path, monkeypatch):
    from research.data.corpus import build_corpus
    original=acquire.fetch_public
    def fetch(url,path,policy):
        return original(url,path,{**policy,'session':Session([Response(b'We used X.')])})
    monkeypatch.setattr(acquire,'fetch_public',fetch)
    config={'sources':[{'name':'openalex','kind':'paper_text','url':'https://example.org/paper','format':'text',
                        'source_record_id':'W1','source_ids':{'doi':'10.1/paper','openalex':'W1'},
                        'public':True,'language':'en','supplied_text_scope':'fulltext',
                        'text_license':'CC-BY-4.0','access_basis':'https://example.org/license'}]}
    p=tmp_path/'config.json';p.write_text(json.dumps(config))
    assert acquire.acquire_sources(p,tmp_path/'raw')['status']=='success'
    report=acquire.audit_sources(tmp_path/'raw/acquisitions.jsonl',tmp_path/'audit')
    assert report['sources']['openalex']['eligible_fulltext']==1
    assert build_corpus(tmp_path/'raw/acquisitions.jsonl',tmp_path/'corpus')['eligible_fulltext']==1
    assert manifest.read_jsonl(tmp_path/'corpus/documents.jsonl')[0]['source_ids']=={'doi':'10.1/paper','openalex':'W1'}
