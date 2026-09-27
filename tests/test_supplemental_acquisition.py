import importlib
import json
from pathlib import Path
import random
from urllib.parse import parse_qs, urlsplit

import pytest

from research.data.exposure import build_exposures
from research.data.manifest import digest, json_bytes, read_jsonl, write_jsonl, write_once


TARGETS = ['ImageJ', 'GROMACS', 'MATLAB', 'SPSS', 'R', 'BLAST']
QUERY = 'LANG:eng AND OPEN_ACCESS:y AND FIRST_PDATE:[2016-01-01 TO 2025-12-31] sort_date:y'


def config(tmp_path, exclusions=None):
    source = tmp_path / 'reserved.jsonl'
    write_jsonl(source, exclusions or [{'source_ids': {'doi': '10.9999/excluded'}}])
    exposures = tmp_path / 'exposures'
    build_exposures({'inputs': [{'kind': 'documents_jsonl', 'path': str(source),
                                'sha256': digest(source.read_bytes()), 'role': 'diagnostic'}]}, exposures)
    return {'seed': 42, 'metadata_fallback': True, 'ecosystems_projects': {},
            'annotation_policy': 'annotations/scibert-v2/policy-2.1-d17.md',
            'exposures': str(exposures), 'exposures_manifest_sha256': digest((exposures / 'manifest.json').read_bytes())}


def article(index, *, license='by/4.0', language='en', doi=None, pmcid=None, article_type='research-article'):
    return f'''<article xml:lang="{language}" article-type="{article_type}"><front><article-meta>
    <article-id pub-id-type="doi">{doi or f'10.1000/p{index}'}</article-id>
    <article-id pub-id-type="pmcid">{pmcid or f'PMC{index}'}</article-id>
    <title-group><article-title>Study {index}</article-title></title-group>
    <permissions><license><p>https://creativecommons.org/licenses/{license}/</p></license></permissions>
    </article-meta></front><body><p>Unique{index} sample{index} observed{index} effect{index}.</p>
    <p>Measured{index} result{index} described{index}.</p><p>Conclusion{index} finished{index}.</p></body></article>'''.encode()


class FakeFetch:
    def __init__(self, metadata=None, targeted=None, overrides=None, core=None):
        self.metadata = metadata or []
        self.targeted = targeted or {}
        self.overrides = overrides or {}
        self.core = core or {}
        self.calls = []
        self.output = None
        self.on_article = None

    def __call__(self, url, destination, policy):
        self.calls.append(url)
        query = parse_qs(urlsplit(url).query)
        if '/fullTextXML' in url:
            if self.on_article:
                self.on_article()
            index = int(url.split('/PMC')[1].split('/')[0])
            payload = self.overrides.get(index, article(index))
        elif query.get('resultType') == ['core']:
            assert self.output is None or len(list((self.output / 'frames').glob('*.json'))) == 7
            doi = query['query'][0][5:-1]
            index = int(doi.split('/p')[1])
            rows = self.core.get(index, [{'doi': doi, 'pmcid': f'PMC{index}', 'isOpenAccess': 'Y',
                                         'language': 'eng', 'pubTypeList': {'pubType': ['Journal Article']}}])
            payload = json_bytes({'resultList': {'result': rows}})
        else:
            assert query['cursorMark'] == ['*'] and query['format'] == ['json'] and query['synonym'] == ['false']
            target = next((name for name in TARGETS if f'TITLE_ABS:"{name if name != "R" else "R software"}"' in query['query'][0]), None)
            assert query['pageSize'] == ['20' if target else '200']
            rows = self.targeted.get(target, []) if target else self.metadata
            payload = json_bytes({'resultList': {'result': rows}, 'nextCursorMark': 'DO_NOT_FETCH'})
        write_once(destination, payload)
        record = {'url': url, 'sha256': digest(payload), 'retrieved_at_utc': '2026-09-27T00:00:00Z', 'status': 'success'}
        write_once(destination.with_name(destination.name + '.json'), json_bytes(record))
        return record


def setup(monkeypatch, tmp_path, fake, exclusions=None):
    module = importlib.import_module('research.data.supplemental')
    monkeypatch.setattr(module, 'fetch_public', fake)
    output = tmp_path / 'out'
    fake.output = output
    return module, config(tmp_path, exclusions), output


def test_candidate_frame_is_frozen_before_filtering(monkeypatch, tmp_path):
    candidates = [{'doi': f'10.1000/p{i}', 'pmcid': f'PMC{i}'} for i in range(1, 211)]
    fake = FakeFetch(candidates)
    module, cfg, output = setup(monkeypatch, tmp_path, fake)
    manifest = module.collect_supplemental(cfg, output)
    frame = json.loads((output / 'frames/metadata.json').read_bytes())
    assert len(frame['candidates']) == 200
    assert frame['query'] == {'query': QUERY, 'format': 'json', 'resultType': 'lite', 'pageSize': 200, 'cursorMark': '*', 'synonym': 'false'}
    selected = read_jsonl(output / 'selection.jsonl')
    shuffled = candidates[:200].copy()
    random.Random(42).shuffle(shuffled)
    assert [r['doi'] for r in selected] == [r['doi'] for r in shuffled[:6]]
    assert all('DO_NOT_FETCH' not in url for url in fake.calls)
    assert manifest['arms']['metadata'] == {'requested': 6, 'accepted': 6, 'shortfall': 0}
    assert manifest['paper_count'] == 6 and manifest['fulltext_eligible'] is False
    assert all(d['source'] == 'europepmc' and d['development_exposed'] is True for d in read_jsonl(output / 'bundle/documents.jsonl'))
    assert len(read_jsonl(output / 'regions.jsonl')) == 18


def test_fallback_is_explicit_and_target_queries_are_exact(monkeypatch, tmp_path):
    module, cfg, output = setup(monkeypatch, tmp_path, FakeFetch())
    cfg.pop('metadata_fallback')
    with pytest.raises(ValueError, match='METADATA_FALLBACK_REQUIRED'):
        module.freeze_candidate_frames(cfg, output)
    assert not output.exists()
    cfg['metadata_fallback'] = True
    result = module.freeze_candidate_frames(cfg, output)
    for target in TARGETS:
        frame = result['frames'][target]
        expected = QUERY.replace(' sort_date:y', f' AND TITLE_ABS:"{target if target != "R" else "R software"}" sort_date:y')
        assert frame['query']['query'] == expected and frame['discovery_source'] == 'europepmc'


@pytest.mark.parametrize(('override', 'reason'), [
    ({'license': 'by-nc/4.0'}, 'TEXT_LICENSE_UNSUPPORTED'),
    ({'language': 'de'}, 'NON_ENGLISH_ARTICLE'),
    ({'doi': '10.1000/conflicting'}, 'ARTICLE_IDENTITY_MISMATCH'),
    ({'pmcid': 'PMC999'}, 'ARTICLE_IDENTITY_MISMATCH'),
    ({'article_type': 'editorial'}, 'UNSUPPORTED_PUBLICATION_TYPE'),
])
def test_jats_verifies_license_language_identity_and_type(monkeypatch, tmp_path, override, reason):
    fake = FakeFetch([{'doi': '10.1000/p1'}], overrides={1: article(1, **override)})
    module, cfg, output = setup(monkeypatch, tmp_path, fake)
    report = module.collect_supplemental(cfg, output)
    assert report['paper_count'] == 0 and report['status'] == 'failed'
    assert not (output / 'bundle').exists()
    assert reason in {row['code'] for row in read_jsonl(output / 'issues.jsonl')}
    assert report['arms']['metadata']['shortfall'] == 6


def test_other_permitted_licenses_do_not_displace_cc_by_4_parents(monkeypatch, tmp_path):
    fake = FakeFetch([{'doi': '10.1000/p1'}, {'doi': '10.1000/p2'}],
                     targeted={'ImageJ': [{'doi': '10.1000/p3'}, {'doi': '10.1000/p4'}]},
                     overrides={2: article(2, license='by/3.0'), 3: article(3, license='by-sa/4.0')})
    module, cfg, output = setup(monkeypatch, tmp_path, fake)
    report = module.collect_supplemental(cfg, output)
    assert {r['doi'] for r in read_jsonl(output / 'selection.jsonl')} == {'10.1000/p1', '10.1000/p4'}
    assert report['license_exclusions'] == {'CC-BY-3.0': 1, 'CC-BY-SA-4.0': 1}
    assert {r['text_license'] for r in read_jsonl(output / 'permitted-not-detector-ready.jsonl')} == {'CC-BY-3.0', 'CC-BY-SA-4.0'}
    assert report['targets']['ImageJ']['accepted'] == 1 and report['targets']['BLAST']['shortfall'] == 1


def test_missing_doi_and_unresolved_type_are_logged(monkeypatch, tmp_path):
    fake = FakeFetch([{'id': 'missing'}, {'doi': '10.1000/p1'}],
                     core={1: [{'doi': '10.1000/p1', 'pmcid': 'PMC1', 'isOpenAccess': 'Y'}]},
                     overrides={1: article(1, article_type='')})
    module, cfg, output = setup(monkeypatch, tmp_path, fake)
    module.collect_supplemental(cfg, output)
    assert {'MISSING_DOI', 'UNSUPPORTED_PUBLICATION_TYPE'} <= {r['code'] for r in read_jsonl(output / 'issues.jsonl')}


def test_exposure_and_earlier_parent_overlap_rejected(monkeypatch, tmp_path):
    fake = FakeFetch([{'doi': '10.1000/p1'}, {'doi': '10.1000/p2'}], targeted={'ImageJ': [{'doi': '10.1000/p2'}]})
    module, cfg, output = setup(monkeypatch, tmp_path, fake, [{'source_ids': {'doi': '10.1000/p1'}}])
    report = module.collect_supplemental(cfg, output)
    assert report['paper_count'] == 1 and report['targets']['ImageJ']['shortfall'] == 1
    assert len([r for r in read_jsonl(output / 'issues.jsonl') if r['code'] == 'EXCLUDED_OVERLAP']) == 2


def test_core_pmcid_conflict_rejects_parent(monkeypatch, tmp_path):
    fake = FakeFetch([{'doi': '10.1000/p1', 'pmcid': 'PMC999'}])
    module, cfg, output = setup(monkeypatch, tmp_path, fake)
    report = module.collect_supplemental(cfg, output)
    assert report['paper_count'] == 0
    assert 'CANDIDATE_IDENTITY_CONFLICT' in {r['code'] for r in read_jsonl(output / 'issues.jsonl')}


def test_pinned_exposure_and_config_source_are_checked_before_fetch(monkeypatch, tmp_path):
    fake = FakeFetch()
    module, cfg, output = setup(monkeypatch, tmp_path, fake)
    cfg['exposures_manifest_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='EXPOSURE_MANIFEST_MISMATCH'):
        module.collect_supplemental(cfg, output)
    assert not fake.calls


def test_exposure_sources_rechecked_before_publication(monkeypatch, tmp_path):
    fake = FakeFetch([{'doi': '10.1000/p1'}])
    module, cfg, output = setup(monkeypatch, tmp_path, fake)
    fake.on_article = lambda: (tmp_path / 'reserved.jsonl').write_text('{}\n')
    with pytest.raises(ValueError):
        module.collect_supplemental(cfg, output)
    assert not (output / 'manifest.json').exists()
    assert not (output / 'bundle/manifest.json').exists()


def test_verified_loader_detects_region_or_source_tampering(monkeypatch, tmp_path):
    fake = FakeFetch([{'doi': '10.1000/p1'}])
    module, cfg, output = setup(monkeypatch, tmp_path, fake)
    module.collect_supplemental(cfg, output)
    loaded = module.load_supplemental(output)
    assert len(loaded['documents']) == 1 and len(loaded['regions']) == 3
    (output / 'regions.jsonl').write_text('{}\n')
    with pytest.raises(ValueError):
        module.load_supplemental(output)


def test_verified_ecosystems_route_retains_real_association_source(monkeypatch, tmp_path):
    fake = FakeFetch()
    module, cfg, output = setup(monkeypatch, tmp_path, fake)
    project = 'https://papers.ecosyste.ms/api/v1/projects/verified-imagej'
    cfg['ecosystems_projects'] = {'ImageJ': {'project_id': '27', 'url': project}}
    responses = {
        project: {'id': 27, 'project_url': project, 'name': 'ImageJ'},
        project + '/mentions?per_page=20&page=1': [{'project_url': project, 'paper_url': 'https://papers.ecosyste.ms/api/v1/papers/10.1000%2Fp1'}],
        'https://papers.ecosyste.ms/api/v1/papers/10.1000%2Fp1': {'doi': '10.1000/p1'},
    }
    def fetch(url, destination, policy):
        if url not in responses:
            return fake(url, destination, policy)
        payload = json_bytes(responses[url])
        write_once(destination, payload)
        record = {'url': url, 'sha256': digest(payload), 'retrieved_at_utc': '2026-09-27T00:00:00Z'}
        write_once(destination.with_name(destination.name + '.json'), json_bytes(record))
        return record
    monkeypatch.setattr(module, 'fetch_public', fetch)
    report = module.collect_supplemental(cfg, output)
    doc = module.load_supplemental(output)['documents'][0]
    assert report['sources'] == {'ecosystems': 1}
    assert doc['source'] == doc['discovery_source'] == 'ecosystems'
    assert doc['acquisition_arm'] == 'targeted' and doc['association']['project_url'] == project
    frame = json.loads((output / 'frames/ImageJ.json').read_bytes())
    assert frame['query'] == {'per_page': 20, 'page': 1} and frame['project_metadata']['url'] == project


def test_bad_project_identity_does_not_silently_fallback(monkeypatch, tmp_path):
    fake = FakeFetch()
    module, cfg, output = setup(monkeypatch, tmp_path, fake)
    project = 'https://papers.ecosyste.ms/api/v1/projects/verified-imagej'
    cfg['ecosystems_projects'] = {'ImageJ': {'project_id': '27', 'url': project}}
    def fetch(url, destination, policy):
        if url != project:
            return fake(url, destination, policy)
        payload = json_bytes({'id': 28, 'project_url': project, 'name': 'ImageJ'})
        write_once(destination, payload)
        return {'url': url, 'sha256': digest(payload)}
    monkeypatch.setattr(module, 'fetch_public', fetch)
    report = module.collect_supplemental(cfg, output)
    assert report['targets']['ImageJ']['shortfall'] == 1
    assert not any('ImageJ' in parse_qs(urlsplit(url).query).get('query', [''])[0] for url in fake.calls)
    assert 'PROJECT_IDENTITY_MISMATCH' in {r['code'] for r in read_jsonl(output / 'issues.jsonl')}
    loaded = module.load_supplemental(output)
    assert loaded['documents'] == loaded['regions'] == [] and 'no_eligible_parents' in loaded['manifest']['blockers']


def test_original_config_bytes_are_preserved_and_rechecked(monkeypatch, tmp_path):
    fake = FakeFetch([{'doi': '10.1000/p1'}])
    module, cfg, output = setup(monkeypatch, tmp_path, fake)
    source = tmp_path / 'source-config.json'
    source.write_text('{"metadata_fallback": true}\n')
    cfg['config_source'] = {'path': str(source), 'sha256': digest(source.read_bytes()), 'bytes_utf8': source.read_text()}
    fake.on_article = lambda: source.write_text('{}\n')
    with pytest.raises(ValueError, match='CONFIG_SOURCE_MISMATCH'):
        module.collect_supplemental(cfg, output)
    assert not (output / 'manifest.json').exists()
    assert json.loads((output / 'config.json').read_bytes())['config_source'] == cfg['config_source']


def test_frozen_frame_tampering_cannot_be_published(monkeypatch, tmp_path):
    fake = FakeFetch([{'doi': '10.1000/p1'}])
    module, cfg, output = setup(monkeypatch, tmp_path, fake)
    fake.on_article = lambda: (output / 'frames/metadata.json').write_text('{}\n')
    with pytest.raises(ValueError):
        module.collect_supplemental(cfg, output)
    assert not (output / 'manifest.json').exists()


def test_exposure_bundle_loaded_once_per_collection(monkeypatch, tmp_path):
    fake = FakeFetch([{'doi': f'10.1000/p{i}'} for i in range(1, 8)])
    module, cfg, output = setup(monkeypatch, tmp_path, fake)
    real_load = module.load_exposures
    loads = []
    def counted_load(path):
        loads.append(path)
        return real_load(path)
    monkeypatch.setattr(module, 'load_exposures', counted_load)
    assert module.collect_supplemental(cfg, output)['paper_count'] == 6
    assert len(loads) == 1


def test_collection_reports_context_exclusion_and_passage_shortfall(monkeypatch, tmp_path):
    raw = article(1).replace(b'<p>Measured1 result1 described1.</p>', b'<p>' + b'x' * 6001 + b'.</p>')
    fake = FakeFetch([{'doi': '10.1000/p1'}], overrides={1: raw})
    module, cfg, output = setup(monkeypatch, tmp_path, fake)
    report = module.collect_supplemental(cfg, output)
    assert report['paper_count'] == 1 and report['passage_count'] == 2
    issues = read_jsonl(output / 'issues.jsonl')
    assert {'CONTEXT_EXCEEDS_LIMIT', 'PASSAGE_SHORTFALL'} <= {row['code'] for row in issues}
    parent = read_jsonl(output / 'selection.jsonl')[0]
    assert parent['random_passages'] == {'requested': 3, 'accepted': 2, 'shortfall': 1}
    assert parent['signal_passages'] == {'requested': 3, 'accepted': 0, 'shortfall': 3}


def test_loader_rechecks_original_exposure_sources(monkeypatch, tmp_path):
    fake = FakeFetch([{'doi': '10.1000/p1'}])
    module, cfg, output = setup(monkeypatch, tmp_path, fake)
    module.collect_supplemental(cfg, output)
    loaded = module.load_supplemental(output)
    (tmp_path / 'reserved.jsonl').write_text('{}\n')
    with pytest.raises(ValueError):
        module.verify_supplemental_sources(loaded)
    with pytest.raises(ValueError):
        module.load_supplemental(output)


def test_loader_requires_all_outer_companions(monkeypatch, tmp_path):
    fake = FakeFetch([{'doi': '10.1000/p1'}])
    module, cfg, output = setup(monkeypatch, tmp_path, fake)
    report = module.collect_supplemental(cfg, output)
    report['files'] = [record for record in report['files'] if record['path'] != 'regions.jsonl']
    (output / 'manifest.json').write_bytes(json_bytes(report))
    with pytest.raises(ValueError, match='MISSING_SUPPLEMENTAL_COMPANION'):
        module.load_supplemental(output)


def test_loader_rejects_mismatched_parent_selection(monkeypatch, tmp_path):
    fake = FakeFetch([{'doi': '10.1000/p1'}])
    module, cfg, output = setup(monkeypatch, tmp_path, fake)
    report = module.collect_supplemental(cfg, output)
    rows = read_jsonl(output / 'selection.jsonl')
    rows[0]['document_id'] = 'wrong-parent'
    (output / 'selection.jsonl').write_text(json.dumps(rows[0]) + '\n')
    for record in report['files']:
        if record['path'] == 'selection.jsonl':
            record['sha256'] = digest((output / 'selection.jsonl').read_bytes())
    (output / 'manifest.json').write_bytes(json_bytes(report))
    with pytest.raises(ValueError, match='INVALID_SUPPLEMENTAL_SELECTION'):
        module.load_supplemental(output)
