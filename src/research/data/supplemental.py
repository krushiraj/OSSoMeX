"""Bounded licensed supplemental acquisition with frozen discovery and exposure evidence."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random
import re
from urllib.parse import unquote, urlencode, urlsplit

from ..contracts import validate_document
from .acquire import AcquisitionError, component_url, fetch_public
from .bundles import load_bundle, materialize_bundle
from .ecosystems import API, EPMC, ExclusionIndex, ecosystem_url
from .exposure import exposure_reasons, load_exposures, verify_exposure_sources
from .jats import normalize_doi, read_jats, safe_xml
from .manifest import digest, json_bytes, read_jsonl, verified_path, write_jsonl, write_once
from .supplemental_passages import passage_candidates, select_passages

ROOT = Path(__file__).resolve().parents[3]
TARGETS = ('ImageJ', 'GROMACS', 'MATLAB', 'SPSS', 'R', 'BLAST')
POLICY = 'annotations/scibert-v2/policy-2.1-d17.md'
QUERY = 'LANG:eng AND OPEN_ACCESS:y AND FIRST_PDATE:[2016-01-01 TO 2025-12-31] sort_date:y'
SCHEMA = 'supplemental-1.0'
SELECTION_POLICY = {
    'metadata_parents': 6, 'targeted_parents': 6, 'targets': list(TARGETS),
    'metadata_frame_limit': 200, 'target_frame_limit': 20, 'seed': 42,
    'random_passages_per_parent': 3, 'signal_passages_per_parent': 3,
    'max_owned_passages': 72, 'max_context_chars': 6000,
    'random_sampling': 'sample eligible sentences in source order before signal selection; seed 42 per parent',
    'metadata_sampling': 'shuffle frozen frame once with seed 42; first six eligible, independent of software matches',
    'signal_sampling': 'unselected literal name with adjacent version-like string first, then literal name; source-order ties',
    'target_sampling': 'first eligible parent in each frozen target frame; no target substitution',
    'detector_text_license': 'CC-BY-4.0', 'annotation_policy': POLICY,
    'version_goal': {'explicit_spans': 20, 'parents': 6, 'verification': 'requires annotation; not measured by acquisition'},
    'transport': {'implementation': 'research.data.acquire.fetch_public', 'attempts': 5,
                  'timeout_seconds': [10, 60], 'max_bytes': 16 * 1024 * 1024},
}


def _verify_config_source(config):
    if 'config_source' not in config:
        return
    source = config['config_source']
    if (not isinstance(source, dict) or not isinstance(source.get('path'), str)
            or not Path(source['path']).is_absolute()):
        raise ValueError('INVALID_CONFIG_SOURCE')
    payload = Path(source['path']).read_bytes()
    if digest(payload) != source.get('sha256') or payload.decode('utf-8') != source.get('bytes_utf8'):
        raise ValueError('CONFIG_SOURCE_MISMATCH')


def _validate_config(config):
    if not isinstance(config, dict):
        raise ValueError('INVALID_SUPPLEMENTAL_CONFIG')
    allowed = {'seed', 'metadata_fallback', 'ecosystems_projects', 'annotation_policy',
               'exposures', 'exposures_manifest_sha256', 'config_source'}
    if set(config) - allowed:
        raise ValueError('UNKNOWN_SUPPLEMENTAL_SETTING')
    if type(config.get('seed', 42)) is not int or config.get('seed', 42) != 42:
        raise ValueError('INVALID_SEED')
    if config.get('annotation_policy', POLICY) != POLICY:
        raise ValueError('ANNOTATION_POLICY_MISMATCH')
    projects = config.get('ecosystems_projects', {})
    if not isinstance(projects, dict) or set(projects) - set(TARGETS):
        raise ValueError('INVALID_ECOSYSTEMS_PROJECTS')
    if len(projects) < len(TARGETS) and config.get('metadata_fallback') is not True:
        raise ValueError('METADATA_FALLBACK_REQUIRED')
    for spec in projects.values():
        if not isinstance(spec, dict) or set(spec) != {'project_id', 'url'}:
            raise ValueError('INVALID_ECOSYSTEMS_PROJECT')
        if not isinstance(spec['project_id'], str) or not spec['project_id'].strip():
            raise ValueError('INVALID_ECOSYSTEMS_PROJECT')
        ecosystem_url(spec['url'], 'projects')
    path = config.get('exposures')
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ValueError('ABSOLUTE_EXPOSURES_REQUIRED')
    if not re.fullmatch(r'[0-9a-f]{64}', str(config.get('exposures_manifest_sha256', ''))):
        raise ValueError('EXPOSURE_MANIFEST_MISMATCH')
    _verify_config_source(config)


def _load_exposures(config):
    path = Path(config['exposures'])
    if digest((path / 'manifest.json').read_bytes()) != config['exposures_manifest_sha256']:
        raise ValueError('EXPOSURE_MANIFEST_MISMATCH')
    exposures = load_exposures(path)
    if exposures['_manifest_sha256'] != config['exposures_manifest_sha256']:
        raise ValueError('EXPOSURE_MANIFEST_MISMATCH')
    verify_exposure_sources(exposures)
    return exposures


def _fetch(url, output, *, is_json=True):
    path = output / 'requests' / digest(url.encode())
    record = fetch_public(url, path, {'max_bytes': 16 * 1024 * 1024,
                                     **({'format': 'json'} if is_json else {})})
    payload = path.read_bytes()
    if len(payload) > 16 * 1024 * 1024 or digest(payload) != record['sha256'] or record['url'] != url:
        raise ValueError('INVALID_REQUEST_EVIDENCE')
    return (json.loads(payload) if is_json else payload), {**record, 'path': str(path.relative_to(output))}


def _results(body):
    if not isinstance(body, dict) or not isinstance(body.get('resultList'), dict):
        raise ValueError('INVALID_PMC_SEARCH_RESPONSE')
    rows = body['resultList'].get('result')
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError('INVALID_PMC_SEARCH_RESULTS')
    return rows


def _freeze(config, output, exposures):
    write_once(output / 'config.json', json_bytes(config))
    write_once(output / 'selection-policy.json', json_bytes(SELECTION_POLICY))
    write_once(output / 'annotation-policy.md', (ROOT / POLICY).read_bytes())
    frames, issues = {}, []
    for target in ('metadata', *TARGETS):
        limit = 200 if target == 'metadata' else 20
        spec = config.get('ecosystems_projects', {}).get(target)
        frame = {'target': None if target == 'metadata' else target, 'limit': limit,
                 'acquisition_arm': 'metadata' if target == 'metadata' else 'targeted',
                 'discovery_source': 'ecosystems' if spec else 'europepmc', 'candidates': [],
                 'selection_bias': 'newest-first bounded frame, not representative of all years' if not spec else 'configured project association frame',
                 'next_page_calls': 0}
        try:
            if spec:
                frame['project'] = deepcopy(spec)
                project, project_record = _fetch(spec['url'], output)
                if (not isinstance(project, dict) or str(project.get('id')) != spec['project_id']
                        or project.get('project_url') != spec['url'] or not isinstance(project.get('name'), str)
                        or not project['name'].strip()):
                    raise ValueError('PROJECT_IDENTITY_MISMATCH')
                frame['project_metadata'] = project_record
                frame['query'] = {'per_page': 20, 'page': 1}
                url = spec['url'] + '/mentions?' + urlencode(frame['query'])
                rows, record = _fetch(url, output)
                if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                    raise ValueError('INVALID_MENTIONS_RESPONSE')
            else:
                query = QUERY if target == 'metadata' else QUERY.replace(
                    ' sort_date:y', ' AND TITLE_ABS:"' + ('R software' if target == 'R' else target) + '" sort_date:y')
                frame['query'] = {'query': query, 'format': 'json', 'resultType': 'lite',
                                  'pageSize': limit, 'cursorMark': '*', 'synonym': 'false'}
                url = EPMC + '/search?' + urlencode(frame['query'])
                body, record = _fetch(url, output)
                rows = _results(body)
            frame.update({'request': record, 'query_url': url, 'returned_count': len(rows),
                          'candidates': rows[:limit], 'status': 'frozen'})
        except (AcquisitionError, ValueError, OSError) as exc:
            frame.update({'status': 'failed', 'code': str(exc)})
            issues.append({'stage': 'frame', 'target': target, 'code': str(exc)})
        write_once(output / 'frames' / f'{target}.json', json_bytes(frame))
        frames[target] = frame
    verify_exposure_sources(exposures)
    _verify_config_source(config)
    artifacts = [{'path': str(path.relative_to(output)), 'sha256': digest(path.read_bytes())}
                 for path in sorted(output.rglob('*')) if path.is_file()]
    return {'frames': frames, 'issues': issues, 'files': artifacts}


def freeze_candidate_frames(config: dict, output: Path) -> dict:
    config = deepcopy(config)
    _validate_config(config)
    exposures = _load_exposures(config)
    return _freeze(config, Path(output), exposures)


def _doi(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('MISSING_DOI')
    doi = normalize_doi(value)
    if not re.fullmatch(r'10\.\d{4,9}/[^\s"<>]+', doi):
        raise ValueError('INVALID_DOI')
    return doi


def _publication_type(core, raw):
    type_list = core.get('pubTypeList') or {}
    if not isinstance(type_list, dict):
        raise ValueError('UNSUPPORTED_PUBLICATION_TYPE')
    types = type_list.get('pubType', [])
    if isinstance(types, str):
        types = [types]
    if not isinstance(types, list) or any(not isinstance(value, str) for value in types):
        raise ValueError('UNSUPPORTED_PUBLICATION_TYPE')
    types = {value.lower().strip() for value in types}
    denied = ('editorial', 'correction', 'erratum', 'retraction', 'reviewer', 'peer review', 'comment', 'letter')
    jats_type = safe_xml(raw).get('article-type', '').lower().strip()
    if any(any(term in value for term in denied) for value in types | {jats_type}):
        raise ValueError('UNSUPPORTED_PUBLICATION_TYPE')
    if jats_type:
        permitted = jats_type in {'research-article', 'review-article', 'review', 'article'}
    else:
        permitted = bool(types & {'journal article', 'research article', 'review', 'article', 'systematic review'})
    if not permitted:
        raise ValueError('UNSUPPORTED_PUBLICATION_TYPE')
    return {'jats_article_type': jats_type or None, 'core_publication_types': sorted(types)}


def _resolve(candidate, frame, output):
    metadata = {}
    aliases = {}
    if frame['discovery_source'] == 'ecosystems':
        project_url = frame['project']['url']
        if candidate.get('project_url') != project_url:
            raise ValueError('ASSOCIATION_PROJECT_MISMATCH')
        paper_url = ecosystem_url(candidate.get('paper_url'), 'papers')
        doi = _doi(unquote(urlsplit(paper_url).path.removeprefix('/api/v1/papers/')))
        paper, record = _fetch(component_url(API + '/papers', doi), output)
        if not isinstance(paper, dict) or _doi(paper.get('doi')) != doi:
            raise ValueError('PAPER_IDENTITY_MISMATCH')
        if paper.get('openalex_id'):
            aliases['openalex'] = paper['openalex_id']
        metadata = {'paper_metadata': record, 'association': deepcopy(candidate),
                    'project_metadata': frame['project_metadata']}
    else:
        doi = _doi(candidate.get('doi'))
    query = {'query': 'DOI:"' + doi + '"', 'format': 'json', 'resultType': 'core', 'pageSize': 10}
    body, resolution = _fetch(EPMC + '/search?' + urlencode(query), output)
    rows = _results(body)
    found = [row for row in rows if normalize_doi(row.get('doi')) == doi
             and isinstance(row.get('pmcid'), str) and re.fullmatch(r'PMC\d+', row['pmcid'])]
    if len(found) != 1 or found[0].get('isOpenAccess') != 'Y':
        raise ValueError('NO_UNIQUE_OPEN_PMC_ARTICLE')
    core = found[0]
    if candidate.get('pmcid') and candidate['pmcid'] != core['pmcid']:
        raise ValueError('CANDIDATE_IDENTITY_CONFLICT')
    url = EPMC + '/' + core['pmcid'] + '/fullTextXML'
    raw, raw_record = _fetch(url, output, is_json=False)
    parsed = read_jats(raw, expected_doi=doi, expected_pmcid=core['pmcid'])
    publication_type = _publication_type(core, raw)
    source = frame['discovery_source']
    return validate_document({
        **parsed, 'document_id': 'supplemental:' + doi, 'source': source, 'discovery_source': source,
        'source_ids': {**parsed['source_ids'], **aliases},
        'acquisition_arm': frame['acquisition_arm'], 'target': frame['target'],
        'source_record_id': doi, 'split': 'train', 'development_exposed': True, 'exposure_role': 'train_reserved',
        'public': True, 'native_split': None, 'annotation_status': 'unannotated',
        'work_group_id': 'work:doi:' + doi, 'fulltext_eligible': False,
        'access_basis': {'article_url': url, 'license_url': parsed['license_url'], 'xml_sha256': raw_record['sha256']},
        'raw_source': raw_record, 'resolution_metadata': resolution, 'discovery_metadata': frame['request'],
        'publication_type': publication_type, **metadata,
    })


def _counts(requested, accepted):
    return {'requested': requested, 'accepted': accepted, 'shortfall': requested - accepted}


def collect_supplemental(config: dict, output: Path) -> dict:
    config, output = deepcopy(config), Path(output)
    _validate_config(config)
    if (output / 'manifest.json').exists():
        loaded = load_supplemental(output)
        if loaded['config'] != config:
            raise ValueError('SUPPLEMENTAL_CONFIG_MISMATCH')
        return loaded['manifest']
    exposures = _load_exposures(config)
    frozen = _freeze(config, output, exposures)
    docs, regions, selection, permitted = [], [], [], []
    issues = frozen['issues'].copy()
    targets, arm_counts = {}, Counter()
    for target, frame in frozen['frames'].items():
        candidates = list(enumerate(frame['candidates']))
        if target == 'metadata':
            random.Random(42).shuffle(candidates)
        accepted, requested = 0, 6 if target == 'metadata' else 1
        for frame_index, candidate in candidates:
            if accepted == requested:
                break
            provenance = {'frame': f'frames/{target}.json', 'frame_index': frame_index,
                          'acquisition_arm': frame['acquisition_arm'], 'target': frame['target']}
            try:
                doc = _resolve(candidate, frame, output)
                conflicts = exposure_reasons(doc, exposures, purpose='new_acquisition') + ExclusionIndex(docs).reasons(doc)
                if conflicts:
                    issues.append({**provenance, 'code': 'EXCLUDED_OVERLAP', 'conflicts': conflicts})
                    continue
                if doc['text_license'] != 'CC-BY-4.0':
                    permitted.append({**provenance, 'document_id': doc['document_id'],
                                      'doi': doc['source_ids']['doi'], 'text_license': doc['text_license'],
                                      'status': 'permitted_but_not_detector_ready', 'document': doc})
                    issues.append({**provenance, 'code': 'LICENSE_NOT_DETECTOR_READY', 'text_license': doc['text_license']})
                    continue
                chosen = select_passages(doc, aliases=list(TARGETS) if target == 'metadata' else [target])
                _, context_issues = passage_candidates(doc)
                issues.extend({**provenance, 'document_id': doc['document_id'], **row} for row in context_issues)
                if not chosen:
                    raise ValueError('NO_ELIGIBLE_PASSAGES')
                regions.extend(chosen)
                random_selected = sum(row['selection_reason'] == 'random_whole_passage' for row in chosen)
                selection.append({**provenance, 'document_id': doc['document_id'], 'text_revision': doc['text_revision'],
                                  'doi': doc['source_ids']['doi'], 'source': doc['source'],
                                  'discovery_source': doc['discovery_source'], 'exposure_role': 'train_reserved',
                                  'random_passages': _counts(3, random_selected),
                                  'signal_passages': _counts(3, len(chosen) - random_selected)})
                if len(chosen) < 6:
                    issues.append({**provenance, 'document_id': doc['document_id'],
                                   'code': 'PASSAGE_SHORTFALL', 'requested': 6, 'accepted': len(chosen), 'shortfall': 6 - len(chosen)})
                docs.append(doc)
                accepted += 1
                arm_counts[frame['acquisition_arm']] += 1
            except (AcquisitionError, ValueError, OSError) as exc:
                issues.append({**provenance, 'code': str(exc), 'candidate': candidate})
        if target != 'metadata':
            targets[target] = _counts(1, accepted)
    write_jsonl(output / 'issues.jsonl', issues)
    write_jsonl(output / 'selection.jsonl', selection)
    write_jsonl(output / 'regions.jsonl', regions)
    write_jsonl(output / 'permitted-not-detector-ready.jsonl', permitted)
    for record in frozen['files']:
        verified_path(output, record)
    for doc in docs + [row['document'] for row in permitted]:
        for key in ('raw_source', 'resolution_metadata', 'discovery_metadata', 'paper_metadata', 'project_metadata'):
            if key in doc:
                verified_path(output, doc[key])
    verify_exposure_sources(exposures)
    _verify_config_source(config)
    if (output / 'annotation-policy.md').read_bytes() != (ROOT / POLICY).read_bytes():
        raise ValueError('ANNOTATION_POLICY_CHANGED')
    if docs:
        materialize_bundle({'documents': docs, 'role': 'train', 'heldout': {},
                            'split_digest': digest(json_bytes([doc['source_ids'] for doc in docs]))}, output / 'bundle')
    report = {
        'schema_version': SCHEMA, 'status': 'ready_for_annotation' if len(docs) == 12 else 'partial' if docs else 'failed',
        'paper_count': len(docs), 'passage_count': len(regions), 'passage_shortfall': 72 - len(regions),
        'arms': {arm: _counts(6, arm_counts[arm]) for arm in ('metadata', 'targeted')}, 'targets': targets,
        'license_exclusions': dict(Counter(row['text_license'] for row in permitted)),
        'sources': dict(Counter(doc['source'] for doc in docs)), 'issue_count': len(issues),
        'fulltext_eligible': False, 'training_ready': False, 'test_eligible': False,
        'blockers': ['annotation_required', 'version_goal_unverified'] + ([] if docs else ['no_eligible_parents']),
        'source_skew': 'metadata and fallback targets use bounded newest-first Europe PMC open-access English frames; project routes use configured associations',
        'exposures_manifest_sha256': config['exposures_manifest_sha256'],
        'annotation_policy_source': {'path': str((ROOT / POLICY).resolve()), 'sha256': digest((output / 'annotation-policy.md').read_bytes())},
        'files': [{'path': str(path.relative_to(output)), 'sha256': digest(path.read_bytes())}
                  for path in sorted(output.rglob('*')) if path.is_file()],
    }
    verify_exposure_sources(exposures)
    _verify_config_source(config)
    write_once(output / 'manifest.json', json_bytes(report))
    return report


def _verified_files(bundle, manifest):
    files = manifest.get('files')
    if not isinstance(files, list) or any(not isinstance(row, dict) or not isinstance(row.get('path'), str) for row in files):
        raise ValueError('INVALID_SUPPLEMENTAL_FILES')
    names = [row['path'] for row in files]
    if len(names) != len(set(names)):
        raise ValueError('DUPLICATE_SUPPLEMENTAL_FILE')
    required = {'config.json', 'selection-policy.json', 'annotation-policy.md', 'issues.jsonl',
                'selection.jsonl', 'regions.jsonl', 'permitted-not-detector-ready.jsonl'}
    required.update(f'frames/{target}.json' for target in ('metadata', *TARGETS))
    if manifest.get('paper_count'):
        required.update({'bundle/manifest.json', 'bundle/documents.jsonl'})
    if not required <= set(names):
        raise ValueError('MISSING_SUPPLEMENTAL_COMPANION')
    return {row['path']: verified_path(bundle, row) for row in files}


def verify_supplemental_sources(loaded: dict) -> None:
    bundle = loaded['_bundle']
    payload = (bundle / 'manifest.json').read_bytes()
    if digest(payload) != loaded['_manifest_sha256']:
        raise ValueError('SUPPLEMENTAL_MANIFEST_CHANGED')
    _verified_files(bundle, loaded['manifest'])
    verify_exposure_sources(loaded['exposures'])
    _verify_config_source(loaded['config'])
    source = loaded['manifest']['annotation_policy_source']
    if digest(Path(source['path']).read_bytes()) != source['sha256']:
        raise ValueError('ANNOTATION_POLICY_CHANGED')


def load_supplemental(bundle: Path) -> dict:
    """Verify an outer acquisition bundle, including upstream exposure sources."""
    bundle = Path(bundle).resolve()
    payload = (bundle / 'manifest.json').read_bytes()
    manifest = json.loads(payload)
    if not isinstance(manifest, dict) or manifest.get('schema_version') != SCHEMA:
        raise ValueError('INVALID_SUPPLEMENTAL_MANIFEST')
    files = _verified_files(bundle, manifest)
    policy = json.loads(files['selection-policy.json'].read_bytes())
    if policy != SELECTION_POLICY:
        raise ValueError('SUPPLEMENTAL_POLICY_MISMATCH')
    config = json.loads(files['config.json'].read_bytes())
    _validate_config(config)
    exposures = _load_exposures(config)
    if manifest.get('exposures_manifest_sha256') != config['exposures_manifest_sha256']:
        raise ValueError('EXPOSURE_MANIFEST_MISMATCH')
    docs = load_bundle(bundle / 'bundle', roles=('train',))[1] if manifest['paper_count'] else []
    regions = read_jsonl(files['regions.jsonl'])
    selection = read_jsonl(files['selection.jsonl'])
    if len(docs) != manifest['paper_count'] or len(regions) != manifest['passage_count'] or len(selection) != len(docs):
        raise ValueError('SUPPLEMENTAL_COUNT_MISMATCH')
    by_id, owned = {doc['document_id']: doc for doc in docs}, {}
    selected_ids = [row.get('document_id') for row in selection]
    if len(set(selected_ids)) != len(selected_ids) or set(selected_ids) != set(by_id):
        raise ValueError('INVALID_SUPPLEMENTAL_SELECTION')
    for row in selection:
        doc = by_id[row['document_id']]
        if (row.get('text_revision') != doc['text_revision'] or row.get('doi') != doc['source_ids']['doi']
                or any(row.get(key) != doc.get(key) for key in ('source', 'discovery_source', 'acquisition_arm', 'target'))):
            raise ValueError('INVALID_SUPPLEMENTAL_SELECTION')
    for doc in docs:
        if (doc.get('text_license') != 'CC-BY-4.0' or doc.get('public') is not True
                or doc.get('fulltext_eligible') is not False or doc.get('development_exposed') is not True):
            raise ValueError('INVALID_SUPPLEMENTAL_DOCUMENT')
    expected_regions = []
    selection_by_id = {row['document_id']: row for row in selection}
    for doc in docs:
        parsed = read_jats(verified_path(bundle, doc['raw_source']).read_bytes(),
                           expected_doi=doc['source_ids']['doi'], expected_pmcid=doc['source_ids']['pmcid'])
        if parsed['text'] != doc['text'] or parsed['paragraphs'] != doc.get('paragraphs'):
            raise ValueError('SUPPLEMENTAL_REGION_SOURCE_MISMATCH')
        if doc.get('acquisition_arm') == 'metadata' and doc.get('target') is None:
            aliases = policy['targets']
        elif doc.get('acquisition_arm') == 'targeted' and doc.get('target') in policy['targets']:
            aliases = [doc['target']]
        else:
            raise ValueError('INVALID_SUPPLEMENTAL_SELECTION')
        expected = select_passages({**parsed, 'document_id': doc['document_id'], 'text_revision': doc['text_revision']},
                                   aliases=aliases, seed=policy['seed'],
                                   random_count=policy['random_passages_per_parent'],
                                   signal_count=policy['signal_passages_per_parent'])
        expected_regions.extend(expected)
        random_selected = sum(row['selection_reason'] == 'random_whole_passage' for row in expected)
        selected = selection_by_id[doc['document_id']]
        if (selected.get('random_passages') != _counts(3, random_selected)
                or selected.get('signal_passages') != _counts(3, len(expected) - random_selected)):
            raise ValueError('SUPPLEMENTAL_SELECTION_COUNT_MISMATCH')
    if regions != expected_regions:
        raise ValueError('SUPPLEMENTAL_REGION_SELECTION_MISMATCH')
    for row in regions:
        doc = by_id.get(row.get('document_id'))
        if not doc or row.get('text_revision') != doc['text_revision']:
            raise ValueError('STALE_SUPPLEMENTAL_REGION')
        span, context = row.get('annotation_region', {}), row.get('context_span', {})
        start, end, cs, ce = span.get('start'), span.get('end'), context.get('start'), context.get('end')
        if (any(type(value) is not int for value in (start, end, cs, ce))
                or not 0 <= cs <= start < end <= ce <= len(doc['text']) or ce - cs > 6000
                or row.get('text') != doc['text'][start:end] or row.get('context_text') != doc['text'][cs:ce]
                or row.get('whole_passage_audit') is not True or row.get('region_kind') != 'sentence'):
            raise ValueError('INVALID_SUPPLEMENTAL_REGION')
        prior = owned.setdefault(doc['document_id'], [])
        if any(start < old_end and old_start < end for old_start, old_end in prior):
            raise ValueError('OVERLAPPING_SUPPLEMENTAL_REGIONS')
        prior.append((start, end))
    loaded = {'manifest': manifest, 'config': config, 'documents': docs, 'regions': regions,
              'selection': selection, 'issues': read_jsonl(files['issues.jsonl']), 'exposures': exposures,
              '_bundle': bundle, '_manifest_sha256': digest(payload)}
    verify_supplemental_sources(loaded)
    return loaded
