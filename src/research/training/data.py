"""Immutable, detector-only training inputs from validated review snapshots."""

import json
import os
from pathlib import Path
import re

from ..annotations.snapshots import read_review_snapshot
from ..data.artifacts import atomic_write_jsonl_new, atomic_write_new
from ..data.bundles import load_bundle
from ..data.ecosystems import ExclusionIndex
from ..data.exposure import exposure_reasons, load_exposures, verify_exposure_sources
from ..data.manifest import digest, json_bytes, read_jsonl, verified_path, write_jsonl, write_once
from ..data.splits import check_training_manifest, identifiers


def _exposure_specs(config):
    specs = config.get('exposure_bundles', [])
    if not isinstance(specs, list):
        raise ValueError('invalid exposure bundles')
    normalized, seen = [], set()
    for spec in specs:
        if (not isinstance(spec, dict) or set(spec) != {'path', 'sha256'}
                or not isinstance(spec['path'], str) or not spec['path'].strip()
                or not isinstance(spec['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', spec['sha256'])):
            raise ValueError('invalid exposure bundle spec')
        path = Path(spec['path']).resolve()
        if path.name != 'manifest.json':
            raise ValueError('exposure manifest path required')
        if path in seen:
            raise ValueError('duplicate exposure bundle')
        seen.add(path)
        normalized.append({'path': str(path), 'sha256': spec['sha256']})
    return normalized


def _load_pinned_exposure(spec):
    try:
        loaded = load_exposures(Path(spec['path']).parent)
    except OSError as exc:
        raise ValueError(f'missing exposure bundle: {spec["path"]}') from exc
    if loaded['_manifest_sha256'] != spec['sha256']:
        raise ValueError(f'changed exposure manifest: {spec["path"]}')
    return loaded


def _check_training_exposures(documents, exposures):
    for document in documents:
        for loaded in exposures:
            if exposure_reasons(document, loaded, purpose='training'):
                raise ValueError(f'training exposure overlap: {document["document_id"]}')


def _load_training_exposures(bundle, manifest, config):
    specs = _exposure_specs(config)
    records = manifest.get('exposure_bundles', [])
    sources = [row for row in manifest.get('sources', []) if row['kind'] == 'exposure_bundles']
    if (not isinstance(records, list) or len(records) != len(specs)
            or sources != [{'kind': 'exposure_bundles', **spec} for spec in specs]
            or ('exposure_bundles' in config) != ('exposure_bundles' in manifest)):
        raise ValueError('training exposure provenance mismatch')
    loaded_bundles = []
    files = {row['path']: row for row in manifest['files']}
    for ordinal, (spec, record) in enumerate(zip(specs, records)):
        prefix = f'sources/exposure_bundles-{ordinal}'
        frozen_path = f'{prefix}/manifest.json'
        if record != {**spec, 'frozen_path': frozen_path}:
            raise ValueError('training exposure provenance mismatch')
        loaded = _load_pinned_exposure(spec)
        evidence = [{'path': 'manifest.json', 'sha256': spec['sha256']}, *loaded['manifest']['files']]
        for row in evidence:
            frozen = {'path': f'{prefix}/{row["path"]}', 'sha256': row['sha256']}
            if files.get(frozen['path']) != frozen:
                raise ValueError('training exposure evidence mismatch')
            verified_path(bundle, frozen)
        loaded_bundles.append(loaded)
    return loaded_bundles


def select_supervision(documents, items):
    parents = {d['document_id']: d for d in documents}
    if len(parents) != len(documents) or check_training_manifest({'documents': documents}, {}):
        raise ValueError('invalid training parents')
    selected, excluded, seen, owned = [], [], set(), {}
    software, versions = set(), set()
    for item in items:
        task, annotation = item['task'], item['annotation']
        parent = parents[task['document_id']]
        context, region = task['context_span'], task['annotation_region']
        if (parent['split'] != 'train' or parent.get('public') is not True
                or parent.get('text_license') != 'CC-BY-4.0' or not parent.get('access_basis')
                or any(task[k] != parent[k] for k in ('text_revision', 'source', 'split', 'text_license', 'access_basis'))
                or task['offset_base'] != context['start']
                or not 0 <= context['start'] <= region['start'] < region['end'] <= context['end'] <= len(parent['text'])
                or parent['text'][context['start']:context['end']] != task['text']):
            raise ValueError('invalid training task provenance or source')
        if task['task_id'] in seen:
            raise ValueError('duplicate task')
        seen.add(task['task_id'])
        coverage = annotation['covered_regions']
        occurrences = annotation['occurrences']
        positive = any(o.get('known', {}).get('software') is True or
                       o.get('known', {}).get('versions') is True and o.get('version_links') for o in occurrences)
        if not positive and not any(any(r['fields'].get(f) is True for f in ('software', 'versions')) for r in coverage):
            excluded.append({'task_id': task['task_id'], 'reason': 'no_detector_supervision'})
            continue
        prior = owned.setdefault(task['document_id'], [])
        if any(s < region['end'] and region['start'] < e for s, e in prior):
            raise ValueError('overlapping owned regions')
        prior.append((region['start'], region['end']))
        for r in coverage:
            if not region['start'] <= r['start'] < r['end'] <= region['end']:
                raise ValueError('coverage outside ownership')
        for o in occurrences:
            span = o['name_span']
            if not region['start'] <= span['start'] < span['end'] <= region['end']:
                raise ValueError('name outside ownership')
            if o['known'].get('software') is True:
                software.add((task['document_id'], span['start'], span['end']))
            if o['known'].get('versions') is True:
                for edge in o['version_links']:
                    vs = edge['span']
                    if not context['start'] <= vs['start'] < vs['end'] <= context['end']:
                        raise ValueError('version outside context')
                    versions.add((task['document_id'], vs['start'], vs['end']))
        selected.append(item)
    if not selected:
        raise ValueError('no detector training data')
    return {'items': selected, 'excluded': excluded,
            'summary': {'documents': len({i['task']['document_id'] for i in selected}),
                        'passages': len(selected), 'software_spans': len(software), 'version_spans': len(versions),
                        'human_reviewed_passages': sum(i['status'] == 'reviewed' for i in selected),
                        'agent_provisional_passages': sum(i['status'] != 'reviewed' for i in selected)}}


def prepare_data(config, output):
    output = Path(output)
    atomic_publication = 'exposure_bundles' in config
    if (os.path.lexists(output) if atomic_publication else output.exists()):
        raise FileExistsError(output)
    documents, items, sources, copies, forbidden = [], [], [], {}, []
    exposures, exposure_provenance = [], []
    exposure_specs = _exposure_specs(config)
    effective_config = {**config, 'exposure_bundles': exposure_specs} if 'exposure_bundles' in config else config
    for ordinal, spec in enumerate(exposure_specs):
        loaded = _load_pinned_exposure(spec)
        exposures.append(loaded)
        sources.append({'kind': 'exposure_bundles', **spec})
        prefix = f'sources/exposure_bundles-{ordinal}'
        exposure_provenance.append({**spec, 'frozen_path': f'{prefix}/manifest.json'})
        evidence = [{'path': 'manifest.json', 'sha256': spec['sha256']}, *loaded['manifest']['files']]
        for row in evidence:
            payload = verified_path(loaded['_bundle'], row).read_bytes()
            if digest(payload) != row['sha256']:
                raise ValueError('changed exposure evidence')
            copies[f'{prefix}/{row["path"]}'] = payload
    for kind in ('parents', 'snapshots', 'companions', 'attribution', 'exclusions'):
        for ordinal, spec in enumerate(config[kind]):
            path = Path(spec['path']).resolve()
            if digest(path.read_bytes()) != spec['sha256']:
                raise ValueError(f'changed source: {path}')
            sources.append({'kind': kind, 'path': str(path), 'sha256': spec['sha256']})
            if kind == 'exclusions':
                forbidden.extend(read_jsonl(path))
                continue
            target = f'sources/{kind}-{ordinal}/{path.name}'
            copies[target] = path.read_bytes()
            if kind in ('parents', 'snapshots', 'companions'):
                manifest = json.loads(path.read_bytes())
                for row in manifest['files']:
                    artifact = verified_path(path.parent, row)
                    copies[f'sources/{kind}-{ordinal}/{row["path"]}'] = artifact.read_bytes()
            if kind == 'parents':
                _, rows = load_bundle(path.parent, roles=('train',))
                documents.extend(rows)
            elif kind == 'snapshots':
                manifest, data = read_review_snapshot(path.parent)
                if manifest['role'] != 'train':
                    raise ValueError('training snapshot required')
                items.extend({**i, 'source_snapshot_sha256': spec['sha256']} for i in data['items.jsonl'])
    index = ExclusionIndex(forbidden)
    heldout = {'work_group_ids': sorted({d['work_group_id'] for d in forbidden if d.get('work_group_id')}),
               'text_revisions': sorted({d['text_revision'] for d in forbidden if d.get('text_revision')}),
               'source_ids': sorted(set().union(*(identifiers(d) for d in forbidden)))}
    if check_training_manifest({'documents': documents}, heldout):
        raise ValueError('heldout training overlap')
    for n, doc in enumerate(documents):
        if index.reasons(doc) or ExclusionIndex(documents[:n]).reasons(doc):
            raise ValueError('training identity or text overlap')
    _check_training_exposures(documents, exposures)
    selected = select_supervision(documents, items)
    if config.get('expected_summary') and selected['summary'] != config['expected_summary']:
        raise ValueError('unexpected detector support')
    for loaded in exposures:
        verify_exposure_sources(loaded)
    if atomic_publication:
        output.mkdir(parents=True, exist_ok=False)
    write_artifact = atomic_write_new if atomic_publication else write_once
    write_rows = atomic_write_jsonl_new if atomic_publication else write_jsonl
    for name, payload in copies.items():
        write_artifact(output / name, payload)
    write_artifact(output / 'config.json', json_bytes(effective_config))
    write_artifact(output / 'forbidden.json', json_bytes(heldout))
    write_rows(output / 'documents.jsonl', documents)
    write_rows(output / 'items.jsonl', selected['items'])
    write_rows(output / 'exclusions.jsonl', selected['excluded'])
    result = {'schema_version': 'detector-data-1', 'role': 'train', 'quality': 'provisional',
              'full_benchmark_ready': False, 'summary': selected['summary'], 'sources': sources,
              'forbidden_documents_checked': len(forbidden),
              'files': [{'path': str(p.relative_to(output)), 'sha256': digest(p.read_bytes())}
                        for p in sorted(output.rglob('*')) if p.is_file()]}
    if 'exposure_bundles' in config:
        result['exposure_bundles'] = exposure_provenance
    for loaded in exposures:
        verify_exposure_sources(loaded)
    write_artifact(output / 'manifest.json', json_bytes(result))
    return result


def load_training_data(bundle):
    bundle = Path(bundle)
    manifest = json.loads((bundle / 'manifest.json').read_bytes())
    if manifest.get('schema_version') != 'detector-data-1' or manifest.get('role') != 'train':
        raise ValueError('invalid detector training bundle')
    paths = [r['path'] for r in manifest['files']]
    if len(paths) != len(set(paths)) or not {'documents.jsonl', 'items.jsonl', 'forbidden.json', 'config.json'} <= set(paths):
        raise ValueError('incomplete detector training bundle')
    for row in manifest['files']:
        verified_path(bundle, row)
    config = json.loads((bundle / 'config.json').read_bytes())
    exposures = _load_training_exposures(bundle, manifest, config)
    documents, items = read_jsonl(bundle / 'documents.jsonl'), read_jsonl(bundle / 'items.jsonl')
    if check_training_manifest({'documents': documents}, json.loads((bundle / 'forbidden.json').read_bytes())):
        raise ValueError('heldout training overlap')
    _check_training_exposures(documents, exposures)
    selected = select_supervision(documents, items)
    if selected['summary'] != manifest['summary']:
        raise ValueError('training support mismatch')
    for loaded in exposures:
        verify_exposure_sources(loaded)
    return manifest, documents, selected['items']
