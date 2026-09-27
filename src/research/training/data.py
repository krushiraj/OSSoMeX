"""Immutable, detector-only training inputs from validated review snapshots."""

import json
from pathlib import Path

from ..annotations.snapshots import read_review_snapshot
from ..data.bundles import load_bundle
from ..data.ecosystems import ExclusionIndex
from ..data.manifest import digest, json_bytes, read_jsonl, verified_path, write_jsonl, write_once
from ..data.splits import check_training_manifest, identifiers


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
    if output.exists():
        raise FileExistsError(output)
    documents, items, sources, copies, forbidden = [], [], [], {}, []
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
    selected = select_supervision(documents, items)
    if config.get('expected_summary') and selected['summary'] != config['expected_summary']:
        raise ValueError('unexpected detector support')
    for name, payload in copies.items():
        write_once(output / name, payload)
    write_once(output / 'config.json', json_bytes(config))
    write_once(output / 'forbidden.json', json_bytes(heldout))
    write_jsonl(output / 'documents.jsonl', documents)
    write_jsonl(output / 'items.jsonl', selected['items'])
    write_jsonl(output / 'exclusions.jsonl', selected['excluded'])
    result = {'schema_version': 'detector-data-1', 'role': 'train', 'quality': 'provisional',
              'full_benchmark_ready': False, 'summary': selected['summary'], 'sources': sources,
              'forbidden_documents_checked': len(forbidden),
              'files': [{'path': str(p.relative_to(output)), 'sha256': digest(p.read_bytes())}
                        for p in sorted(output.rglob('*')) if p.is_file()]}
    write_once(output / 'manifest.json', json_bytes(result))
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
    documents, items = read_jsonl(bundle / 'documents.jsonl'), read_jsonl(bundle / 'items.jsonl')
    if check_training_manifest({'documents': documents}, json.loads((bundle / 'forbidden.json').read_bytes())):
        raise ValueError('heldout training overlap')
    selected = select_supervision(documents, items)
    if selected['summary'] != manifest['summary']:
        raise ValueError('training support mismatch')
    return manifest, documents, selected['items']
