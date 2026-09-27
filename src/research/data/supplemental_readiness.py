"""Report supplemental supervision and fitting gates without fitting a model."""

from collections import Counter
import json
from pathlib import Path

from ..annotations.review_workflow import project_review
from ..annotations.snapshots import read_review_snapshot
from ..annotations.supplemental import frozen_tasks
from ..contracts import FIELDS
from ..training.data import select_supervision
from ..training.features import build_token_features
from .exposure import exposure_reasons
from .manifest import digest, json_bytes, verified_path, write_once
from .supplemental import load_supplemental, verify_supplemental_sources


BASE_MODEL = 'allenai/scibert_scivocab_cased'
BASE_REVISION = 'ddf0be025f8e432a1870e34811997ba6725bf04a'
REVIEW_KINDS = ('human_reviewed', 'agent_provisional')


def _load_tokenizer():
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    base = snapshot_download(BASE_MODEL, revision=BASE_REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(base, do_lower_case=False, use_fast=True,
                                               local_files_only=True, trust_remote_code=False)
    if not tokenizer.is_fast or tokenizer.do_lower_case:
        raise ValueError('cased fast tokenizer required')
    return tokenizer


def _counts():
    return {'passages': 0, 'by_label': {label: dict.fromkeys(REVIEW_KINDS, 0)
                                      for label in ('SOFTWARE', 'VERSION')}}


def _support(documents, items, records):
    parents = {d['document_id']: d for d in documents}
    result = {**_counts(), 'by_paper': {}, 'by_arm': {},
              'by_review_kind': dict.fromkeys(REVIEW_KINDS, 0),
              'fields': {field: {'human_reviewed': 0, 'agent_provisional': 0,
                                'unknown': 0, 'known_absent': 0} for field in FIELDS},
              'coverage': [], 'remaining_human_workload': {'needs_decisions': 0, 'proposals_to_confirm': 0}}
    histories = {}
    for record in records:
        histories.setdefault(record['task_id'], []).append({
            'payload': json.loads(record['payload']), 'result': json.loads(record['result']),
            'recorded_at': record['recorded_at']})
    versions, version_papers, projections = set(), set(), []
    for item in items:
        task, annotation = item['task'], item['annotation']
        projection = project_review(item, histories.get(task['task_id'], []))
        projections.append(projection)
        paper = result['by_paper'].setdefault(task['document_id'], _counts())
        arm = result['by_arm'].setdefault(parents[task['document_id']]['acquisition_arm'], _counts())
        for counts in (result, paper, arm):
            counts['passages'] += 1
        kind = 'human_reviewed' if projection['workflow_status'] == 'approved' else 'agent_provisional'
        result['by_review_kind'][kind] += 1
        for name in result['remaining_human_workload']:
            result['remaining_human_workload'][name] += projection[name]
        seen = set()
        for occurrence in annotation['occurrences']:
            states = projection['fields'][occurrence['mention_id']]
            for field in FIELDS:
                known = occurrence['known'][field]
                review_kind = 'human_reviewed' if states[field]['state'] == 'confirmed' else 'agent_provisional'
                result['fields'][field][review_kind if known else 'unknown'] += 1
                if field == 'versions' and known and occurrence['version_status'] == 'absent':
                    result['fields'][field]['known_absent'] += 1
            spans = []
            if occurrence['known']['software']:
                spans.append(('SOFTWARE', occurrence['name_span'], 'software'))
            if occurrence['known']['versions']:
                spans.extend(('VERSION', edge['span'], 'versions') for edge in occurrence['version_links'])
            for label, span, field in spans:
                kind = 'human_reviewed' if states[field]['state'] == 'confirmed' else 'agent_provisional'
                key = (label, span['start'], span['end'], kind)
                if key in seen:
                    continue
                seen.add(key)
                for counts in (result, paper, arm):
                    counts['by_label'][label][kind] += 1
                if label == 'VERSION':
                    versions.add((task['document_id'], span['start'], span['end']))
                    version_papers.add(task['document_id'])
        for region in annotation['covered_regions']:
            for field, label in (('software', 'SOFTWARE'), ('versions', 'VERSION')):
                relevant = [o for o in annotation['occurrences']
                            if region['start'] < o['name_span']['end'] and o['name_span']['start'] < region['end']]
                known = region['fields'].get(field) is True and all(o['known'][field] for o in relevant)
                human = (known and projection['name_audit'] == 'confirmed' and not projection['source_issues']
                         and (field == 'software' or projection['workflow_status'] == 'approved'
                              and all(projection['fields'][o['mention_id']][field]['state'] == 'confirmed'
                                      for o in relevant)))
                result['coverage'].append({'task_id': task['task_id'], 'label': label,
                    'start': region['start'], 'end': region['end'], 'known': known,
                    'review_kind': ('human_reviewed' if human else 'agent_provisional') if known else None})
    return result, {'explicit_spans': len(versions), 'papers': len(version_papers),
                    'required_spans': 20, 'required_papers': 6,
                    'span_shortfall': max(0, 20 - len(versions)), 'paper_shortfall': max(0, 6 - len(version_papers)),
                    'met': len(versions) >= 20 and len(version_papers) >= 6}, projections


def _alignment(documents, items):
    identity = {'model_id': BASE_MODEL, 'revision': BASE_REVISION, 'local_files_only': True,
                'max_length': 482, 'overlap': 64}
    try:
        tokenizer = _load_tokenizer()
    except (OSError, ImportError) as exc:
        return {**identity, 'status': 'unavailable', 'reason': 'tokenizer_cache_missing',
                'detail': str(exc), 'documents': [], 'exclusions': []}
    rows, exclusions = [], []
    for document in documents:
        own = [i['annotation'] for i in items if i['task']['document_id'] == document['document_id']]
        features = build_token_features(document, [o for a in own for o in a['occurrences']],
                                        [r for a in own for r in a['covered_regions']], tokenizer, identity)
        rows.append({'document_id': document['document_id'], 'token_count': features['token_count'],
                     'active_tokens': sum(sum(w['active_mask']) for w in features['windows'])})
        exclusions.extend({'document_id': document['document_id'], **row} for row in features['exclusions'])
    return {**identity, 'status': 'checked', 'documents': rows, 'exclusions': exclusions}


def assess_readiness(bundle: Path, snapshot: Path | None, output: Path) -> dict:
    bundle, output = Path(bundle), Path(output)
    if output.exists():
        raise FileExistsError(output)
    loaded = load_supplemental(bundle)
    documents = loaded['documents']
    policy, tasks = frozen_tasks(loaded)
    expected = {t['task_id']: t for t in tasks}
    annotation_blockers = []
    if not tasks:
        annotation_blockers.append('no_eligible_parents')
    if any(d.get('text_license') != 'CC-BY-4.0' for d in documents):
        annotation_blockers.append('text_license_unsupported')
    overlaps = [{'parent_document_id': d['document_id'], **reason} for d in documents
                for reason in exposure_reasons(d, loaded['exposures'], purpose='training')]
    if overlaps:
        annotation_blockers.append('forbidden_exposure_overlap')
    detector_blockers = list(annotation_blockers)
    items, records, snapshot_provenance = [], [], None
    if snapshot is None:
        detector_blockers.append('snapshot_missing')
    else:
        snapshot = Path(snapshot)
        payload = (snapshot / 'manifest.json').read_bytes()
        manifest, data = read_review_snapshot(snapshot)
        if manifest.get('role') != 'train' or manifest.get('policy') != policy:
            raise ValueError('SUPPLEMENTAL_SNAPSHOT_POLICY_OR_ROLE_MISMATCH')
        if manifest != json.loads(payload):
            raise ValueError('SNAPSHOT_CHANGED_DURING_READ')
        items, records = data['items.jsonl'], data['decision-records.jsonl']
        for item in items:
            task = item['task']
            frozen = expected.get(task['task_id'])
            if frozen is None or any(task.get(key) != value for key, value in frozen.items()):
                raise ValueError('SUPPLEMENTAL_SNAPSHOT_TASK_MISMATCH')
        if {i['task']['task_id'] for i in items} != set(expected):
            detector_blockers.append('snapshot_tasks_missing')
        snapshot_provenance = {'path': str(snapshot.resolve()), 'manifest_sha256': digest(payload),
                               'files': manifest['files']}
    selected = None
    if items and not annotation_blockers:
        try:
            selected = select_supervision(documents, items)
        except ValueError as exc:
            if str(exc) != 'no detector training data':
                raise
            detector_blockers.append('no_detector_supervision')
    elif snapshot is not None:
        detector_blockers.append('no_detector_supervision')
    support, version_goal, projections = _support(documents, items, records)
    if any(p['source_issues'] for p in projections):
        detector_blockers.append('review_source_issue')
    sources = sorted({d['source'] for d in documents})
    fit_blockers = list(detector_blockers)
    if sources != ['ecosystems']:
        fit_blockers.append('sampler_unsupported')
    alignment = {'status': 'not_checked', 'documents': [], 'exclusions': []}
    if selected is not None:
        alignment = _alignment(documents, selected['items'])
        if alignment['status'] == 'unavailable':
            fit_blockers.append('tokenizer_cache_missing')
        elif not any(r['active_tokens'] for r in alignment['documents']):
            fit_blockers.append('no_aligned_supervision')
    goal_blockers = [] if version_goal['met'] else ['version_goal_shortfall']
    report = {'schema_version': 'supplemental-readiness-1', 'status': 'reported',
        'annotation_ready': not annotation_blockers, 'detector_bundle_ready': not detector_blockers,
        'fit_ready': not fit_blockers, 'full_benchmark_ready': False, 'fulltext_eligible': False,
        'blockers': sorted(set(fit_blockers)), 'goal_shortfalls': goal_blockers,
        'blocker_groups': {'annotation': annotation_blockers, 'detector_bundle': detector_blockers,
                           'fit': fit_blockers, 'version_goal': goal_blockers},
        'sources': sources, 'source_counts': dict(Counter(d['source'] for d in documents)),
        'support': support, 'version_goal': version_goal, 'tokenizer_alignment': alignment,
        'exposure_exclusions': overlaps, 'supervision': None if selected is None else {
            'summary': selected['summary'], 'excluded': selected['excluded']},
        'provenance': {'supplemental_bundle': str(bundle.resolve()),
            'supplemental_manifest_sha256': loaded['_manifest_sha256'],
            'exposures_manifest_sha256': loaded['config']['exposures_manifest_sha256'],
            'policy': policy, 'snapshot': snapshot_provenance}}
    verify_supplemental_sources(loaded)
    if snapshot_provenance:
        if digest((snapshot / 'manifest.json').read_bytes()) != snapshot_provenance['manifest_sha256']:
            raise ValueError('SNAPSHOT_CHANGED_DURING_READ')
        for record in snapshot_provenance['files']:
            verified_path(snapshot, record)
    write_once(output / 'report.json', json_bytes(report))
    write_once(output / 'manifest.json', json_bytes({'status': 'reported',
        'files': [{'path': 'report.json', 'sha256': digest((output / 'report.json').read_bytes())}]}))
    return report
