"""Small, source-bound training tasks; selection cues never become labels."""

from copy import deepcopy
import json
import os
from pathlib import Path
import random
import re

from ..annotations.policies import ROOT, load_policy, snapshot_prompts
from ..annotations.tasks import check_authorization, make_region_task
from ..contracts import check_span, validate_document
from ..training.data import load_training_data
from .artifacts import atomic_write_jsonl_new, atomic_write_new
from .exposure import exposure_reasons, load_exposures, verify_exposure_sources
from .manifest import digest, json_bytes
from .supplemental_passages import passage_candidates


POLICY = ROOT / 'annotations/scibert-v2/policy-2.1-d17.md'


def _overlap(first, second):
    return first['start'] < second['end'] and second['start'] < first['end']


def select_regions(documents: list[dict], items: list[dict], config: dict) -> list[dict]:
    limits = {'max_passages': (1, 72), 'random_per_parent': (0, 3), 'signal_per_parent': (0, 3)}
    if type(config.get('seed')) is not int or config['seed'] != 42:
        raise ValueError('invalid topup seed')
    for name, (low, high) in limits.items():
        if type(config.get(name)) is not int or not low <= config[name] <= high:
            raise ValueError('invalid topup limit: ' + name)
    cues = config.get('cues')
    if not isinstance(cues, list) or not cues or any(not isinstance(c, str) or not c.strip() for c in cues):
        raise ValueError('nonempty topup cues required')
    parents, owned, task_ids = {}, {}, set()
    for original in documents:
        doc = validate_document(original)
        if (doc['document_id'] in parents or doc.get('split') != 'train'
                or doc.get('public') is not True or doc.get('text_license') != 'CC-BY-4.0'
                or not doc.get('access_basis')):
            raise ValueError('invalid topup training parent')
        parents[doc['document_id']] = doc
    for item in items:
        task = item['task']
        doc = parents.get(task['document_id'])
        if doc is None or task['text_revision'] != doc['text_revision']:
            raise ValueError('stale ownership revision')
        region = task['annotation_region']
        check_span(region, task, 'annotation_region', len(doc['text']))
        prior = owned.setdefault(doc['document_id'], [])
        if task['task_id'] in task_ids or any(_overlap(region, previous) for previous in prior):
            raise ValueError('duplicate or overlapping ownership')
        task_ids.add(task['task_id'])
        prior.append(region)
    selected_parents = config.get('document_ids', list(parents))
    if (not isinstance(selected_parents, list) or not selected_parents
            or any(not isinstance(name, str) or name not in parents for name in selected_parents)
            or len(set(selected_parents)) != len(selected_parents)):
        raise ValueError('invalid topup parent selection')
    rng, result = random.Random(config['seed']), []
    for doc_id, doc in sorted(parents.items()):
        if doc_id not in selected_parents:
            continue
        parsed = deepcopy(doc)
        if not parsed.get('paragraphs'):
            starts = [0, *[m.end() for m in re.finditer(r'\n\s*\n', doc['text'])]]
            parsed['paragraphs'] = [{'start': start, 'end': end, 'kind': 'p'}
                                    for start, end in zip(starts, starts[1:] + [len(doc['text'])]) if start < end]
        candidates, _ = passage_candidates(parsed)
        eligible = [row for row in candidates if not any(
            _overlap(row['annotation_region'], region) for region in owned.get(doc_id, []))]
        signals = [row for row in eligible if any(c.casefold() in row['text'].casefold() for c in cues)]
        chosen = signals[:config['signal_per_parent']]
        selected = {r['annotation_region']['start'] for r in chosen}
        controls = [row for row in eligible if row['annotation_region']['start'] not in selected]
        random_rows = rng.sample(controls, min(config['random_per_parent'], len(controls)))
        for row, reason in [*((r, 'training_gap_candidate') for r in chosen),
                            *((r, 'random_whole_passage') for r in random_rows)]:
            if len(result) == config['max_passages']:
                return result
            result.append({**row, 'selection_reason': reason,
                           'whole_passage_audit': reason == 'random_whole_passage'})
    return result


def _pinned(spec: dict) -> Path:
    path = Path(spec['path'])
    if path.name != 'manifest.json' or digest(path.read_bytes()) != spec.get('sha256'):
        raise ValueError('changed topup source manifest')
    return path


def prepare_topup(config: dict, output: Path) -> dict:
    output = Path(output)
    if os.path.lexists(output):
        raise FileExistsError(output)
    source = _pinned(config['training_data'])
    _, documents, items = load_training_data(source.parent)
    exposure_path = _pinned(config['exposures'])
    exposures = load_exposures(exposure_path.parent)
    for doc in documents:
        if exposure_reasons(doc, exposures, purpose='training'):
            raise ValueError('topup training exposure overlap')
        if not any(row['document_id'] == doc['document_id'] and row['role'] == 'train_reserved'
                   and row.get('text_revision') == doc['text_revision'] for row in exposures['documents']):
            raise ValueError('missing topup training reservation')
    regions = select_regions(documents, items, config['selection'])
    if not regions:
        raise ValueError('no eligible unowned training regions')
    policy = load_policy(POLICY)
    if config.get('policy_sha256') != policy['policy_hash']:
        raise ValueError('changed topup annotation policy')
    approvals = json.loads((ROOT / 'approvals.json').read_bytes())
    parents = {row['document_id']: row for row in documents}
    tasks = []
    for region in regions:
        task = make_region_task(parents[region['document_id']], policy,
                                region['annotation_region'], region['context_span'], region_kind='sentence')
        check_authorization(task, approvals)
        tasks.append({**task, 'whole_passage_audit': region['whole_passage_audit'],
                      'topup_selection': {'reason': region['selection_reason'], 'seed': config['selection']['seed']}})
    verify_exposure_sources(exposures)
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_new(output / 'config.json', json_bytes(config))
    atomic_write_new(output / 'policy.md', POLICY.read_bytes())
    atomic_write_jsonl_new(output / 'regions.jsonl', regions)
    atomic_write_jsonl_new(output / 'tasks.jsonl', tasks)
    prompts = snapshot_prompts(output, policy)
    result = {'schema_version': 'training-topup-1', 'role': 'train', 'status': 'ready_for_annotation',
              'task_count': len(tasks), 'policy': policy, 'source_training': config['training_data'],
              'exposures': config['exposures'],
              'policy_source': {'path': 'policy.md', 'sha256': policy['policy_hash'], 'source_path': str(POLICY)},
              'prompt_sources': prompts,
              'whole_passage_audit_ids': [t['task_id'] for t in tasks if t['whole_passage_audit']],
              'files': [{'path': str(p.relative_to(output)), 'sha256': digest(p.read_bytes())}
                        for p in sorted(output.rglob('*')) if p.is_file()]}
    _pinned(config['training_data'])
    load_training_data(source.parent)
    _pinned(config['exposures'])
    verify_exposure_sources(exposures)
    if load_policy(POLICY)['policy_hash'] != policy['policy_hash']:
        raise ValueError('changed topup annotation policy')
    atomic_write_new(output / 'manifest.json', json_bytes(result))
    return result
