"""Stratified audit of CZI name->package links. Verdicts are provisional until reviewed."""

import json
import math
import random
from pathlib import Path

from ..data.acquire import fetch_public
from ..data.manifest import json_bytes, read_jsonl, write_jsonl, write_once

VERDICTS = {'correct', 'wrong', 'unsure'}


def sample(links, top=200, random_n=200, seed=42):
    ordered = sorted(links, key=lambda r: (-r['mentions_count'], r['czi_id']))
    head = ordered[:top]
    rest = sorted(ordered[top:], key=lambda r: r['czi_id'])
    tail = random.Random(seed).sample(rest, min(random_n, len(rest)))
    return [{**r, 'stratum': 'top'} for r in head] + [{**r, 'stratum': 'random'} for r in tail]


def wilson(k, n, z=1.96):
    if n == 0:
        return None, None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def _summary(rows):
    judged = [r for r in rows if r['verdict'] != 'unsure']
    wrong = [r for r in judged if r['verdict'] == 'wrong']
    low, high = wilson(len(wrong), len(judged))
    weight = sum(r['mentions_count'] for r in judged)
    return {'tasks': len(rows), 'judged': len(judged), 'wrong': len(wrong),
            'unsure': len(rows) - len(judged), 'wrong_rate': len(wrong) / len(judged) if judged else None,
            'wrong_rate_95ci': [low, high],
            'mention_weighted_wrong_rate': sum(r['mentions_count'] for r in wrong) / weight if weight else None}


def report(tasks, labels):
    by_id = {t['task_id']: t for t in tasks}
    rows = []
    for label in labels:
        if label['task_id'] not in by_id:
            raise ValueError(f"UNKNOWN_AUDIT_TASK: {label['task_id']}")
        if label['verdict'] not in VERDICTS:
            raise ValueError(f"INVALID_VERDICT: {label['verdict']}")
        rows.append({**by_id[label['task_id']], 'verdict': label['verdict']})
    return {'labelled': len(rows), 'unlabelled': len(tasks) - len(rows),
            'strata': {s: _summary([r for r in rows if r['stratum'] == s]) for s in sorted({t['stratum'] for t in tasks})},
            'by_flag': {'flagged': _summary([r for r in rows if r['flags']]),
                        'unflagged': _summary([r for r in rows if not r['flags']])}}


def _paper_evidence(link, raw, policy, papers):
    mentions_url = link['project_url'] + '/mentions?per_page=' + str(papers)
    name = f"{link['czi_id']}-mentions.json"
    mentions = json.loads(_fetch(mentions_url, raw / name, policy))
    evidence = []
    for i, mention in enumerate(mentions[:papers]):
        paper = json.loads(_fetch(mention['paper_url'], raw / f"{link['czi_id']}-paper-{i + 1}.json", policy))
        evidence.append({'doi': paper.get('doi'), 'openalex_id': paper.get('openalex_id'),
                         'title': paper.get('title'), 'publication_date': (paper.get('publication_date') or '')[:10]})
    return evidence


def _fetch(url, destination, policy):
    fetch_public(url, destination, policy)
    return destination.read_bytes()


def sample_run(run: Path, top=200, random_n=200, seed=42, papers=5, policy=None) -> dict:
    run = Path(run)
    links = read_jsonl(run / 'czi_links.jsonl')
    entities = {e['entity_id']: e for e in read_jsonl(run / 'entities.jsonl')}
    policy = {'format': 'json', **(policy or {})}
    tasks = []
    for row in sample(links, top, random_n, seed):
        entity = entities[row['entity_id']]
        tasks.append({
            'task_id': f"{row['stratum']}-{row['czi_id']}", 'stratum': row['stratum'], 'czi_id': row['czi_id'],
            'czi_name': row['czi_name'], 'entity_id': row['entity_id'], 'mentions_count': row['mentions_count'],
            'flags': row['flags'], 'package_ids': entity['package_ids'], 'description': entity['description'],
            'repo_keys': entity['repo_keys'],
            'same_name_entities': [{'entity_id': o, 'names': entities[o]['names'][:3],
                                    'description': entities[o]['description']} for o in row['same_name_entities'][:5]],
            'papers': _paper_evidence(row, run / 'raw/audit', policy, papers)})
    write_jsonl(run / 'audit/tasks.jsonl', tasks)
    summary = {'status': 'success', 'tasks': len(tasks), 'top': top, 'random': random_n, 'seed': seed,
               'papers_per_task': papers}
    write_once(run / 'audit/sample.json', json_bytes(summary))
    return summary


def report_run(run: Path, labels_path: Path) -> dict:
    run = Path(run)
    result = {'status': 'success', 'labels': str(labels_path),
              **report(read_jsonl(run / 'audit/tasks.jsonl'), read_jsonl(Path(labels_path)))}
    (run / 'audit/report.json').write_bytes(json_bytes(result))
    return result
