"""Seeded, paper-disjoint evaluation selection, independent of model outputs."""

from copy import deepcopy
import json
import os
from pathlib import Path
import random

from ..comparison.contracts import validate_input
from .ecosystems import ExclusionIndex
from .exposure import _identity_keys, exposure_reasons, load_exposures, verify_exposure_sources
from .manifest import digest, json_bytes, read_jsonl, verified_path, write_once


def select_evaluation(candidates, exposures, *, seed, evaluation_count, development_count):
    for key, value in [('seed', seed), ('evaluation_count', evaluation_count),
                       ('development_count', development_count)]:
        if type(value) is not int or value < 0:
            raise ValueError('invalid sampling option: ' + key)
    rows = sorted((validate_input(row) for row in candidates), key=lambda row: row['document_id'])
    if len({row['document_id'] for row in rows}) != len(rows):
        raise ValueError('duplicate evaluation document ID')
    parents = list(range(len(rows)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    keys = [_identity_keys(row) for row in rows]
    for i, left in enumerate(rows):
        index = ExclusionIndex([left])
        for j in range(i + 1, len(rows)):
            if keys[i] & keys[j] or index.reasons(rows[j]):
                parents[root(j)] = root(i)
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(root(i), []).append(row)
    rng = random.Random(seed)
    pool, excluded = [], []
    for group in groups.values():
        reasons = [reason for row in group for reason in exposure_reasons(row, exposures, purpose='heldout')]
        if reasons:
            excluded.extend({'document_id': row['document_id'], 'reason': 'historical_exposure',
                             'matches': reasons} for row in group)
            continue
        representative = rng.choice(group)
        pool.append(representative)
        excluded.extend({'document_id': row['document_id'], 'reason': 'same_paper_or_duplicate',
                         'representative': representative['document_id']}
                        for row in group if row is not representative)
    rng.shuffle(pool)
    evaluation = pool[:evaluation_count]
    development = pool[evaluation_count:evaluation_count + development_count]
    return deepcopy({'evaluation': evaluation, 'development': development,
                     'reserve': pool[evaluation_count + development_count:],
                     'excluded': excluded, 'seed': seed,
                     'shortages': {'evaluation': max(0, evaluation_count - len(evaluation)),
                                   'development': max(0, development_count - len(development))},
                     'sampling': 'seeded_uniform_paper_components_then_one_snippet',
                     'population': 'software_query_enriched_ranked_search_snippets',
                     'gold_status': 'unannotated', 'training_eligible': False})


def freeze_evaluation(collections, exposures_path, output, *, seed=42,
                      evaluation_count=50, development_count=20):
    output = Path(output)
    if os.path.lexists(output):
        raise FileExistsError(output)
    exposures = load_exposures(Path(exposures_path))
    verify_exposure_sources(exposures)
    sources, candidates = [], {}
    for path in collections:
        path = Path(path).resolve()
        payload = (path / 'manifest.json').read_bytes()
        manifest = json.loads(payload)
        records = {row['path']: row for row in manifest['files']}
        if 'inputs.jsonl' not in records or len(records) != len(manifest['files']):
            raise ValueError('invalid collection inventory')
        for row in records.values():
            verified_path(path, row)
        sources.append({'path': str(path / 'manifest.json'), 'sha256': digest(payload),
                        'status': manifest['status']})
        for row in read_jsonl(path / 'inputs.jsonl'):
            row = validate_input(row)
            ident = row['document_id']
            if ident in candidates:
                previous = candidates[ident]
                if previous['text_revision'] != row['text_revision'] or previous['text'] != row['text']:
                    raise ValueError('conflicting duplicate document')
                previous.setdefault('source_associations', []).extend(row.get('source_associations', []))
                previous['source_associations'].append({'source_ids': row.get('source_ids', {})})
            else:
                candidates[ident] = deepcopy(row)
    result = select_evaluation(list(candidates.values()), exposures, seed=seed,
                              evaluation_count=evaluation_count, development_count=development_count)
    # Recheck source identities before publishing a frozen population.
    for source in sources:
        if digest(Path(source['path']).read_bytes()) != source['sha256']:
            raise ValueError('changed collection manifest')
    verify_exposure_sources(exposures)
    output.mkdir(parents=True, exist_ok=False)
    files = []
    for name, rows in [('inputs', result['evaluation']), ('development', result['development']),
                       ('reserve', result['reserve']), ('excluded', result['excluded'])]:
        payload = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows).encode()
        write_once(output / (name + '.jsonl'), payload)
        files.append({'path': name + '.jsonl', 'sha256': digest(payload)})
    report = {key: value for key, value in result.items()
              if key not in ('evaluation', 'development', 'reserve', 'excluded')}
    report.update(schema_version='focused-evaluation-1', human_reviewed=False,
                  status='shortfall' if any(result['shortages'].values()) else 'frozen',
                  source_manifests=sources, exposures={'path': str(Path(exposures_path).resolve()),
                  'sha256': exposures['_manifest_sha256']},
                  counts={key: len(result[key]) for key in ('evaluation', 'development', 'reserve', 'excluded')},
                  files=files)
    write_once(output / 'manifest.json', json_bytes(report))
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', action='append', required=True, type=Path)
    parser.add_argument('--exposures', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--evaluation-count', type=int, default=50)
    parser.add_argument('--development-count', type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(freeze_evaluation(args.collection, args.exposures, args.output, seed=args.seed,
        evaluation_count=args.evaluation_count, development_count=args.development_count), indent=2))
