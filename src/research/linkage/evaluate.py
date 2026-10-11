"""Paper-level link precision/recall in WP4 KPI terms, with paper-clustered bootstrap intervals."""

import random

IDENTIFIABLE = {None, 'not_in_registry'}


def link_metrics(gold, predictions):
    predicted = {p['task_id']: p['entity_id'] for p in predictions}
    missing = [g['task_id'] for g in gold if g['task_id'] not in predicted]
    if missing:
        raise ValueError(f'MISSING_PREDICTIONS: {len(missing)} tasks, e.g. {missing[0]}')
    counts = {'predicted_links': 0, 'correct_links': 0, 'gold_links': 0}
    errors = {'wrong_entity': 0, 'linked_nil': 0, 'missed': 0}
    nil = {'gold': 0, 'abstained': 0}
    identifiable = 0
    for g in gold:
        guess = predicted[g['task_id']]
        linked = bool(g['entity_ids'])
        counts['gold_links'] += linked
        counts['predicted_links'] += guess is not None
        identifiable += g['nil_reason'] in IDENTIFIABLE
        if linked:
            if guess in g['entity_ids']:
                counts['correct_links'] += 1
            elif guess is None:
                errors['missed'] += 1
            else:
                errors['wrong_entity'] += 1
        else:
            nil['gold'] += 1
            if guess is None:
                nil['abstained'] += 1
            else:
                errors['linked_nil'] += 1
    p = counts['correct_links'] / counts['predicted_links'] if counts['predicted_links'] else None
    r = counts['correct_links'] / counts['gold_links'] if counts['gold_links'] else None
    f1 = 2 * p * r / (p + r) if p and r else None
    return {**counts, 'precision': p, 'recall': r, 'f1': f1, 'nil': nil, 'errors': errors,
            'registry_coverage': counts['gold_links'] / identifiable if identifiable else None}


def bootstrap(gold, predictions, samples=2000, seed=0):
    papers = {}
    for g in gold:
        papers.setdefault(g['paper_id'], []).append(g)
    keys = sorted(papers)
    rng = random.Random(seed)
    values = {'precision': [], 'recall': [], 'f1': []}
    for _ in range(samples):
        resample = [g for k in rng.choices(keys, k=len(keys)) for g in papers[k]]
        m = link_metrics(resample, predictions)
        for name in values:
            if m[name] is not None:
                values[name].append(m[name])
    interval = lambda xs: [sorted(xs)[int(0.025 * (len(xs) - 1))], sorted(xs)[int(0.975 * (len(xs) - 1))]] if xs else None
    return {name: interval(xs) for name, xs in values.items()}
