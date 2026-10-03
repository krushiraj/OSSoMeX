"""Does the Softcite holdout expect names the training split already contains?

Softcite reports a single F1 over its holdout. That number hides whether the task is
mostly recall of a fixed vocabulary or genuine generalisation to names never seen in
training. This quantifies the overlap:

  - unique name overlap between train and holdout
  - how many holdout gold spans name something that appears in training
  - coverage weighted by span frequency, not just by unique type
  - the largest unseen holdout vocabularies, which are where generalisation is tested

Names are normalised for matching (casefold, collapse internal whitespace) because
Softcite's surface forms vary for the same tool.

Usage:
  .venv-scibert/bin/python reports/scibert-v2/softcite-gold-001/scripts/name_overlap.py \
      --output reports/scibert-v2/softcite-train-001/name-overlap.json
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_detector import holdout_articles, rows, ROOT


def norm(name):
    return re.sub(r'\s+', ' ', (name or '').strip()).casefold()


def load(holdout):
    """Split paragraph units into train/holdout occurrence lists.

    Membership is decided by the article DOI: holdout articles come from Softcite's
    published holdout file, everything else is training material.
    """
    train, test = [], []
    for unit in rows(ROOT / 'paragraphs.jsonl'):
        doi = (unit['article'] or '').strip().lower()
        bucket = test if doi in holdout else train
        for s in unit['software_names']:
            key = norm(s['name'])
            if not key:
                continue
            bucket.append({'document_id': unit['document_id'], 'article': doi,
                           'paragraph': unit['paragraph'], 'name': s['name'],
                           'normalized': key, 'start': s['start'], 'end': s['end']})
    return train, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output')
    args = ap.parse_args()

    hold = holdout_articles()
    train_occ, test_occ = load(hold)
    train_counts = Counter(o['normalized'] for o in train_occ)
    test_counts = Counter(o['normalized'] for o in test_occ)

    train_types = set(train_counts)
    test_types = set(test_counts)
    seen_types = train_types & test_types
    unseen_types = test_types - train_types

    test_span_total = sum(test_counts.values())
    seen_span_total = sum(c for n, c in test_counts.items() if n in train_types)
    unseen_span_total = test_span_total - seen_span_total
    train_types_not_in_test = train_types - test_types

    result = {
        'population': {
            'train_articles': len({o['article'] for o in train_occ}),
            'holdout_articles': len({o['article'] for o in test_occ}),
            'train_spans': sum(train_counts.values()),
            'holdout_spans': test_span_total,
        },
        'unique_names': {
            'train_types': len(train_types),
            'holdout_types': len(test_types),
            'overlap_types': len(seen_types),
            'holdout_types_seen_in_train': len(seen_types),
            'holdout_types_unseen': len(unseen_types),
            'train_types_absent_from_holdout': len(train_types_not_in_test),
            'type_coverage_recall': (len(seen_types) / len(test_types)) if test_types else None,
            'type_coverage_precision': (len(seen_types) / len(train_types)) if train_types else None,
        },
        'span_weighted': {
            'holdout_spans_naming_a_trained_name': seen_span_total,
            'holdout_spans_naming_an_unseen_name': unseen_span_total,
            'span_coverage_recall': seen_span_total / test_span_total if test_span_total else None,
        },
        'top_train_types': train_counts.most_common(25),
        'top_holdout_types': test_counts.most_common(25),
        'top_unseen_holdout_types': sorted(
            ((n, c) for n, c in test_counts.items() if n not in train_types),
            key=lambda kv: (-kv[1], kv[0]))[:40],
        'seen_holdout_types': sorted(
            ((n, c) for n, c in test_counts.items() if n in train_types),
            key=lambda kv: (-kv[1], kv[0]))[:40],
    }

    p = result['population']
    u = result['unique_names']
    s = result['span_weighted']
    print(f"train: {p['train_articles']} articles, {p['train_spans']} spans, "
          f"{u['train_types']} unique names")
    print(f"holdout: {p['holdout_articles']} articles, {p['holdout_spans']} spans, "
          f"{u['holdout_types']} unique names")
    print()
    print('UNIQUE NAMES')
    print(f"  holdout names also in train : {u['overlap_types']:5} "
          f"({u['type_coverage_recall']:.1%} of holdout vocabulary)")
    print(f"  holdout names NOT in train  : {u['holdout_types_unseen']:5} "
          f"({1 - u['type_coverage_recall']:.1%})")
    print(f"  train names NOT in holdout  : {u['train_types_absent_from_holdout']:5}")
    print()
    print('SPAN-WEIGHTED (what the F1 is actually computed over)')
    print(f"  holdout spans naming a trained name : {s['holdout_spans_naming_a_trained_name']:5} "
          f"({s['span_coverage_recall']:.1%})")
    print(f"  holdout spans naming an unseen name : {s['holdout_spans_naming_an_unseen_name']:5} "
          f"({1 - s['span_coverage_recall']:.1%})")
    print()
    print('most frequent UNSEEN holdout names:')
    for n, c in result['top_unseen_holdout_types'][:20]:
        print(f'  {c:4}  {n}')

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
        # occurrence-level dump so downstream scripts can join on document/offset
        Path(str(args.output).replace('.json', '-occurrences.json')).write_text(
            json.dumps({'train': train_occ, 'holdout': test_occ}, indent=2) + '\n')
        print(f'\nwrote {args.output}')


if __name__ == '__main__':
    main()