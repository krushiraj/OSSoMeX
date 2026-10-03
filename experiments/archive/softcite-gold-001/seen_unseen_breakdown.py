"""Per-arm performance split by whether the gold name was seen in training.

`name_overlap.py` shows 59% of holdout spans name software absent from the training
split. That makes the obvious question measurable: do the arms differ on the names
they could have memorised versus the names they must generalise to?

Spans are matched exactly (start, end) as `evaluate_v2` does. Each gold span is
labelled seen/unseen by its normalised name, then:

  recall_seen / recall_unseen  gold spans recovered, per category
  precision_seen / precision_unseen  predictions that matched a gold span of that
                                category, over predictions made in that category

A prediction is attributed to the category of the gold span it matched. Predictions
matching nothing are false positives and counted separately.

Usage:
  .venv-scibert/bin/python reports/scibert-v2/softcite-gold-001/scripts/seen_unseen_breakdown.py \
      --detector 008=checkpoints/scibert-detector-005 \
      --detector 011=checkpoints/scibert-detector-011-softcite \
      --output reports/scibert-v2/softcite-train-001/seen-unseen.json
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, 'src')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from research.training.predict import Detector
from train_detector import holdout_articles, rows, ROOT


def norm(name):
    return re.sub(r'\s+', ' ', (name or '').strip()).casefold()


def gold_spans():
    """Holdout gold software spans keyed by document, plus seen/unseen flags."""
    hold = holdout_articles()
    train_names = set()
    for unit in rows(ROOT / 'paragraphs.jsonl'):
        if (unit['article'] or '').strip().lower() in hold:
            continue
        for s in unit['software_names']:
            k = norm(s['name'])
            if k:
                train_names.add(k)
    out = {}
    for unit in rows(ROOT / 'paragraphs.jsonl'):
        if (unit['article'] or '').strip().lower() not in hold:
            continue
        spans = []
        for s in unit['software_names']:
            k = norm(s['name'])
            if k:
                spans.append({'start': s['start'], 'end': s['end'], 'name': s['name'],
                              'seen': k in train_names})
        if spans:
            out[unit['document_id']] = spans
    return out, train_names


def softcite_spans(path):
    """Softcite raw responses -> exact (start, end) pairs per document.

    Mentions live under `native.mentions`, each with the software name at
    `software-name.offsetStart/offsetEnd`. Records also carry `arm_id`; keep it so
    several arms can share one file.
    """
    out = {}
    for rec in rows(path):
        arm = rec.get('arm_id') or 'softcite'
        native = rec.get('native') or {}
        spans = out.setdefault(arm, {})
        for m in native.get('mentions', []):
            name = m.get('software-name') or {}
            start, end = name.get('offsetStart'), name.get('offsetEnd')
            if isinstance(start, int) and isinstance(end, int):
                spans.setdefault(rec['document_id'], set()).add((start, end))
    return out


def score(gold, preds):
    """Exact-match scoring, split by whether the gold name was seen in training."""
    stat = {k: Counter() for k in ('seen', 'unseen')}
    unmatched = 0
    for did, spans in gold.items():
        predicted = preds.get(did, set())
        remaining = set(predicted)
        for g in spans:
            key = (g['start'], g['end'])
            cat = 'seen' if g['seen'] else 'unseen'
            stat[cat]['gold'] += 1
            if key in remaining:
                stat[cat]['tp'] += 1
                remaining.discard(key)
            else:
                stat[cat]['fn'] += 1
        unmatched += len(remaining)
    stat['unmatched_predictions'] = unmatched
    # Predictions are attributed to the category of the gold span they matched, so a
    # false positive belongs to neither category and is reported separately. That
    # keeps per-category precision interpretable instead of double-counting the same
    # prediction against both categories.
    stat['overall'] = Counter()
    for cat in ('seen', 'unseen'):
        stat['overall']['tp'] += stat[cat]['tp']
        stat['overall']['gold'] += stat[cat]['gold']
        stat['overall']['fn'] += stat[cat]['fn']
    o = stat['overall']
    o['fp'] = unmatched
    for cat in ('seen', 'unseen', 'overall'):
        s = stat[cat]
        denom = s['tp'] + s.get('fp', 0)
        s['precision'] = s['tp'] / denom if denom else None
        s['recall'] = s['tp'] / s['gold'] if s['gold'] else None
        s['f1'] = (2 * s['precision'] * s['recall'] / (s['precision'] + s['recall'])
                   if s['precision'] and s['recall'] else None)
    return stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', action='append', default=[],
                    help='label=checkpoint, repeatable')
    ap.add_argument('--softcite', action='append', default=[],
                    help='label=raw-jsonl, repeatable')
    ap.add_argument('--output')
    args = ap.parse_args()

    gold, train_names = gold_spans()
    print(f'holdout documents with gold spans: {len(gold)}')
    print(f'training vocabulary: {len(train_names)} normalized names')
    print()

    results = {}
    inputs = {r['document_id']: r for r in rows(ROOT / 'inputs-para.jsonl')}
    for spec in args.detector:
        label, ckpt = spec.split('=', 1)
        det = Detector(ckpt)
        preds = {did: {(s['start'], s['end']) for s in det.predict(inputs[did])['spans']
                       if s['label'] == 'SOFTWARE'} for did in gold}
        results[label] = score(gold, preds)
        print(f'{label} scored', flush=True)
    for spec in args.softcite:
        label, path = spec.split('=', 1)
        per_arm = softcite_spans(path)
        if label in per_arm:
            per_arm = {label: per_arm[label]}
        else:
            for arm in per_arm:
                print(f'  note: {path} declares arm_id={arm}, scored under that name')
        for arm, spans in per_arm.items():
            results[arm] = score(gold, spans)
            print(f'{arm} scored', flush=True)

    header = f"{'arm':30} {'cat':8} {'gold':>5} {'tp':>5} {'fn':>5} {'prec':>8} {'recall':>8}"
    print(header)
    print('-' * len(header))
    for label, stat in results.items():
        for cat in ('seen', 'unseen'):
            s = stat[cat]
            p = 'n/a' if s['precision'] is None else f'{s["precision"]:.4f}'
            r = 'n/a' if s['recall'] is None else f'{s["recall"]:.4f}'
            print(f'{label:30} {cat:8} {s["gold"]:5} {s["tp"]:5} {s["fn"]:5} {p:>8} {r:>8}')
        print(f'{label:30} {"unmatched":8} {stat["unmatched_predictions"]:5} '
              f'false positives (excluded from per-category precision)')
        o = stat['overall']
        p = 'n/a' if o['precision'] is None else f'{o["precision"]:.4f}'
        r = 'n/a' if o['recall'] is None else f'{o["recall"]:.4f}'
        f = 'n/a' if o['f1'] is None else f'{o["f1"]:.4f}'
        print(f'{label:30} {"OVERALL":8} {o["gold"]:5} {o["tp"]:5} {o["fn"]:5} '
              f'{p:>8} {r:>8}  F1 {f}')
        print()

    if args.output:
        Path(args.output).write_text(json.dumps({
            'population': {'documents': len(gold), 'train_vocabulary': len(train_names)},
            'arms': {k: {c: dict(v) for c, v in s.items() if isinstance(v, Counter)}
                     for k, s in results.items()},
        }, indent=2) + '\n')
        print(f'wrote {args.output}')


if __name__ == '__main__':
    main()