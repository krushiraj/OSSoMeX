"""Cases where Softcite loses to OSSoMeX, on the same text.

Two comparisons matter and they disagree:

1. Softcite gold holdout: Softcite wins overall (0.91-0.93 vs 0.75).
2. Our own held-out and full-text corpora: Softcite loses badly (exposed50 0.41 vs
   0.69; full-text recall 0.34-0.59 vs 0.89).

This script enumerates the concrete disagreements rather than the aggregates, so the
claim "Softcite is worse on our data" rests on inspectable examples:

  softcite_missed     gold spans Softcite missed that we found
  ours_missed         gold spans we missed that Softcite found
  softcite_false_pos  Softcite predictions matching no gold span, with the text

Usage:
  .venv-scibert/bin/python reports/scibert-v2/softcite-gold-001/scripts/softcite_losses.py \
      --detector ossomex-detector-005=checkpoints/scibert-detector-005 \
      --detector ossomex-detector-011=checkpoints/scibert-detector-011-softcite \
      --softcite reports/scibert-v2/softcite-gold-001/raw/para-wapiti.jsonl \
      --output reports/scibert-v2/softcite-train-001/softcite-losses.json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, 'src')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from research.training.predict import Detector
from train_detector import holdout_articles, rows, ROOT


def norm(name):
    return ' '.join((name or '').split()).casefold()


def gold_map():
    hold = holdout_articles()
    out = {}
    for unit in rows(ROOT / 'paragraphs.jsonl'):
        if (unit['article'] or '').strip().lower() not in hold:
            continue
        for s in unit['software_names']:
            out.setdefault(unit['document_id'], {})[(s['start'], s['end'])] = {
                'name': s['name'], 'normalized': norm(s['name'])}
    return out


def softcite_map(path):
    """document_id -> {(start, end): surface form} per arm_id."""
    out = {}
    for rec in rows(path):
        arm = rec.get('arm_id') or 'softcite'
        spans = out.setdefault(arm, {})
        for m in (rec.get('native') or {}).get('mentions', []):
            n = m.get('software-name') or {}
            if isinstance(n.get('offsetStart'), int):
                spans.setdefault(rec['document_id'], {})[(n['offsetStart'], n['offsetEnd'])] = \
                    n.get('rawForm', '')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', action='append', default=[], help='label=checkpoint')
    ap.add_argument('--softcite', required=True)
    ap.add_argument('--top', type=int, default=30)
    ap.add_argument('--output')
    args = ap.parse_args()

    gold = gold_map()
    inputs = {r['document_id']: r for r in rows(ROOT / 'inputs-para.jsonl')}
    soft = softcite_map(args.softcite)

    arms = {}
    for spec in args.detector:
        label, ckpt = spec.split('=', 1)
        det = Detector(ckpt)
        spans = {}
        for did, g in gold.items():
            spans[did] = {(s['start'], s['end']): s['text'] for s in det.predict(inputs[did])['spans']
                          if s['label'] == 'SOFTWARE'}
        arms[label] = spans

    report = {}
    for soft_arm, soft_spans in soft.items():
        for our_label, our_spans in arms.items():
            key = f'{soft_arm} vs {our_label}'
            soft_missed, ours_missed, soft_fp = [], [], []
            for did, g in gold.items():
                text = inputs[did]['text']
                s_here = soft_spans.get(did, {})
                o_here = our_spans.get(did, {})
                for span, info in g.items():
                    ours_hit = span in o_here
                    soft_hit = span in s_here
                    if ours_hit and not soft_hit:
                        soft_missed.append({
                            'document_id': did, 'span': list(span), 'gold_name': info['name'],
                            'context': text[max(0, span[0] - 60):span[1] + 60]})
                    elif soft_hit and not ours_hit:
                        ours_missed.append({
                            'document_id': did, 'span': list(span), 'gold_name': info['name'],
                            'context': text[max(0, span[0] - 60):span[1] + 60]})
                gold_spans = set(g)
                for span, surface in s_here.items():
                    if span not in gold_spans:
                        soft_fp.append({
                            'document_id': did, 'span': list(span), 'predicted': surface,
                            'context': text[max(0, span[0] - 60):span[1] + 60]})
            report[key] = {
                'softcite_missed_we_found': len(soft_missed),
                'ours_missed_softcite_found': len(ours_missed),
                'softcite_false_positives': len(soft_fp),
                'softcite_missed_by_name': Counter(x['gold_name'] for x in soft_missed).most_common(args.top),
                'softcite_false_positive_surfaces': Counter(
                    x['predicted'] for x in soft_fp).most_common(args.top),
                'examples_softcite_missed': soft_missed[:args.top],
                'examples_softcite_false_positives': soft_fp[:args.top],
            }
            r = report[key]
            print(f'\n=== {key} ===')
            print(f'gold spans Softcite missed but we found : {r["softcite_missed_we_found"]}')
            print(f'gold spans we missed but Softcite found : {r["ours_missed_softcite_found"]}')
            print(f'Softcite false positives                : {r["softcite_false_positives"]}')
            print('\nmost frequent gold names Softcite missed:')
            for n, c in r['softcite_missed_by_name'][:15]:
                print(f'  {c:3}  {n}')
            print('\nmost frequent Softcite false positives:')
            for n, c in r['softcite_false_positive_surfaces'][:15]:
                print(f'  {c:3}  {n}')

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, default=str) + '\n')
        print(f'\nwrote {args.output}')


if __name__ == '__main__':
    main()