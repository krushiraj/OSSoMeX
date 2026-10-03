"""Regression gate: run one checkpoint on any inputs/references pair.

Used to check that a candidate adapted on Softcite gold has not regressed on our own
held-out sets. Accepts arbitrary paths so the same script covers the exposed50,
Fresh30 and full-text corpora.

Note on reference provenance: exposed50 and Fresh30 references are agent-provisional,
not human gold. They are a regression signal only, never a benchmark claim.

Usage:
  .venv-scibert/bin/python reports/scibert-v2/softcite-gold-001/scripts/regress.py \
      --label 010 --checkpoint checkpoints/scibert-full-label-010 \
      --inputs data/scibert-v2/focused-evaluation-001/inputs.jsonl \
      --references reports/scibert-v2/focused-cycle-003/exposed50/spans/references.jsonl \
      --coverage reports/scibert-v2/focused-cycle-003/exposed50/spans/references.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, 'src')
from research.contracts import validate_document
from research.evaluation.metrics import evaluate_v2
from research.training.full_label import FullLabelPipeline


def rows(p):
    return [json.loads(l) for l in Path(p).read_text(encoding='utf-8').split('\n') if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--inputs', required=True)
    ap.add_argument('--references', required=True)
    ap.add_argument('--coverage', help='defaults to --references when omitted')
    ap.add_argument('--output')
    args = ap.parse_args()

    documents = sorted((validate_document(d) for d in rows(args.inputs)),
                       key=lambda d: d['document_id'])
    gold = rows(args.references)
    coverage = rows(args.coverage or args.references)

    pipe = FullLabelPipeline(Path(args.checkpoint))
    preds, statuses, caps = [], [], None
    for doc in documents:
        n = pipe.predict(doc)
        if isinstance(n.get('capabilities'), dict):
            caps = n['capabilities']
        for occ in (n.get('occurrences') or []):
            preds.append({**occ, 'document_id': doc['document_id']})
        statuses.append({'document_id': doc['document_id'], 'status': n['status']})

    cap = {'software': caps.get('software_spans') is True,
           'versions': caps.get('version_linking') is True,
           'version_offsets': caps.get('version_linking') is True,
           'intents': caps.get('intent') is True,
           'sentiment': caps.get('sentiment') is True}
    result = evaluate_v2(documents, gold, preds, coverage, statuses, cap)
    m = result['mention_detection']
    print(f"{args.label}  tp {m['tp']}  fp {m['fp']}  fn {m['fn']}")
    print(f"  precision {m['precision']:.4f}  recall {m['recall']:.4f}  F1 {m['f1']:.4f}")
    for k, v in result.items():
        if k == 'mention_detection':
            continue
        print(f'  {k}: {json.dumps(v)[:160]}')
    if args.output:
        Path(args.output).write_text(json.dumps(
            {'label': args.label, 'checkpoint': args.checkpoint,
             'inputs': args.inputs, 'references': args.references,
             'mention_detection': m, 'result': result}, indent=2) + '\n')
        print(f'wrote {args.output}')


if __name__ == '__main__':
    main()