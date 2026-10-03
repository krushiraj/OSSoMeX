"""Score a single full-label checkpoint on the Softcite paragraph holdout.

`score.py` re-runs every arm, which costs roughly 25 minutes per checkpoint on CPU.
This scores one checkpoint and merges the result into `results-para.json`, so new
candidates can be compared against the existing arms without redoing them.

Usage:
  .venv-scibert/bin/python reports/scibert-v2/softcite-gold-001/scripts/score_one.py \
      --label ossomex-full-label-010-softcite --checkpoint checkpoints/scibert-full-label-010
"""
import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research.contracts import validate_document
from research.evaluation.metrics import evaluate_v2
from research.training.full_label import FullLabelPipeline
from score import ROOT, rows, pipeline_occurrences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--suffix', default='para')
    ap.add_argument('--save', action='store_true')
    args = ap.parse_args()

    documents = sorted((validate_document(d) for d in rows(ROOT / f'inputs-{args.suffix}.jsonl')),
                       key=lambda d: d['document_id'])
    gold = rows(ROOT / f'references-{args.suffix}.jsonl')
    coverage = rows(ROOT / f'coverage-{args.suffix}.jsonl')

    pipe = FullLabelPipeline(Path(args.checkpoint))
    preds, statuses, caps = [], [], None
    for doc in documents:
        n = pipe.predict(doc)
        if isinstance(n.get('capabilities'), dict):
            caps = n['capabilities']
        preds.extend(pipeline_occurrences(doc, n))
        statuses.append({'document_id': doc['document_id'], 'status': n['status']})

    cap = {'software': caps.get('software_spans') is True,
           'versions': caps.get('version_linking') is True,
           'version_offsets': caps.get('version_linking') is True,
           'intents': caps.get('intent') is True,
           'sentiment': caps.get('sentiment') is True}
    m = evaluate_v2(documents, deepcopy(gold), preds, deepcopy(coverage),
                    statuses, cap)['mention_detection']
    print(f"{args.label}\n  tp {m['tp']}  fp {m['fp']}  fn {m['fn']}")
    print(f"  precision {m['precision']:.4f}  recall {m['recall']:.4f}  F1 {m['f1']:.4f}")

    out = ROOT / f'results-{args.suffix}.json'
    if args.save and out.exists():
        payload = json.loads(out.read_text())
        payload['arms'][args.label] = m
        payload['f1_margins'] = {
            k: v for k, v in payload.get('f1_margins', {}).items()
            if not k.startswith(args.label)}
        out.write_text(json.dumps(payload, indent=2) + '\n')
        print(f'merged into {out}')
    else:
        print('(not saved; pass --save to merge into the results file)')


if __name__ == '__main__':
    main()