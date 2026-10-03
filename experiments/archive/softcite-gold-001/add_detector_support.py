"""Recompute detector support statistics and record them in a checkpoint manifest.

`load_pipeline_manifest` requires each stage's declared `support` to match the value
it derives from the checkpoint, so a newly trained detector needs real support counts
rather than a previous detector's.

Classification is by label prefix, with `O` counted as its own class. Note that a
naive `'SOFTWARE' in label` test misfiles `O` into whichever bucket is the `else`
branch, which silently inflates that class.

Usage:
  .venv-scibert/bin/python reports/scibert-v2/softcite-gold-001/scripts/add_detector_support.py \
      --checkpoint checkpoints/scibert-detector-011-softcite --unit document
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, 'src')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from research.training.features import LABELS, build_token_features
from research.training.runner import BASE_MODEL, BASE_REVISION
from train_detector import holdout_articles, occurrences_for, rows, ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--unit', choices=('document', 'paragraph'), default='document')
    args = ap.parse_args()
    ckpt = Path(args.checkpoint)
    manifest = json.loads((ckpt / 'manifest.json').read_text())
    prov = manifest['provenance']
    config = manifest['recipe']

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    base = Path(snapshot_download(BASE_MODEL, revision=BASE_REVISION, local_files_only=True))
    tokenizer = AutoTokenizer.from_pretrained(base, do_lower_case=False, use_fast=True,
                                              local_files_only=True, trust_remote_code=False)

    hold = holdout_articles()
    dev_articles = set(prov['dev_article_ids'])
    ablation = prov['ablation_negative_fraction']

    if args.unit == 'document':
        items = [d for d in rows(ROOT / 'gold-documents.jsonl')
                 if (d['ids'].get('DOI') or '').strip().lower() not in hold
                 and (d['ids'].get('DOI') or '').strip().lower() not in dev_articles]
        built = []
        r = random.Random(prov.get('seed', 42))
        for rec in items:
            text = rec['text']
            coverage = [{'start': 0, 'end': len(text),
                         'fields': {'software': True, 'versions': True}}]
            occ = occurrences_for(rec)
            built.append(({'document_id': rec['document_id'], 'text': text, 'offset_base': 0},
                          occ, coverage))
            if occ and r.random() < ablation:
                built.append(({'document_id': rec['document_id'] + '#ablated', 'text': text,
                               'offset_base': 0}, [], coverage))
    else:
        items = [u for u in rows(ROOT / 'paragraphs.jsonl')
                 if (u['article'] or '').strip().lower() not in hold
                 and (u['article'] or '').strip().lower() not in dev_articles]
        built = []
        r = random.Random(prov.get('seed', 42))
        for u in items:
            text = u['text']
            coverage = [{'start': 0, 'end': len(text),
                         'fields': {'software': True, 'versions': True}}]
            occ = [{'name': s['name'], 'name_span': {'start': s['start'], 'end': s['end']},
                    'known': {'software': True, 'versions': True}, 'version_links': []}
                   for s in u['software_names']]
            built.append(({'document_id': u['document_id'], 'text': text, 'offset_base': 0},
                          occ, coverage))
            if occ and r.random() < ablation:
                built.append(({'document_id': u['document_id'] + '#ablated', 'text': text,
                               'offset_base': 0}, [], coverage))

    counts = {'SOFTWARE': 0, 'VERSION': 0, 'outside': 0, 'O': 0}
    works = {k: set() for k in counts}
    for document, occ, coverage in built:
        f = build_token_features(document, occ, coverage, tokenizer, config)
        for w in f['windows']:
            if not any(w['active_mask']):
                continue
            for active, allowed in zip(w['active_mask'], w['allowed_labels']):
                if not active:
                    continue
                if all(allowed):
                    key = 'outside'
                elif allowed.count(True) == 1:
                    label = LABELS[allowed.index(True)]
                    key = ('O' if label == 'O'
                           else 'SOFTWARE' if 'SOFTWARE' in label else 'VERSION')
                else:
                    continue  # ambiguous token, no single target
                counts[key] += 1
                works[key].add(document['document_id'])

    support = {'counts': counts,
               'works': {k: len(v) for k, v in works.items()},
               'coverage_warnings': [], 'trainable': True, 'unavailable_reasons': []}
    manifest['training']['support'] = support
    manifest['support'] = support
    manifest['provenance']['support_note'] = (
        'O counts supervised tokens whose only allowed label is O; `outside` counts '
        'unconstrained tokens. Detector-011 supervises both software and versions.')
    (ckpt / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(support, indent=2))


if __name__ == '__main__':
    main()