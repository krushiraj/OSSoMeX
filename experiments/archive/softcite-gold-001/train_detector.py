"""Train a SOFTWARE+VERSION span detector on Softcite's published human gold.

Supersedes detector-010, which declared only `software` coverage. Under
`partial_token_loss` a token whose allowed-label set is {O, B-VERSION, I-VERSION} is
already satisfied by a VERSION prediction, so leaving versions unconstrained gave the
model a zero-cost way to label everything as a version. Detector-010 did exactly that:
on the 20 full-text documents it emitted 10,358 spans on one document against
detector-005's 285, and independent recall collapsed to 0.

This trainer fixes that by supervising both fields. Versions are real annotations in
Softcite's TEI (`other_entities` with `role == 'version'`), so rather than declaring
versions absent we use them as positives and declare full coverage for both fields.
Every token then has an allowed set of at most {O, B-SOFTWARE, I-SOFTWARE}, so an
unannotated token is unambiguously a negative.

Training runs at document level, not paragraph level, because that is the unit the
detector actually sees at inference on full text.

Train pool: articles NOT in `doc/reports/all.negative.empty.holdout.tei.xml`. The 235
recovered holdout articles are never read here.

Usage:
  .venv-scibert/bin/python reports/scibert-v2/softcite-gold-001/scripts/train_detector.py \
      --output checkpoints/scibert-detector-011-softcite --epochs 5 --unit document
"""
import argparse
import json
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, 'src')
from research.training.features import LABELS, build_token_features
from research.training.runner import (BASE_MODEL, BASE_REVISION, choose_device,
                                      save_detector, train_model)

ROOT = Path('reports/scibert-v2/softcite-gold-001')
HOLDOUT = ROOT / 'softcite-holdout.tei.xml'
T = '{http://www.tei-c.org/ns/1.0}'


def rows(p):
    return [json.loads(l) for l in Path(p).read_text(encoding='utf-8').split('\n') if l.strip()]


def holdout_articles():
    return {(i.text or '').strip().lower()
            for i in ET.parse(HOLDOUT).getroot().iter(T + 'idno')
            if (i.text or '').strip().lower().startswith('10.')}


def occurrences_for(rec):
    """Build feature-builder occurrences from one gold document.

    Versions nested in a software name attach to that occurrence; versions standing
    alone become their own occurrence with only the version field known.
    """
    text = rec['text']
    names = rec.get('software_names') or []
    versions = [e for e in (rec.get('other_entities') or []) if e.get('role') == 'version']

    def as_edge(v):
        return {'span': {'start': v['start'], 'end': v['end']},
                'text': text[v['start']:v['end']]}

    occ, claimed = [], set()
    for n in names:
        nested = [v for v in versions
                  if v['start'] >= n['start'] and v['end'] <= n['end'] + 2]
        for v in nested:
            claimed.add((v['start'], v['end']))
        occ.append({'name': n['name'], 'name_span': {'start': n['start'], 'end': n['end']},
                    'known': {'software': True, 'versions': True},
                    'version_links': [as_edge(v) for v in nested]})
    loose = [v for v in versions if (v['start'], v['end']) not in claimed]
    if loose:
        occ.append({'name': '', 'name_span': {'start': 0, 'end': 0},
                    'known': {'software': False, 'versions': True},
                    'version_links': [as_edge(v) for v in loose]})
    return occ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--learning-rate', type=float, default=5e-5)
    ap.add_argument('--microbatch-size', type=int, default=2)
    ap.add_argument('--gradient-accumulation', type=int, default=16)
    ap.add_argument('--max-length', type=int, default=482)
    ap.add_argument('--overlap', type=int, default=64)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dev-article-fraction', type=float, default=0.12)
    ap.add_argument('--ablation-fraction', type=float, default=0.15)
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, BertForTokenClassification
    import torch

    hold = holdout_articles()
    docs = [d for d in rows(ROOT / 'gold-documents.jsonl')
            if (d['ids'].get('DOI') or '').strip().lower() not in hold]
    articles = sorted({(d['ids'].get('DOI') or '').strip().lower() for d in docs})
    rng = random.Random(args.seed)
    shuffled = articles[:]
    rng.shuffle(shuffled)
    n_dev = max(1, int(len(shuffled) * args.dev_article_fraction))
    dev_articles = set(shuffled[:n_dev])
    train_docs = [d for d in docs if (d['ids'].get('DOI') or '').strip().lower() not in dev_articles]
    dev_docs = [d for d in docs if (d['ids'].get('DOI') or '').strip().lower() in dev_articles]

    config = {'base_model': BASE_MODEL, 'base_revision': BASE_REVISION,
              'seed': args.seed, 'epochs': args.epochs,
              'learning_rate': args.learning_rate, 'weight_decay': 0.01,
              'warmup_ratio': 0.1, 'microbatch_size': args.microbatch_size,
              'gradient_accumulation': args.gradient_accumulation,
              'max_grad_norm': 1.0, 'max_length': args.max_length,
              'overlap': args.overlap, 'min_steps_per_epoch': 10}

    base = Path(snapshot_download(BASE_MODEL, revision=BASE_REVISION, local_files_only=True))
    tokenizer = AutoTokenizer.from_pretrained(base, do_lower_case=False, use_fast=True,
                                              local_files_only=True, trust_remote_code=False)

    def examples(items, seed):
        r = random.Random(seed)
        out = []
        for rec in items:
            text = rec['text']
            coverage = [{'start': 0, 'end': len(text),
                         'fields': {'software': True, 'versions': True}}]
            occ = occurrences_for(rec)
            out.append(({'document_id': rec['document_id'], 'text': text, 'offset_base': 0},
                        occ, coverage))
            if occ and r.random() < args.ablation_fraction:
                out.append(({'document_id': rec['document_id'] + '#ablated', 'text': text,
                             'offset_base': 0}, [], coverage))
        return out

    def featurize(exs):
        groups, counts, exclusions = {}, {'SOFTWARE': 0, 'VERSION': 0, 'outside': 0}, 0
        for document, occ, coverage in exs:
            f = build_token_features(document, occ, coverage, tokenizer, config)
            if not f['windows']:
                continue
            groups[document['document_id']] = f['windows']
            exclusions += len(f['exclusions'])
            for w in f['windows']:
                if not any(w['active_mask']):
                    continue
                for active, allowed in zip(w['active_mask'], w['allowed_labels']):
                    if not active:
                        continue
                    if all(allowed):
                        counts['outside'] += 1
                    elif allowed.count(True) == 1:
                        label = LABELS[allowed.index(True)]
                        counts['SOFTWARE' if 'SOFTWARE' in label else 'VERSION'] += 1
        return groups, counts, exclusions

    train_groups, train_counts, train_excl = featurize(examples(train_docs, args.seed))
    dev_groups, dev_counts, _ = featurize(examples(dev_docs, args.seed + 1))
    print(f'train articles {len(articles) - n_dev} | dev articles {n_dev}')
    print(f'train windows {sum(len(v) for v in train_groups.values())} | '
          f'dev windows {sum(len(v) for v in dev_groups.values())}')
    print(f'supervised tokens {train_counts} | exclusions {train_excl}')

    torch.manual_seed(args.seed)
    model = BertForTokenClassification.from_pretrained(
        base, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
        label2id={l: n for n, l in enumerate(LABELS)}, classifier_dropout=.1,
        local_files_only=True, trust_remote_code=False, attn_implementation='eager')
    report = train_model(model, tokenizer, train_groups, config, args.device,
                         on_epoch=lambda row: print(json.dumps(row), flush=True))

    def works(exs, want):
        ids = set()
        for document, occ, coverage in exs:
            f = build_token_features(document, occ, coverage, tokenizer, config)
            for w in f['windows']:
                if not any(w['active_mask']):
                    continue
                for active, allowed in zip(w['active_mask'], w['allowed_labels']):
                    if not active:
                        continue
                    hit = (all(allowed) if want == 'outside'
                           else allowed.count(True) == 1
                           and (want in LABELS[allowed.index(True)]))
                    if hit:
                        ids.add(document['document_id'])
                        break
        return len(ids)

    tr_ex, dv_ex = examples(train_docs, args.seed), examples(dev_docs, args.seed + 1)
    support = {
        'counts': train_counts,
        'works': {'SOFTWARE': works(tr_ex, 'SOFTWARE'), 'VERSION': works(tr_ex, 'VERSION'),
                  'outside': works(tr_ex, 'outside')},
        'coverage_warnings': [], 'trainable': True, 'unavailable_reasons': [],
    }
    provenance = {
        'purpose': 'benchmark_adaptation_on_softcite_published_gold',
        'training_source': 'softcite/software-mentions doc/reports/all.tei.xml',
        'excluded': 'all articles in doc/reports/all.negative.empty.holdout.tei.xml',
        'unit': 'document',
        'train_articles': len(articles) - n_dev, 'dev_articles': n_dev,
        'dev_article_ids': sorted(dev_articles),
        'documents_train': len(train_docs), 'documents_dev': len(dev_docs),
        'gold_spans_train': sum(len(d.get('software_names') or []) for d in train_docs),
        'gold_versions_train': sum(
            1 for d in train_docs for e in (d.get('other_entities') or [])
            if e.get('role') == 'version'),
        'ablation_negative_fraction': args.ablation_fraction,
        'supervised_fields': ['software', 'versions'],
        'fields_known': ['software', 'versions'],
        'not_supervised': ['intents', 'sentiment'],
        'fixes': ('detector-010 left versions unconstrained, so partial_token_loss '
                  'accepted VERSION anywhere and the model labelled everything a '
                  'version. Both fields are now covered and versions are supervised '
                  'from Softcite role=="version" annotations.'),
        'note': ('Detector-only adaptation. Intent, sentiment and alias heads are '
                 'unchanged and come from the existing full-label bundle.'),
        'base_model': BASE_MODEL, 'base_revision': BASE_REVISION,
        'support': support,
    }
    manifest = save_detector(model, tokenizer, args.output, config, report, provenance)
    manifest['training']['support'] = support
    manifest['support'] = support
    (Path(args.output) / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (Path(args.output) / 'softcite-devsplit.json').write_text(json.dumps(
        {'dev_articles': n_dev, 'dev_article_ids': sorted(dev_articles),
         'dev_supervised_tokens': dev_counts}, indent=2) + '\n')
    print(json.dumps(support, indent=2))
    print(f'wrote {args.output} (status={manifest["status"]})')


if __name__ == '__main__':
    main()