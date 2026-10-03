"""Score OSSoMeX vs Softcite on Softcite's own published gold holdout.

Population: the subset of Softcite's 994-article holdout whose gold text is
publicly available, at the paragraph granularity Softcite itself scores at.
Gold provenance: https://github.com/softcite/software-mentions doc/reports/all.tei.xml

Softcite's published reference point (doc/latex/scores/scores-summary.tex,
SciBERT-CRF + active sampling, exact-match span level, 994 holdout articles):
    software name   P 69.31  R 72.84  F1 71.03
    publisher                  F1 79.00
    version                    F1 83.88
    URL                        F1 54.55
    F1 micro average           74.56

Every arm is re-run here on the identical paragraphs with the identical
`evaluate_v2` metric, so the table is internally controlled even though the
recovered subset is smaller than their full 994-article holdout.

Offset alignment: Softcite computes annotation offsets on a trimmed accumulator,
so a span preceded by whitespace lands one codepoint early. Softcite's own
evaluator repairs this by locating the text and snapping when the offset is off
by less than the text length; the same repair is applied to both gold and
predicted spans here, and anything still unresolvable is rejected and counted.

Usage:
  .venv-scibert/bin/python reports/scibert-v2/softcite-gold-001/scripts/score.py --suffix para
"""
import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, 'src')
from research.contracts import validate_document
from research.evaluation.metrics import evaluate_v2
from research.training.full_label import FullLabelPipeline
from research.comparison.attributes import pipeline_occurrences, softcite_occurrences

ROOT = Path('reports/scibert-v2/softcite-gold-001')
OURS = {'ossomex-full-label-006': 'checkpoints/scibert-full-label-006',
        'ossomex-full-label-008': 'checkpoints/scibert-full-label-008'}
if os.environ.get('EXTRA_CHECKPOINT'):
    label, path = os.environ['EXTRA_CHECKPOINT'].split('=', 1)
    OURS[label] = path
SOFT_CAP = {'software': True, 'versions': False, 'version_offsets': False,
            'intents': False, 'sentiment': False}
PUBLISHED = {'software_name_precision': 0.6931, 'software_name_recall': 0.7284,
             'software_name_f1': 0.7103, 'micro_average_f1': 0.7456,
             'holdout_articles': 994, 'model': 'scibert-crf+active-sampling'}
ORDER = ['ossomex-full-label-006', 'ossomex-full-label-008',
         'softcite-scibert-0.8.1', 'softcite-wapiti-0.8.1']


def rows(p):
    return [json.loads(l) for l in Path(p).read_text(encoding='utf-8').split('\n')
            if l.strip()]


def snap(text, raw, start):
    """Locate `raw` near `start`; return (start, end) or None."""
    if not raw:
        return None
    if text[start:start + len(raw)] == raw:
        return start, start + len(raw)
    lo = max(0, start - len(raw))
    idx = text.find(raw, lo)
    if 0 <= idx <= start + len(raw):
        return idx, idx + len(raw)
    return None


def softcite_arm(documents, path):
    stored = {r['document_id']: r for r in rows(path)}
    preds, statuses = [], []
    stats = {'mentions': 0, 'exact': 0, 'snapped': 0, 'rejected': 0}
    for doc in documents:
        rec = stored.get(doc['document_id'])
        if rec is None:
            statuses.append({'document_id': doc['document_id'], 'status': 'not_run'})
            continue
        native, code = rec['native'], rec['status_code']
        mentions = (native.get('software') or native.get('mentions')
                    if isinstance(native, dict) else None)
        if code == 204 or not isinstance(mentions, list) or not mentions:
            statuses.append({'document_id': doc['document_id'],
                             'status': rec['status'] or 'no_mentions'})
            continue
        text = doc['text']
        for mention in mentions:
            stats['mentions'] += 1
            name = ((mention.get('software-name') or {}).get('rawForm'))
            start = (mention.get('software-name') or {}).get('offsetStart')
            if name is None or start is None:
                stats['rejected'] += 1
                continue
            fixed = snap(text, name, start)
            if fixed is None:
                stats['rejected'] += 1
                continue
            stats['exact' if fixed[0] == start else 'snapped'] += 1
            patched = dict(mention)
            patched['software-name'] = dict(mention['software-name'],
                                            offsetStart=fixed[0], offsetEnd=fixed[1])
            version = mention.get('version')
            if isinstance(version, dict) and version.get('rawForm') is not None:
                vs, ve = snap(text, version['rawForm'], version.get('offsetStart', 0))
                if vs is None:
                    patched['version'] = None
                else:
                    patched['version'] = dict(version, offsetStart=vs, offsetEnd=ve)
            body = {k: v for k, v in native.items() if k != 'mentions'}
            body['software'] = [patched]
            try:
                preds.extend(softcite_occurrences(
                    doc, [{'window': {'text': text, 'start': 0},
                           'raw': {'native': body}}], 'utf16'))
            except ValueError:
                stats['rejected'] += 1
        statuses.append({'document_id': doc['document_id'], 'status': rec['status']})
    return preds, statuses, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suffix', default='para')
    args = ap.parse_args()
    sfx = args.suffix

    documents = sorted((validate_document(d) for d in rows(ROOT / f'inputs-{sfx}.jsonl')),
                       key=lambda d: d['document_id'])
    gold = rows(ROOT / f'references-{sfx}.jsonl')
    coverage = rows(ROOT / f'coverage-{sfx}.jsonl')

    all_preds, integrity = {}, {}
    for arm, ckpt in OURS.items():
        pipe = FullLabelPipeline(Path(ckpt))
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
        all_preds[arm] = (preds, statuses, cap)

    for arm, filename in (('softcite-scibert-0.8.1', f'para-scibert.jsonl'),
                          ('softcite-wapiti-0.8.1', f'para-wapiti.jsonl')):
        all_preds[arm] = softcite_arm(documents, ROOT / 'raw' / filename)
        integrity[arm] = all_preds[arm][2]

    print(f'population : {len(documents)} paragraphs, {len(gold)} gold software-name spans, '
          f'{sum(len(d["text"]) for d in documents)} chars')
    print(f'gold       : Softcite published TEI annotations (human reviewed)')
    print()
    print(f"{'arm':28} {'tp':>5} {'fp':>5} {'fn':>5} {'prec':>8} {'recall':>8} {'F1':>8}")
    print('-' * 76)
    summary = {}
    for arm in ORDER:
        preds, statuses = all_preds[arm][:2]
        cap = all_preds[arm][2] if arm in OURS else SOFT_CAP
        m = evaluate_v2(documents, deepcopy(gold), preds, deepcopy(coverage),
                        statuses, cap)['mention_detection']
        summary[arm] = m
        fmt = (lambda v: '     n/a' if v is None else f'{v:8.4f}')
        print(f"{arm:28} {m['tp']:5} {m['fp']:5} {m['fn']:5} "
              f"{fmt(m['precision'])} {fmt(m['recall'])} {fmt(m['f1'])}")

    best = max(ORDER, key=lambda a: summary[a]['f1'])
    print()
    print(f'best arm: {best}  F1 {summary[best]["f1"]:.4f} '
          f'(P {summary[best]["precision"]:.4f} / R {summary[best]["recall"]:.4f})')
    margins = {}
    for base in ('softcite-scibert-0.8.1', 'softcite-wapiti-0.8.1'):
        d = summary[best]['f1'] - summary[base]['f1']
        margins[f'{best}_vs_{base}'] = d
        print(f'  {best} vs {base:26} {d:+.4f}')
    pub = PUBLISHED['software_name_f1']
    margins['best_vs_softcite_published'] = summary[best]['f1'] - pub
    print(f'  {best} vs Softcite PUBLISHED name F1 {pub:.4f} '
          f'(their 994-article holdout, not directly comparable): '
          f'{summary[best]["f1"] - pub:+.4f}')

    print()
    print('Softcite span alignment against the same text')
    for arm, v in integrity.items():
        print(f'  {arm:28} mentions {v["mentions"]:5} exact {v["exact"]:5} '
              f'snapped {v["snapped"]:4} rejected {v["rejected"]:4}')

    (ROOT / f'results-{sfx}.json').write_text(json.dumps({
        'population': {'unit': 'paragraph',
                       'paragraphs': len(documents),
                       'gold_software_name_spans': len(gold),
                       'total_characters': sum(len(d['text']) for d in documents),
                       'gold_provenance':
                           'softcite/software-mentions doc/reports/all.tei.xml'},
        'softcite_published_reference': PUBLISHED,
        'arms': summary, 'f1_margins': margins,
        'softcite_alignment': integrity,
    }, indent=2) + '\n', encoding='utf-8')
    print(f'\nwrote {ROOT}/results-{sfx}.json')


if __name__ == '__main__':
    main()