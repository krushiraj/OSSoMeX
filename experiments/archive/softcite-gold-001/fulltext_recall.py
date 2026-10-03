"""Independent software recall for one detector run over the full-text corpus.

Reuses the ecosystem project links in the corpus manifest as an external reference,
the same source `compare_fulltext.py` uses. Offsets are not available for those
links, so this reports recall only -- never precision or F1.

Usage:
  .venv-scibert/bin/python reports/scibert-v2/softcite-gold-001/scripts/fulltext_recall.py \
      --run reports/scibert-v2/softcite-train-001/fulltext-010/results.jsonl \
      --label ossomex-detector-010-softcite \
      --baseline reports/scibert-v2/fulltext-001/run-fourway-002/results.jsonl
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

IN = Path('reports/scibert-v2/fulltext-001/inputs.jsonl')
MANIFEST = Path('reports/scibert-v2/fulltext-001/corpus-manifest.json')


def rows(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def norm(s):
    return re.sub(r'[^a-z0-9]+', '', unicodedata.normalize('NFKD', s or '').lower())


def project_name(url):
    """`ecosystems_projects` entries are `<registry>/<project>`, not URLs."""
    return url.rsplit('/', 1)[-1] if '/' in url else url


def recall(docs, projects_by_doc, spans_by_doc):
    """Substring containment either way, matching compare_fulltext.py's matching rule."""
    tot = hit = 0
    for did, projects in projects_by_doc.items():
        predicted = {norm(s['text']) for s in spans_by_doc.get(did, []) if norm(s['text'])}
        for entry in projects:
            key = norm(project_name(entry))
            if not key:
                continue
            tot += 1
            if any(key in p or p in key for p in predicted):
                hit += 1
    return hit, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--baseline')
    ap.add_argument('--baseline-label', help='arm name for a baseline file without arm_id')
    ap.add_argument('--output')
    args = ap.parse_args()

    docs = {d['document_id']: d for d in rows(IN)}
    projects_by_doc = {p['document_id']: p['ecosystems_projects']
                       for p in json.loads(MANIFEST.read_text())['papers']}

    def load(path, key=None):
        """Group spans by arm.

        Manifest runs carry an explicit `arm_id`; `detector predict` output does not,
        so it is grouped under the caller-supplied key.
        """
        spans = {}
        for rec in rows(path):
            k = rec.get('arm_id') or key
            spans.setdefault(k, {})[rec['document_id']] = [
                s for s in rec.get('spans', []) if s.get('label') == 'SOFTWARE']
        return spans

    baseline_key = args.baseline_label or 'baseline'
    runs = load(args.run, args.label)
    if args.baseline:
        for arm, spans in load(args.baseline, baseline_key).items():
            runs.setdefault(arm, spans)

    out = {}
    for arm, spans_by_doc in runs.items():
        hit, tot = recall(docs, projects_by_doc, spans_by_doc)
        if not tot:
            print(f'{arm:36} no reference projects, skipped')
            continue
        r = hit / tot
        out[arm] = {'hit': hit, 'total': tot, 'recall': r}
        print(f'{arm:36} recall {hit:4}/{tot:4} = {r:.4f}')

    if args.output:
        Path(args.output).write_text(json.dumps(
            {'source': 'ecosystems project links (no offsets; recall only)',
             'documents': len(docs), 'arms': out}, indent=2) + '\n')
        print(f'wrote {args.output}')


if __name__ == '__main__':
    main()