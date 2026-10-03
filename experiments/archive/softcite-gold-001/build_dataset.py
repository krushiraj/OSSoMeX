"""Build a fixed evaluation population from Softcite's own gold annotations.

Source: https://github.com/softcite/software-mentions
  - doc/reports/all.tei.xml                     gold TEI corpus (text + <rs type="software">)
  - doc/reports/all.negative.empty.holdout.tei.xml   holdout article DOI list

Softcite's published span-level scores (doc/latex/scores/scores-summary.tex) use exact
match over a 994-article holdout. This script recovers the subset of that holdout for
which gold text is publicly available, and emits the repo's v2 evaluation contracts:

  inputs.jsonl      fixed population documents (frozen text_revision)
  references.jsonl  gold software-name occurrences with explicit field masks
  coverage.jsonl    per-document coverage declaring `software` known over the full text

Only the `software` field is known for these gold spans: Softcite annotates names,
publishers, versions and URLs, but the recoverable text corpus carries only
software-name spans with exact offsets, so intents and sentiment stay unknown and are
scored as unknown rather than as negatives.

Usage:
  .venv-scibert/bin/python reports/scibert-v2/softcite-gold-001/scripts/build_dataset.py \
      --gold reports/scibert-v2/softcite-gold-001/gold-documents.jsonl \
      --holdout-dois /tmp/sc/holdout_dois.json
"""
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from src.research.contracts import FIELDS, text_revision  # noqa: E402

T = '{http://www.tei-c.org/ns/1.0}'
OUT = Path('reports/scibert-v2/softcite-gold-001')

PROVENANCE = {
    'kind': 'softcite_published_gold',
    'source_repository': 'https://github.com/softcite/software-mentions',
    'source_files': ['doc/reports/all.tei.xml',
                     'doc/reports/all.negative.empty.holdout.tei.xml'],
    'annotation_guide': 'doc/annotation_guidelines_tei_xml.md',
    'human_reviewed': True,
    'fields_known': ['software'],
    'note': ('Softcite human annotations for software names. Publisher/version/URL '
             'annotations are not present as offsets in the recoverable text corpus.'),
}


def holdout_dois(path):
    if path.suffix == '.json':
        return {d.strip().lower() for d in json.load(open(path))}
    out = set()
    for i in ET.parse(path).getroot().iter(T + 'idno'):
        v = (i.text or '').strip().lower()
        if v.startswith('10.'):
            out.add(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gold', required=True)
    ap.add_argument('--holdout-dois', required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    hold = holdout_dois(Path(args.holdout_dois))
    docs = [json.loads(l) for l in open(args.gold)]
    selected = [d for d in docs
                if (d.get('ids', {}).get('DOI') or '').strip().lower() in hold]

    inputs, references, coverage = [], [], []
    for d in selected:
        text = d['text']
        rev = text_revision(text)
        inputs.append({
            'document_id': d['document_id'],
            'text': text,
            'text_revision': rev,
            'metadata': {'source': 'softcite-gold', 'doi': d.get('ids', {}).get('DOI'),
                         'pmc': d.get('ids', {}).get('PMC'),
                         'pmid': d.get('ids', {}).get('PMID')},
        })
        for span in d['software_names']:
            start, end = span['start'], span['end']
            assert text[start:end] == span['name'], d['document_id']
            references.append({
                'record_kind': 'legacy_evaluation_only',
                'document_id': d['document_id'],
                'text_revision': rev,
                'name': span['name'],
                'name_span': {'start': start, 'end': end},
                'known': {field: field == 'software' for field in FIELDS},
                'intents': None,
                'sentiment': None,
                'version_links': [],
                'provenance': PROVENANCE,
            })
        coverage.append({
            'document_id': d['document_id'],
            'text_revision': rev,
            'start': 0,
            'end': len(text),
            'complete': True,
            'fields': {field: field == 'software' for field in FIELDS},
            'review_kind': 'published_gold',
        })

    for name, rows in (('inputs.jsonl', inputs),
                       ('references.jsonl', references),
                       ('coverage.jsonl', coverage)):
        (OUT / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n'
                                        for r in rows))

    annotated = [d for d in selected if d['software_names']]
    (OUT / 'manifest.json').write_text(json.dumps({
        'population_documents': len(selected),
        'documents_with_gold_software_names': len(annotated),
        'documents_without_gold_software_names': len(selected) - len(annotated),
        'gold_software_name_spans': sum(len(d['software_names']) for d in selected),
        'total_characters': sum(len(d['text']) for d in selected),
        'holdout_dois_in_list': len(hold),
        'provenance': PROVENANCE,
    }, indent=2))

    print(f'holdout DOIs in list            : {len(hold)}')
    print(f'recovered documents (with text) : {len(selected)}')
    print(f'  of which with >=1 gold span  : {len(annotated)}')
    print(f'gold software-name spans        : '
          f'{sum(len(d["software_names"]) for d in selected)}')
    print(f'total characters               : '
          f'{sum(len(d["text"]) for d in selected)}')


if __name__ == '__main__':
    main()