"""Build the paragraph-level evaluation population from Softcite's own holdout.

Population unit: one `<p>` paragraph, matching the granularity at which Softcite
produces and scores annotations (TEICorpusSaxHandler writes one entry per `<p>`).

Restricted to articles listed in Softcite's holdout file
`doc/reports/all.negative.empty.holdout.tei.xml`, and only for paragraphs whose
gold software-name spans were recovered exactly (0 mismatches).

Usage:
  .venv-scibert/bin/python reports/scibert-v2/softcite-gold-001/scripts/build_paragraph_dataset.py \
      --paragraphs reports/scibert-v2/softcite-gold-001/paragraphs.jsonl \
      --holdout reports/scibert-v2/softcite-gold-001/softcite-holdout.tei.xml
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
    'unit': 'paragraph',
    'note': ('Softcite human software-name annotations, resolved at the paragraph '
             'granularity and offset granularity Softcite itself uses. Only the '
             'software field is known; intents and sentiment stay unknown.'),
}


def rows(p):
    return [json.loads(l) for l in Path(p).read_text(encoding='utf-8').split('\n')
            if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paragraphs', required=True)
    ap.add_argument('--holdout', required=True)
    ap.add_argument('--suffix', default='para')
    args = ap.parse_args()

    hold = {(i.text or '').strip().lower()
            for i in ET.parse(args.holdout).getroot().iter(T + 'idno')
            if (i.text or '').strip().lower().startswith('10.')}

    units = [u for u in rows(args.paragraphs)
             if (u['article'] or '').strip().lower() in hold]

    inputs, references, coverage = [], [], []
    for u in units:
        if not u['software_names']:
            continue  # Softcite's holdout is negative.empty: annotated paragraphs only
        text = u['text']
        rev = text_revision(text)
        doc_id = f'{u["document_id"]}'
        inputs.append({'document_id': doc_id, 'text': text, 'text_revision': rev,
                       'metadata': {'source': 'softcite-gold',
                                    'doi': u['ids'].get('DOI'),
                                    'article': u['article'],
                                    'paragraph': u['paragraph']}})
        for span in u['software_names']:
            s, e = span['start'], span['end']
            assert text[s:e] == span['name'], (doc_id, s, e, span['name'])
            references.append({
                'record_kind': 'legacy_evaluation_only',
                'document_id': doc_id, 'text_revision': rev,
                'name': span['name'], 'name_span': {'start': s, 'end': e},
                'known': {f: f == 'software' for f in FIELDS},
                'intents': None, 'sentiment': None, 'version_links': [],
                'provenance': PROVENANCE})
        coverage.append({'document_id': doc_id, 'text_revision': rev,
                         'start': 0, 'end': len(text), 'complete': True,
                         'fields': {f: f == 'software' for f in FIELDS},
                         'review_kind': 'published_gold'})

    sfx = args.suffix
    for name, data in (('inputs.jsonl', inputs),
                       ('references.jsonl', references),
                       ('coverage.jsonl', coverage)):
        (OUT / f'{name[:-6]}-{sfx}.jsonl').write_text(
            ''.join(json.dumps(r, ensure_ascii=True) + '\n' for r in data),
            encoding='utf-8')

    articles = {u['article'] for u in units}
    (OUT / f'manifest-{sfx}.json').write_text(json.dumps({
        'unit': 'paragraph',
        'holdout_articles_in_list': len(hold),
        'recovered_articles': len(articles),
        'annotated_paragraphs': len(inputs),
        'gold_software_name_spans': len(references),
        'total_characters': sum(len(i['text']) for i in inputs),
        'provenance': PROVENANCE,
    }, indent=2), encoding='utf-8')

    print(f'holdout articles recovered : {len(articles)}')
    print(f'annotated paragraphs       : {len(inputs)}')
    print(f'gold software-name spans   : {len(references)}')
    print(f'total characters           : {sum(len(i["text"]) for i in inputs)}')


if __name__ == '__main__':
    main()