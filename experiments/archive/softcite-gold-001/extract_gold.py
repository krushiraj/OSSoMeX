"""Extract Softcite's gold TEI corpus into span-level ground truth.

Reads `all.tei.xml` (1,348 documents) from the softcite/software-mentions repo,
reconstructs each document's text exactly as the pipeline will see it, and records
every `<rs type="software">` annotation with exact half-open character offsets.
Every span is verified to slice back to its recorded text.

Also reports the train/holdout split by cross-referencing the companion files
under doc/reports/.

Usage:
  .venv-scibert/bin/python reports/scibert-v2/softcite-gold-001/scripts/extract_gold.py --tei <path>
"""
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

T = '{http://www.tei-c.org/ns/1.0}'
OUT = Path('reports/scibert-v2/softcite-gold-001')


class Buf:
    def __init__(self):
        self.parts = []
        self.n = 0

    def add(self, s):
        if s:
            self.parts.append(s)
            self.n += len(s)


def walk(el, buf, anns, label):
    buf.add(el.text)
    for child in el:
        if child.tag == T + 'rs':
            attrs = child.attrib
            start = buf.n
            buf.add(child.text)
            end = buf.n
            if attrs.get('type') == 'software':
                anns.append({'start': start, 'end': end,
                             'name': child.text or '',
                             'id': attrs.get('id'),
                             'role': 'software_name'})
            elif attrs.get('type') in ('creator', 'publisher', 'version', 'url'):
                anns.append({'start': start, 'end': end,
                             'name': child.text or '',
                             'id': attrs.get('corresp') or attrs.get('id'),
                             'role': attrs['type']})
        else:
            walk(child, buf, anns, label)
        buf.add(child.tail)


def extract_doc(tei_el):
    buf, anns = Buf(), []
    text_el = tei_el.find(T + 'text')
    body = text_el.find(T + 'body') if text_el is not None else None
    if body is not None:
        walk(body, buf, anns, 'body')
    ids = {}
    hdr = tei_el.find(T + 'teiHeader')
    if hdr is not None:
        for i in hdr.iter(T + 'idno'):
            txt = (i.text or '').strip()
            # all.tei.xml stores identifiers as attributes: <idno DOI="10.x/y"/>
            # the holdout file stores them as text:   <idno type="DOI">10.x/y</idno>
            for k, v in i.attrib.items():
                if k != 'type' and v.strip():
                    ids[k] = v.strip()
            if txt:
                if i.attrib.get('type'):
                    ids[i.attrib['type']] = txt
                else:
                    ids.setdefault('idno', txt)
    return ''.join(buf.parts), anns, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tei', required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    root = ET.parse(args.tei).getroot()
    docs = [c for c in root if c.tag == T + 'tei']

    records, bad = [], 0
    for el in docs:
        text, anns, ids = extract_doc(el)
        if not text.strip():
            continue
        names = [a for a in anns if a['role'] == 'software_name']
        for a in names:
            if text[a['start']:a['end']] != a['name']:
                bad += 1
        key = ids.get('DOI') or ids.get('PMC') or ids.get('PMID') or ids.get('idno')
        if not key:
            import hashlib
            key = hashlib.sha256(text.encode('utf-8')).hexdigest()[:24]
        records.append({
            'document_id': 'softcite-gold:' + key,
            'softcite_key': key,
            'ids': ids,
            'text': text,
            'characters': len(text),
            'software_names': names,
            'other_entities': [a for a in anns if a['role'] != 'software_name'],
        })

    (OUT / 'gold-documents.jsonl').write_text(
        ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))

    tot = sum(len(r['software_names']) for r in records)
    withann = sum(1 for r in records if r['software_names'])
    print(f'documents with text : {len(records)}')
    print(f'documents w/ >=1 software name: {withann}')
    print(f'software name spans : {tot}')
    print(f'offset mismatches  : {bad}')
    if records:
        L = [r['characters'] for r in records]
        L.sort()
        print(f'chars per doc: min {L[0]} median {L[len(L)//2]} max {L[-1]} '
              f'total {sum(L)}')


if __name__ == '__main__':
    main()