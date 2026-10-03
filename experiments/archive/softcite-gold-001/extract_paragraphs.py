"""Reconstruct Softcite's own TEI -> text extraction so gold and predicted spans
share one coordinate system.

Mirrors src/main/java/org/grobid/core/sax/TEICorpusSaxHandler.java:

  * every `<p>` becomes one unit;
  * unit text is the trimmed concatenation of all character data inside it,
    passed through clean(): \\n, \\t and the special wide space become U+0020,
    then any run of spaces collapses to a single space;
  * an `<rs>` annotation offset is `currentOffset = getText().length()` evaluated
    at the moment `<rs>` opens, i.e. the cleaned length of everything emitted so
    far in the paragraph, and `end` is INCLUSIVE (start + len - 1).

Softcite reports and scores offsets in this per-paragraph space, so this script
emits one document per paragraph rather than one per article.

Usage:
  .venv-scibert/bin/python reports/scibert-v2/softcite-gold-001/scripts/extract_paragraphs.py \
      --tei /tmp/sc/all.tei.xml --out reports/scibert-v2/softcite-gold-001/paragraphs.jsonl
"""
import argparse
import json
import re
import xml.etree.ElementTree as ET

T = '{http://www.tei-c.org/ns/1.0}'
# Java semantics, matched exactly:
#   String.trim()        removes leading/trailing chars with codepoint <= 0x20
#   \p{Space}            POSIX ASCII space class: [ \t\n\x0B\f\r]  (NOT Unicode)
#   the literal wide space replaced on line 70 is U+2003 EM SPACE
WIDE = ' '
JAVA_SPACE = re.compile('[ \t\n\x0b\f\r]+')


def java_trim(s):
    """Equivalent of java.lang.String.trim()."""
    start, end = 0, len(s)
    while start < end and ord(s[start]) <= 0x20:
        start += 1
    while end > start and ord(s[end - 1]) <= 0x20:
        end -= 1
    return s[start:end]


def clean(text):
    """TEICorpusSaxHandler.clean(): wide/tab/newline to space, then collapse."""
    text = text.replace('\n', ' ').replace('\t', ' ').replace(WIDE, ' ')
    return JAVA_SPACE.sub(' ', text)


def get_text(accumulated):
    """TEICorpusSaxHandler.getText(): clean(accumulator.toString().trim())."""
    return clean(java_trim(accumulated))


def paragraph_units(tei_el):
    """Yield (text, [(start, end_inclusive, name), ...]) per <p>, in document order."""
    text_el = tei_el.find(T + 'text')
    body = text_el.find(T + 'body') if text_el is not None else None
    if body is None:
        return

    buf = []
    anns = []
    para = []
    para_anns = []

    def flush_p():
        if not para:
            return
        para_text = clean(''.join(para))
        if para_text:
            para_anns[:] = [(s, e, n) for (s, e, n) in para_anns]
            yield_out.append((para_text, para_anns))
        para.clear()
        para_anns.clear()

    yield_out = []

    def walk(el):
        buf.append(el.text or '')
        for child in el:
            if child.tag == T + 'rs' and child.get('type') == 'software':
                # offset = cleaned length of everything emitted so far in this <p>
                start = len(clean(''.join(para)))
                name = clean(child.text or '')
                para_anns.append((start, start + len(name) - 1, child.text or ''))
                para.append(child.text or '')
            else:
                walk(child)
            tail = child.tail or ''
            para.append(tail)
        if el.text:
            pass

    walk(body)

    # rebuild paragraph boundaries: <p> elements are direct/indirect children of body
    def collect(el):
        nonlocal buf
        if el.tag == T + 'p':
            para.clear()
            para_anns.clear()
            for chunk in chunks_of_p(el):
                para.append(chunk['text'])
                para_anns.extend(chunk['anns'])
            flush_p()
            return
        for child in el:
            collect(child)

    # simpler: walk each <p> collecting its character data and rs annotations
    def paragraphs(el):
        for child in el.iter(T + 'p'):
            pieces = []
            anns = []
            start = 0

            def rec(node):
                nonlocal start
                pieces.append(node.text or '')
                for c in node:
                    if c.tag == T + 'rs' and c.get('type') == 'software':
                        s = len(get_text(''.join(pieces)))
                        nm = get_text(c.text or '')
                        if nm:
                            anns.append((s, s + len(nm) - 1, c.text or ''))
                        pieces.append(c.text or '')
                    else:
                        rec(c)
                    pieces.append(c.tail or '')

            rec(child)
            t = get_text(''.join(pieces))
            if not t:
                continue
            # TEICorpusSaxHandler computes an annotation offset as getText().length()
            # at the moment <rs> opens, and getText() trims. A span preceded by a
            # space therefore lands one codepoint early. Softcite's own evaluator
            # repairs this (SoftwareExtendedEval): locate the annotation text and
            # snap to it when the offset is off by less than the text length.
            # Apply the same repair here so gold offsets are exact.
            resolved = []
            for hint, _end_incl, raw in anns:
                width = len(raw)
                lo = max(0, hint - width)
                idx = t.find(raw, lo)
                if idx < 0 or idx > hint + width:
                    idx = t.find(raw)
                if idx >= 0:
                    resolved.append((idx, idx + width - 1, raw))
            yield t, resolved

    return list(paragraphs(body))


def doc_ids(tei_el):
    ids = {}
    hdr = tei_el.find(T + 'teiHeader')
    if hdr is not None:
        for i in hdr.iter(T + 'idno'):
            txt = (i.text or '').strip()
            for k, v in i.attrib.items():
                if k != 'type' and v.strip():
                    ids[k] = v.strip()
            if txt:
                ids[i.attrib.get('type') or 'idno'] = txt
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tei', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    root = ET.parse(args.tei).getroot()
    docs = [c for c in root if c.tag == T + 'tei']
    out, total_units, total_anns, bad = [], 0, 0, 0

    for el in docs:
        ids = doc_ids(el)
        key = ids.get('DOI') or ids.get('PMC') or ids.get('PMID') or 'unknown'
        for index, (text, anns) in enumerate(paragraph_units(el)):
            keep = []
            for s, e_incl, raw in anns:
                end = e_incl + 1  # convert inclusive -> half-open
                if 0 <= s < end <= len(text) and text[s:end] == raw:
                    keep.append({'start': s, 'end': end, 'name': raw})
                else:
                    bad += 1
            out.append({'document_id': f'softcite-gold:{key}#p{index}',
                        'article': key, 'paragraph': index, 'text': text,
                        'software_names': keep, 'ids': ids})
            total_units += 1
            total_anns += len(keep)

    with open(args.out, 'w', encoding='utf-8') as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=True) + '\n')

    print(f'paragraph units : {total_units}')
    print(f'gold spans kept : {total_anns}')
    print(f'gold spans bad  : {bad}')
    if out:
        L = sorted(len(r['text']) for r in out)
        print(f'chars/unit: min {L[0]} median {L[len(L)//2]} max {L[-1]} total {sum(L)}')


if __name__ == '__main__':
    main()