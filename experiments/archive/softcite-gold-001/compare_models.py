"""Head-to-head on arbitrary text: Softcite WAPITI, Softcite SciBERT, and one of ours.

All three arms see byte-identical text. Softcite goes over HTTP (form-urlencoded `text`),
ours runs in-process. Timings are wall-clock around each call, which is what a user
actually waits for, so the numbers stay comparable even though the transports differ.

Two practical details this script handles so the UI does not have to:

1. Softcite's GROBID backend dies with HTTP 500 on inputs over roughly 21,200
   characters, verified on both the WAPITI (8060) and SciBERT (8062) containers. Long
   input is therefore split on paragraph boundaries and offsets are shifted back after
   each piece is scored. Chunk size is set below that ceiling with margin.
2. Softcite sometimes returns spans whose offsets do not match the submitted text.
   Those are counted and dropped rather than silently rendering a highlight in the
   wrong place.

Usage:
  .venv-scibert/bin/python compare_models.py --bundle checkpoints/scibert-full-label-011 \
      --text paper.txt --label my-paper
"""
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, 'src')

SOFTCITE_ARMS = [
    ('softcite-wapiti', 8060),
    ('softcite-scibert', 8062),
]

# Softcite returns 500 past ~21,200 characters; 18,000 leaves headroom.
CHUNK_CHARS = 18000


def chunk_text(text, limit=CHUNK_CHARS):
    """Split on blank lines, packing paragraphs up to `limit` characters.

    Returns [(start_offset, piece)] so spans can be shifted back to absolute
    positions. A single oversized paragraph is split on line breaks. Chunk
    boundaries are always breaks in the original text, so concatenating the
    pieces reproduces `text` exactly.
    """
    units = []
    for block in re.split(r'(?<=\n)', text):
        while len(block) > limit:
            cut = block.rfind('\n', 0, limit)
            if cut <= 0:
                cut = limit
            units.append(block[:cut])
            block = block[cut:]
        units.append(block)

    pieces, buf, pos = [], [], 0
    for unit in units:
        if not buf:
            start = pos
        current = sum(len(u) for u in buf)
        if current + len(unit) > limit and current:
            pieces.append((start, ''.join(buf)))
            start, buf = pos, []
        buf.append(unit)
        pos += len(unit)
    if buf:
        pieces.append((start, ''.join(buf)))

    assert ''.join(p for _, p in pieces) == text, 'chunking must preserve text exactly'
    return [(s, p) for s, p in pieces if p.strip()]


def call_softcite_chunk(port, text):
    body = urllib.parse.urlencode({'text': text}).encode()
    req = urllib.request.Request(
        f'http://localhost:{port}/service/processSoftwareText', data=body,
        headers={'Content-Type': 'application/x-www-form-urlencoded'})
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.loads(resp.read())


def sanitize_for_softcite(text):
    """Replace control characters with spaces, 1:1 so offsets survive.

    pdftotext emits control bytes for math formulas (U+0001, U+0010, U+0011 in the
    ViTeX paper). Softcite's GROBID backend returns HTTP 500 on those, which silently
    cost half the document. Substituting a space keeps every offset valid.
    """
    return ''.join(c if (ord(c) >= 32 or c in '\n\t\r') else ' ' for c in text)


def softcite_mentions(payload):
    """Pull mentions out of a Softcite response.

    The live service returns `mentions` at the top level. Files written by
    fetch_softcite.py wrap the whole body under `native`, so accept either.
    """
    if isinstance(payload.get('mentions'), list):
        return payload['mentions']
    return (payload.get('native') or {}).get('mentions', [])


def call_softcite(port, text):
    spans, rejected, failures = [], 0, []
    pieces = chunk_text(text)
    for start, piece in pieces:
        try:
            payload = call_softcite_chunk(port, sanitize_for_softcite(piece))
        except urllib.error.HTTPError as exc:
            failures.append({'offset': start, 'error': f'HTTP {exc.code}'})
            continue
        for m in softcite_mentions(payload):
            n = m.get('software-name') or {}
            s, e = n.get('offsetStart'), n.get('offsetEnd')
            if not isinstance(s, int) or not isinstance(e, int) or not (0 <= s < e <= len(piece)):
                rejected += 1
                continue
            spans.append({
                'start': start + s, 'end': start + e,
                'text': piece[s:e], 'rawForm': n.get('rawForm', ''),
                'offset_valid': True,
            })
    return {'spans': spans, 'rejected_offsets': rejected, 'chunk_failures': failures,
            'chunks': len(pieces)}


def call_ours(bundle, text, pipeline=None):
    """Score with one of our bundles.

    `pipeline` lets a long-lived caller (the UI server) reuse a loaded model instead of
    paying the load cost on every request.
    """
    if pipeline is None:
        from research.training.full_label import FullLabelPipeline
        pipeline = FullLabelPipeline(bundle)
    t0 = time.perf_counter()
    out = pipeline.predict({'text': text, 'document_id': 'tmp'})
    elapsed = time.perf_counter() - t0
    spans = []
    for occ in out['occurrences']:
        s, e = occ['name_span']['start'], occ['name_span']['end']
        spans.append({
            'start': s, 'end': e, 'text': text[s:e],
            'name': occ.get('name'),
            'intents': occ.get('intents', []),
            'versions': [v['text'] for v in occ.get('version_links', [])],
            'offset_valid': True,
        })
    return {'seconds': elapsed, 'spans': spans, 'prediction': out}


def main():
    global CHUNK_CHARS
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', default='checkpoints/scibert-full-label-011')
    ap.add_argument('--text', required=True, help='file with the text, or - for stdin')
    ap.add_argument('--label', default='input')
    ap.add_argument('--ports', default='8060,8062')
    ap.add_argument('--chunk-chars', type=int, default=CHUNK_CHARS)
    ap.add_argument('--output')
    args = ap.parse_args()

    CHUNK_CHARS = args.chunk_chars

    text = sys.stdin.read() if args.text == '-' else Path(args.text).read_text(errors='replace')
    ports = [int(p) for p in args.ports.split(',')]
    chunks = len(chunk_text(text))
    print(f'input {args.label}: {len(text):,} characters, {chunks} chunk(s) for Softcite\n')

    results = {}
    for (default_label, default_port), port in zip(SOFTCITE_ARMS, ports):
        label = default_label if port == default_port else f'softcite-{port}'
        t0 = time.perf_counter()
        try:
            r = call_softcite(port, text)
            r['seconds'] = time.perf_counter() - t0
        except Exception as exc:
            print(f'{label}: FAILED ({exc})')
            results[label] = {'error': str(exc), 'spans': [], 'seconds': None}
            continue
        results[label] = r
        note = ''
        if r['rejected_offsets']:
            note += f'  ({r["rejected_offsets"]} rejected offsets)'
        if r['chunk_failures']:
            note += f'  ({len(r["chunk_failures"])} chunk failures)'
        print(f'{label:28} {r["seconds"]:8.2f}s  {len(r["spans"]):4} spans{note}')

    ours_label = f'ossomex ({Path(args.bundle).name})'
    ours = call_ours(args.bundle, text)
    results[ours_label] = ours
    print(f'{ours_label:28} {ours["seconds"]:8.2f}s  {len(ours["spans"]):4} spans')

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps({
            'label': args.label,
            'characters': len(text),
            'arms': {k: {'seconds': v.get('seconds'), 'count': len(v['spans']),
                         'rejected_offsets': v.get('rejected_offsets', 0),
                         'spans': v['spans']}
                     for k, v in results.items()},
        }, indent=2) + '\n')
        print(f'\nwrote {args.output}')


if __name__ == '__main__':
    main()