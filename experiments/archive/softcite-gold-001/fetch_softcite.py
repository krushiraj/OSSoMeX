"""Fetch Softcite responses for the recovered Softcite gold holdout.

Stores one JSON line per document with the verbatim service response so scoring
is reproducible later without the services running.

Usage:
  .venv-scibert/bin/python reports/scibert-v2/softcite-gold-001/scripts/fetch_softcite.py \
      --endpoint http://localhost:8060/process/software \
      --arm softcite-wapiti-0.8.1 \
      --out reports/scibert-v2/softcite-gold-001/raw/softcite-wapiti-0.8.1.jsonl
"""
import argparse
import json
import time
from pathlib import Path

import requests

ROOT = Path('reports/scibert-v2/softcite-gold-001')


def rows(p):
    # Split on '\n' only: splitlines() also breaks on U+2028/U+2029/U+0085.
    return [json.loads(l) for l in Path(p).read_text(encoding='utf-8').split('\n')
            if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--endpoint', required=True)
    ap.add_argument('--arm', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--inputs', default=str(ROOT / 'inputs.jsonl'))
    ap.add_argument('--timeout', type=int, default=300)
    args = ap.parse_args()

    docs = rows(args.inputs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    done = set()
    if out.exists():
        for line in Path(out).read_text(encoding='utf-8').split('\n'):
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)['document_id'])
            except json.JSONDecodeError:
                continue
        print(f'resuming: {len(done)} documents already stored')

    started = time.time()
    for i, d in enumerate(docs, 1):
        if d['document_id'] in done:
            continue
        t0 = time.time()
        status, code, body = 'model_failed', None, None
        try:
            # Softcite's text endpoint expects form-urlencoded `text`, not JSON.
            resp = session.post(args.endpoint, data={'text': d['text']},
                                timeout=args.timeout)
            code = resp.status_code
            if code == 204:
                status = 'no_mentions'
            elif code == 200:
                status = 'success'
                body = resp.json()
            else:
                status = 'model_failed'
                body = {'error': resp.text[:500]}
        except requests.RequestException as exc:
            status = 'model_failed'
            body = {'error': str(exc)}
        with out.open('a') as fh:
            fh.write(json.dumps({
                'arm_id': args.arm, 'document_id': d['document_id'],
                'text_revision': d['text_revision'], 'status': status,
                'status_code': code, 'seconds': round(time.time() - t0, 3),
                'native': body,
            }, ensure_ascii=True) + '\n')
        if i % 10 == 0 or i == len(docs):
            print(f'{args.arm}: {i}/{len(docs)} elapsed {time.time()-started:.0f}s',
                  flush=True)

    print(f'wrote {out}')


if __name__ == '__main__':
    main()