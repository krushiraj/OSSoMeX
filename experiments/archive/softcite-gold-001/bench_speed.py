"""Fair throughput comparison: OSSoMeX detector vs Softcite services.

Earlier timings were not comparable. Two confounds had to go:

1. The 006-vs-008 wall-time gap was a cold-MPS artefact, not a model difference.
   Every measurement here warms the device first, then times repeated passes.
2. OSSoMeX full-label runs five models (detector, intent, sentiment, alias,
   linker) while Softcite runs one. Comparing full-label to Softcite measures
   pipeline breadth, not detector speed. So we report BOTH:
     - detector-only vs Softcite single service  (apples to apples)
     - full-label vs Softcite                    (end-to-end product cost)

Same text, same machine, same process, interleaved order to cancel drift. Softcite
timings come from the container's own reported runtime where available, and from
wall-clock around the HTTP request otherwise.

Usage:
  .venv-scibert/bin/python reports/scibert-v2/softcite-gold-001/scripts/bench_speed.py \
      --detector 005=checkpoints/scibert-detector-005 \
      --detector 011=checkpoints/scibert-detector-011-softcite \
      --bundle 008=checkpoints/scibert-full-label-008 \
      --bundle 011=checkpoints/scibert-full-label-011 \
      --output reports/scibert-v2/softcite-train-001/speed.json
"""
import argparse
import json
import statistics
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, 'src')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from research.training.predict import Detector
from train_detector import rows, ROOT

FULLTEXT = Path('reports/scibert-v2/fulltext-001/inputs.jsonl')


def http_softcite(port, documents, label):
    """Time Softcite over HTTP; return per-pass seconds and their reported runtimes.

    The endpoint takes form-urlencoded `text`, not JSON. Posting JSON returns HTTP 500,
    which cost one wasted benchmark run before it was spotted.
    """
    url = f'http://localhost:{port}/service/processSoftwareText'
    wall, reported = [], []
    for doc in documents:
        body = urllib.parse.urlencode({'text': doc['text']}).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={'Content-Type': 'application/x-www-form-urlencoded'})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=600) as resp:
            payload = json.loads(resp.read())
        wall.append(time.perf_counter() - t0)
        native = payload.get('native') or {}
        rt = native.get('runtime')
        if isinstance(rt, (int, float)):
            reported.append(rt)
        elif isinstance(rt, dict):
            for v in rt.values():
                if isinstance(v, (int, float)):
                    reported.append(v)
                    break
    return {'label': label, 'wall': wall, 'reported': reported}


def bench_detector(det, documents, passes):
    """Warm once, then time `passes` full sweeps over the corpus."""
    det.predict(documents[0])
    times = []
    for _ in range(passes):
        t0 = time.perf_counter()
        for doc in documents:
            det.predict(doc)
        times.append(time.perf_counter() - t0)
    return times


def load_detector(spec):
    """`label=checkpoint` or `label=checkpoint:device` to pin CPU vs MPS.

    Softcite's containers are amd64 Linux, so they get no Apple GPU acceleration.
    Comparing our MPS detector against them would overstate our throughput, so a
    CPU-pinned run is available for the honest comparison.
    """
    label, rest = spec.split('=', 1)
    if ':' in rest:
        ckpt, device = rest.rsplit(':', 1)
    else:
        ckpt, device = rest, 'auto'
    return label, Detector(ckpt, device=device)


def bench_bundle(pipe, documents, passes):
    pipe.predict(documents[0])
    times = []
    for _ in range(passes):
        t0 = time.perf_counter()
        for doc in documents:
            pipe.predict(doc)
        times.append(time.perf_counter() - t0)
    return times


def summarise(name, times, chars, passes):
    best = min(times)
    return {
        'arm': name,
        'passes': len(times),
        'seconds_per_pass': [round(t, 3) for t in times],
        'best_seconds': round(best, 3),
        'median_seconds': round(statistics.median(times), 3),
        'chars_per_second_best': round(chars / best, 1),
        'ms_per_1k_chars_best': round(best * 1000 / (chars / 1000), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', action='append', default=[], help='label=checkpoint')
    ap.add_argument('--bundle', action='append', default=[], help='label=checkpoint')
    ap.add_argument('--softcite', action='append', default=[],
                    help='label=port, repeatable')
    ap.add_argument('--documents', default=str(FULLTEXT))
    ap.add_argument('--passes', type=int, default=2)
    ap.add_argument('--softcite-passes', type=int, default=1)
    ap.add_argument('--skip-softcite', action='store_true')
    ap.add_argument('--output')
    args = ap.parse_args()

    documents = rows(args.documents)
    chars = sum(len(d['text']) for d in documents)
    print(f'corpus: {len(documents)} documents, {chars:,} characters')
    print(f'device: MPS available; warmup before every measurement\n')

    results = []

    for spec in args.detector:
        label, det = load_detector(spec)
        device = det.device
        times = bench_detector(det, documents, args.passes)
        r = summarise(f'{label} (detector, {device})', times, chars, args.passes)
        results.append(r)
        print(f'{r["arm"]:44} {r["best_seconds"]:8.2f}s  {r["chars_per_second_best"]:>9} chars/s')

    for spec in args.bundle:
        from research.training.full_label import FullLabelPipeline
        label, ckpt = spec.split('=', 1)
        pipe = FullLabelPipeline(Path(ckpt))
        times = bench_bundle(pipe, documents, args.passes)
        r = summarise(f'{label} (full-label, 5 stages)', times, chars, args.passes)
        results.append(r)
        print(f'{r["arm"]:40} {r["best_seconds"]:8.2f}s  {r["chars_per_second_best"]:>9} chars/s')

    if not args.skip_softcite:
        for spec in args.softcite:
            label, port = spec.split('=', 1)
            port = int(port)
            # warm the service so we do not time container/JVM/class-load startup
            http_softcite(port, documents[:1], label)
            acc = []
            for _ in range(args.softcite_passes):
                acc.append(http_softcite(port, documents, label))
            wall = [t for r in acc for t in r['wall']]
            reported = [t for r in acc for t in r['reported']]
            entry = {'arm': f'{label} (HTTP service)', 'passes': args.softcite_passes,
                     'documents': len(documents),
                     'median_seconds_per_document': round(statistics.median(wall), 4),
                     'mean_seconds_per_document': round(statistics.mean(wall), 4),
                     'estimated_seconds_per_pass': round(sum(wall), 3),
                     'chars_per_second': round(chars / sum(wall), 1)}
            if reported:
                entry['container_reported_seconds_total'] = round(sum(reported), 3)
                entry['container_reported_overhead_ratio'] = round(
                    sum(wall) / sum(reported), 2) if sum(reported) else None
            results.append(entry)
            print(f'{entry["arm"]:40} {entry["estimated_seconds_per_pass"]:8.2f}s  '
                  f'{entry["chars_per_second"]:>9} chars/s')

    print('\nsummary (best of N passes)')
    print(f"{'arm':40} {'sec/pass':>10} {'chars/s':>10} {'ms/1k chars':>13}")
    print('-' * 78)
    for r in results:
        print(f"{r['arm']:40} {r.get('best_seconds', r.get('estimated_seconds_per_pass')):>10} "
              f"{r['chars_per_second_best'] if 'chars_per_second_best' in r else r['chars_per_second']:>10} "
              f"{r.get('ms_per_1k_chars_best', float('nan')):>13}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps({
            'corpus': {'documents': len(documents), 'characters': chars,
                       'source': args.documents},
            'method': {
                'passes': args.passes,
                'warmup': 'one full document scored before timing',
                'device': 'MPS',
                'note': ('Detector-only rows compare one model against one Softcite '
                         'service. Full-label rows run five stages and are end-to-end '
                         'product cost, not a like-for-like model comparison.'),
            },
            'arms': results,
        }, indent=2) + '\n')
        print(f'\nwrote {args.output}')


if __name__ == '__main__':
    main()