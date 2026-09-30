"""Benchmark warmed full-label checkpoints and the resident Softcite service."""

import argparse
import json
from pathlib import Path

from research.comparison.pipeline_timing import run_benchmark


def bounded_repeats(value):
    number = int(value)
    if not 1 <= number <= 20:
        raise argparse.ArgumentTypeError('repeats must be between 1 and 20')
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Frozen document JSONL')
    parser.add_argument('--checkpoint', action='append', default=[], metavar='NAME=PATH',
                        help='Named full-label checkpoint; may be repeated')
    parser.add_argument('--softcite-config', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'mps'), required=True)
    parser.add_argument('--repeats', type=bounded_repeats, default=3)
    parser.add_argument('--batch-size', type=int, choices=(1,), default=1)
    parser.add_argument('--output', type=Path, required=True, help='New immutable directory')
    args = parser.parse_args(argv)
    checkpoints = {}
    for value in args.checkpoint:
        name, separator, path = value.partition('=')
        if not separator or not name.strip() or not path.strip() or name in checkpoints:
            parser.error('--checkpoint requires distinct NAME=PATH values')
        checkpoints[name] = Path(path)
    manifest = run_benchmark(args.input, checkpoints, args.softcite_config, args.output,
                             device=args.device, repeats=args.repeats, batch_size=args.batch_size)
    print(json.dumps({'output': str(args.output), 'arms': [arm['arm_id'] for arm in manifest['arms']],
                      'documents': manifest['document_count'], 'repeats': manifest['repeats']}))
    return int(any(arm['summary']['failed'] or arm['summary']['not_attempted'] or
                   arm['warmup_status'] not in ('success', 'no_mentions') for arm in manifest['arms']))


if __name__ == '__main__':
    raise SystemExit(main())
