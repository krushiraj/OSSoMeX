"""Restore frozen experiment Python helpers into a new local reports directory."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=('threeway-012-2026-10-02', 'softcite-gold-001'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    report_root = ROOT / 'reports/scibert-v2'
    if args.experiment == 'threeway-012-2026-10-02' and output.parent != report_root:
        raise ValueError('threeway helpers require reports/scibert-v2/<new-run> layout')
    if args.experiment == 'softcite-gold-001' and output != report_root / 'softcite-gold-001/scripts':
        raise ValueError('the historical trainer requires reports/scibert-v2/softcite-gold-001/scripts')
    if output.exists():
        raise FileExistsError('use a new directory; never overwrite completed evidence')
    source = ROOT / 'experiments/archive' / args.experiment
    index = json.loads((ROOT / 'experiments/archive/source-index.json').read_bytes())
    payloads = {}
    for path in sorted(source.glob('*.py')):
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != index[path.relative_to(ROOT).as_posix()]['sha256']:
            raise ValueError(f'archive source changed: {path.name}')
        payloads[path.name] = payload
    output.mkdir(parents=True)
    for name, payload in payloads.items():
        (output / name).write_bytes(payload)
    print(f'Restored {len(payloads)} source files. No inference, download or training was started.')


if __name__ == '__main__':
    main()
