"""Create a full-label bundle that reuses every 008 stage except the detector.

The bundle's `stages.detector` entry is repointed at a newly trained detector, with
its manifest sha256 recomputed so the bundle stays self-verifying. Nothing else is
touched: intent, sentiment, alias and boundary policy are identical to 008.

Usage:
  .venv-scibert/bin/python reports/scibert-v2/softcite-gold-001/scripts/make_bundle.py \
      --base checkpoints/scibert-full-label-008 \
      --detector checkpoints/scibert-detector-010-softcite \
      --output checkpoints/scibert-full-label-010
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def manifest_sha256(directory):
    return hashlib.sha256((Path(directory) / 'manifest.json').read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--detector', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    base, out = Path(args.base), Path(args.output)
    if out.exists():
        raise SystemExit(f'refusing to overwrite existing {out}')
    manifest = json.loads((base / 'manifest.json').read_text())

    previous = manifest['stages']['detector']
    detector_manifest = json.loads((Path(args.detector) / 'manifest.json').read_text())
    # `support` must describe the *new* detector, not the one being replaced:
    # load_pipeline_manifest cross-checks it against the checkpoint.
    manifest['stages']['detector'] = {
        **previous,
        'path': f'../{Path(args.detector).name}',
        'manifest_sha256': manifest_sha256(args.detector),
        'support': detector_manifest['support'],
    }
    # Provenance of the substitution, recorded in the bundle itself.
    manifest['stages']['detector']['replaces'] = {
        'path': previous['path'], 'manifest_sha256': previous['manifest_sha256'],
    }
    manifest['status'] = 'trained_experimental'
    manifest['quality_evaluated'] = False
    manifest['detector_adaptation'] = {
        'purpose': 'benchmark_adaptation_on_softcite_published_gold',
        'base_bundle': str(base),
        'reused_stages': [k for k in manifest['stages'] if k != 'detector'],
        'changed_stage': 'detector',
        'detector_training_provenance': detector_manifest.get('provenance'),
    }

    out.mkdir(parents=True)
    shutil.copy2(base / 'manifest.json', out / 'manifest.json')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'wrote {out}')
    print(f'  detector -> {manifest["stages"]["detector"]["path"]}')
    print(f'  reused   -> {", ".join(manifest["detector_adaptation"]["reused_stages"])}')


if __name__ == '__main__':
    main()