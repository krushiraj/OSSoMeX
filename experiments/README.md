# Experiment source and reproduction

This directory separates **tracked research code** from ignored generated evidence. `archive/` preserves 39 Python helpers byte-for-byte, with original paths and SHA-256 values in `archive/source-index.json`. Do not run archived files directly: some use their original directory depth to find the repository. These are historical recipes, not maintained public APIs.

## Fresh three-model comparison (012)

```sh
python3 scripts/restore_experiment.py threeway-012-2026-10-02 \
  --output reports/scibert-v2/threeway-reproduction-001
```

Restore only into a new directory. This creates source files, not data, and runs no inference. Before executing:

1. Obtain the local artifact inputs enumerated in restored `run.py` and `prepare_corrected.py`, model bundles and annotation references. The source checkout does not contain them.
2. Start the pinned Softcite Wapiti/SciBERT deployments and verify which extraction engine each loads. Historical endpoints are localhost ports 8060 and 8062; service configs are in `configs/scibert/`.
3. The original quality runner uses MPS and four CPU threads. Adapt explicitly for another device and record the changed environment; do not label it a byte-identical rerun.
4. Install report dependencies using the `reports` extra if rendering HTML/figures. The archived plot helper contains a workstation-specific Python path. Use the portable renderer below on saved scores instead.
5. Read the tracked [original reproduction notes](threeway-reproduction.md). Never overwrite the completed original run or combine overlapping dataset variants.

For portable charts from existing aggregate results:

```sh
uv pip install --python .venv-scibert/bin/python -e '.[reports]'
.venv-scibert/bin/python scripts/render_benchmark_chart.py \
  --scores docs/research/benchmark-012-scores.json \
  --output output/name-f1.png
```

The maintained, configurable `python -m research comparison --help` command group is the starting point for new matched-input experiments. The archive exists to explain and reproduce the particular reported study, including its historical defects and controls.

## Softcite detector adaptation (011 weights used by 012)

`archive/softcite-gold-001/` contains the recovered-data conversion, training, scoring and bundle helpers. In a fresh checkout:

```sh
python3 scripts/restore_experiment.py softcite-gold-001 \
  --output reports/scibert-v2/softcite-gold-001/scripts
```

Obtain the exact Softcite source exports and recovered gold inputs separately. Read each script's CLI before execution; helpers use the original `reports/scibert-v2/softcite-gold-001/` layout. They do not ship those datasets.

**Known harmful historical default:** `train_detector.py --ablation-fraction 0.15` duplicates positive text with empty labels. This is preserved in the archive for reproducibility, not recommended for new training. Explicitly use `--ablation-fraction 0` for the proposed ablation and match optimizer/sample budgets before comparing outcomes. No retraining has been performed for v0.0.1.

## Preservation policy

Archived hashes describe original source bytes, not validation of the algorithms. Fixes belong in a new recipe or the maintained package, with a new experiment identity. New results go under `reports/`; reusable new code belongs under `src/`, `scripts/` or an explicitly documented experiment directory.
