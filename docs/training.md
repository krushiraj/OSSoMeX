# Training and evaluation

Use the pinned contributor environment and run from the repository root. Training requires the pinned base SciBERT files, correctly licensed source text, annotation exports and data manifests. These are not included in Git. A trained inference bundle alone is not a training dataset.

## Maintained full-label workflow

```sh
.venv-scibert/bin/python -m research full-label prepare --help
.venv-scibert/bin/python -m research full-label train --help
.venv-scibert/bin/python -m research comparison --help
.venv-scibert/bin/python -m research annotate --help
```

Start from `configs/scibert/full-label-poc-001.json` for the training recipe. The data config passed to `full-label prepare` identifies manifests and hashes, not just free text. Use `research data --help` and `research annotate --help` for acquisition and annotation commands. Field contracts are in `schemas/scibert-v2/`; synthetic contract fixtures are in `examples/scibert-contract-examples.json` and `examples/scibert-alias-examples.json`.

```sh
# Supply your own reviewed data config and new output paths.
.venv-scibert/bin/python -m research full-label prepare \
  --config /path/to/reviewed-data-config.json --output data/my-training-001
.venv-scibert/bin/python -m research full-label train \
  --stage detector --data data/my-training-001 \
  --config configs/scibert/full-label-poc-001.json \
  --output checkpoints/my-detector-001 --device cpu
```

Train `linker`, `intent`, `sentiment` and `alias` separately with sufficient field-specific annotation support. `research.training.full_label_train.publish_pipeline(detector, stages, output)` assembles verified stage manifests; inspect its signature and prerequisites before publishing a bundle. Unsupported heads must remain explicitly unavailable rather than randomly initialized and described as trained.

## Reproduce existing research

The [experiment source archive](../experiments/README.md) contains the 011 Softcite adaptation and the fresh 012 three-model comparison. Their required datasets and baseline services must be obtained separately. Reproduction requires the original source hashes, labels, model identity, field projection, input-window policy and hardware record, not merely the same checkpoint number.

Historical data configs contain absolute workstation paths as provenance. Create a new reviewed config using your own paths and correct hashes; do not search-and-replace completed manifests or label the result the same dataset snapshot.

## Evaluation rules

- Split by paper and audit duplicates before training; keep a new human-reviewed lockbox untouched.
- Keep local inspected snippets for regression. A successful fix on those snippets does not prove generalization.
- Measure exact names, version spans, name-version links and rare attributes separately.
- Include genuine negatives; unannotated regions must not become negative labels.
- Compare native baseline outputs and explicitly disclose schema projection and offset normalization.
- Save failures as failures and report coverage. Do not remove failed requests from a scheduled population.
- Keep runtime measurements separate from quality; record hardware and dense-window tail latency.

[Current results](research/benchmark-012.md) and [roadmap](../ROADMAP.md) explain the next experiments. v0.0.1 does not fix training defects or promote a newly trained model.
