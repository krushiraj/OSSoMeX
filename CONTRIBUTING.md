# Contributing

Start with the [README](README.md), [architecture](docs/architecture.md), and [roadmap](ROADMAP.md). Keep model changes and evaluation changes reviewable separately.

## Environment

```sh
uv venv --python 3.11 .venv-scibert
uv pip sync --python .venv-scibert/bin/python requirements-scibert.lock
uv pip install --python .venv-scibert/bin/python --no-deps -e .
```

This is the reference macOS environment, including the model runtime and test packages. The lighter `requirements-scibert-dev.lock` alone is insufficient for the complete suite: several tests import PyTorch and prompt_toolkit directly. On Linux, use the CPU environment in the [README](README.md), then install `.[dev]` into that environment. CI installs `.[dev,ml,interactive]` with CPU PyTorch and records its resolved environment; it is a compatibility check rather than an exact reproduction of the macOS benchmark lock. The `reports` optional extra supplies Markdown, matplotlib and PDF dependencies for historical report generation; it is not required for inference.

## Tests

Run only when appropriate for your change:

```sh
.venv-scibert/bin/python -m pytest tests
```

Browser checks require `.venv-scibert/bin/python -m playwright install chromium`. Seven retained-data checks require licensed research inputs excluded from Git. They explicitly skip when those inputs are absent; synthetic contract/model/UI tests still run. With the retained data installed, those checks execute normally, including hash and lossless-restore assertions. A skipped local-artifact check is not production validation. The CI workflow is manually dispatched. The verified Linux run passed 1,862 tests with seven retained-data skips; the full local run passed 1,869. See [verification](docs/releases/v0.0.12-validation.json).

## Layout

| Path | Purpose |
|---|---|
| `src/research/` | Maintained package and CLI |
| `tests/` | Contract, model, evaluation and UI checks |
| `examples/quickstart*` | Small synthetic inference inputs |
| `models/registry.json` | Model identities, stage hashes and distribution status |
| `configs/` | Recipes; dated configs preserve historical provenance |
| `experiments/` | Recoverable experiment source, separate from generated results |
| `docs/` | Installation, API, model and evaluation guides |
| `scripts/package_models.py` | Verify and package local checkpoint files |
| `reports/`, `data/`, `checkpoints/`, `output/` | Local artifacts, excluded from Git |

## Changes and experiments

- Use commit subjects in `type: description` form, such as `feat: add new checkpoint for model 012`.
- Use relative/configurable paths in new operational recipes. Historical absolute paths identify old sources; they are not portable setup instructions.
- Never change a hash-pinned checkpoint or completed run in place. Use a new output directory and record the parent identity.
- Preserve exact input text, annotation masks, document groups, model identity and native baseline outputs.
- Record software-name precision/recall/F1, version-link quality and runtime separately. Exposed snippets are regressions, not independent tests.
- Treat unknown labels as unknown. Do not train positive text as empty-label negatives.
- Keep weights, private annotations, raw paper text and credentials out of commits. Dataset rights are distinct from project-code terms.
- Update documentation and changelog when changing behavior. Include reproduction commands and limitations in a pull request.

## Releases

Update the version in `pyproject.toml`, `src/research/__init__.py`, citation metadata and changelog together. A release needs an annotated Git tag, model registry, checksums, release notes and a stated verification status. Commit, tag and publication are separate steps; this repository's owner authorizes external publication.
