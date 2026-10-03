# Reproducing this comparison

The scripts run from the repository root. Preserve this completed artifact directory. For another experiment, copy only the Python helpers into a new directory under `reports/scibert-v2/` and use that path below.

## Requirements

- The local checkpoints and source datasets referenced by `run.py` and `prepare_corrected.py`.
- `.venv-scibert`: versions in `host-hardware.json`. Inference uses Python3.11, local-only model loading, four PyTorch CPU threads, and Apple MPS for the main OSSoMeX arm.
- Existing Wapiti service on127.0.0.1:8060 and SciBERT service on127.0.0.1:8062. Match engines, weights, image and configuration in `engine-verification.json`, `image-repository-digests.json`, and the two service config files. Do not assume two endpoints imply two different loaded models.
- Charts use Python3.12 and the separate dependencies recorded in `render-versions.json`. This machine already has them under `/tmp/ossomex-article-deps`; `plot_results.py` selects the installed Python3.12 interpreter. Adapt that interpreter path on another machine. HTML rendering uses the Markdown package from the same dependency folder. No rendering dependency is needed for inference or metric computation.

## Execution order

Replace `<new-run>` consistently. These commands intentionally run sequentially; the progress files are also the queue completion signals.

```sh
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv-scibert/bin/python reports/scibert-v2/<new-run>/run.py prepare
HF_HUB_OFFLINE=1 .venv-scibert/bin/python reports/scibert-v2/<new-run>/prepare_corrected.py
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv-scibert/bin/python reports/scibert-v2/<new-run>/run.py ossomex012 > reports/scibert-v2/<new-run>/progress-ossomex012.log 2>&1
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv-scibert/bin/python reports/scibert-v2/<new-run>/run_remaining.py > reports/scibert-v2/<new-run>/queue.log 2>&1
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv-scibert/bin/python reports/scibert-v2/<new-run>/run_whitespace_control.py > reports/scibert-v2/<new-run>/whitespace-control.log 2>&1
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv-scibert/bin/python reports/scibert-v2/<new-run>/finalize.py > reports/scibert-v2/<new-run>/finalization.log 2>&1
```

The finalizer also runs `audit_training_labels.py` and `build_examples.py`. Optional service diagnostic: `probe_offsets.py`. The offset probe calls live services and should be kept outside timing measurements. Hardware/image snapshots and service trace notes were captured separately with read-only system/container commands.

## Scoring existing raw outputs

Run `score.py` on the main directory, with `corrected/` as its argument, and with `whitespace-control/` as its argument. Run `score_whitespace_canonical.py` for the separate posthoc whitespace-only sensitivity, then run `surface_diagnostic.py`, `audit_training_labels.py`, `diagnostics.py`, `alias_diagnostic.py`, `build_error_viewer.py`, `build_examples.py`, `plot_results.py`, and `build_report.py`. `verify_run.py` checks completeness and source identities after all inference and CPU control runs exist.

Important details:

- Read JSONL on LF only (`jsonl_io.py`). Requests’ text/plain decoding is reversed losslessly and decoded as UTF8 before interpreting Softcite JSON.
- Do not repair offsets by nearest-string search. Native failures are retained and unranked; the separate whitespace replay changes the submitted text consistently for all three models and maps positions back exactly.
- Do not combine paragraph/document gold, original/corrected imports, or replay results as independent evidence.
- Raw rows in `whitespace-control/` tagged `reused_fresh_response` are identical-input responses reused from this same experiment, with source hashes. Only changed windows are rerun.
- Scores use reference coverage masks. Unknown labels and unjudged spans are not negative labels.
- Service failure records remain part of the scheduled population. Successful verification means the evidence is complete and structurally consistent, not that all service responses or model predictions are correct.
