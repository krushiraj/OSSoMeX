# Inference and output

Install the environment and complete model bundle using the [quick start](../README.md) first. Run commands from the repository root.

```sh
.venv-scibert/bin/python -m research full-label predict --help
.venv-scibert/bin/python -m research full-label predict \
  --model model-assets/checkpoints/scibert-full-label-006 --device cpu \
  --input examples/quickstart.jsonl --output output/example-predictions.jsonl
```

Create the output parent first (`mkdir -p output`). Output files are immutable: choose a new filename for another run. The text example is synthetic and is not training data or benchmark gold.

## Python

```python
from pathlib import Path
from research.training.full_label import FullLabelPipeline

pipeline = FullLabelPipeline(Path("model-assets/checkpoints/scibert-full-label-006"), device="cpu")
result = pipeline.predict({"document_id": "example", "text": "We used NumPy 1.24.3."})
print(result["public_rows"])
```

Load the pipeline once for many documents. The full result retains fields that may not appear in `public_rows`; an empty public-row list alone is not proof that inference succeeded.

## Interactive demo

After the Hugging Face download in the README, start a resident session:

```sh
.venv-scibert/bin/python -m research full-label interactive \
  --model model-assets/checkpoints/scibert-full-label-006 --device cpu --format table
```

Wait for `text>`; the first prompt appears after the five stages load. Submit each example separately:

```text
We used NumPy 1.24.3 for numerical analysis.
The simulations were performed in MATLAB R2023b.
We used Python 3.11 and pandas 2.0.3 to process the data.
The samples were stored at room temperature.
```

These are synthetic walkthrough inputs, not benchmark references. The [recorded GIF](assets/terminal-demo.gif), [plain-text outputs](assets/terminal-demo.txt) and [recording metadata](assets/terminal-demo.json) show what checkpoint 006 actually produced. The recording uses the local `checkpoints/` layout; downloaded models under `model-assets/checkpoints/` work the same way when the complete sibling directories are preserved. Results can differ with another checkpoint.

| Key | Action |
|---|---|
| Enter | Submit the current draft |
| Escape, then Enter | Insert a newline |
| Ctrl+C | Clear the draft |
| Ctrl+D | Exit |

For checkpoint 012, replace the final model directory with `scibert-full-label-012`. On a compatible Mac, change `--device cpu` to `--device mps`. Keep the same exact inputs when comparing models. Use `predict --stdin` for files whose original whitespace and offsets must be preserved.

## Troubleshooting

| Symptom | What to do |
|---|---|
| `No module named research` | Run the editable install from the README using the same virtual environment as the command. |
| Missing manifest or checkpoint file | Download the full bundle; check whether your model is under `model-assets/checkpoints/` or `checkpoints/`. |
| Checkpoint hash mismatch | Download the original files again. Keep stage manifests, weights and tokenizers together; do not edit their contents. |
| Interactive input requires a terminal | Run in a terminal, or use `full-label predict --stdin` for redirected input. |
| Output file already exists | Choose a new output filename; saved predictions are immutable. |
| Slow first prompt | All five stages load before the prompt. Leave the interactive session open to reuse them. |
| Empty result or `!` marker | Check the JSON status and review reasons. An empty successful prediction can still miss a real software mention. |

## Read the fields correctly

- `status`: distinguish success/no mentions, partial output and failure. Failures must not become negative examples.
- `public_rows`: name, version, context sentence, intent and sentiment. Several rows can describe one name with multiple proposed versions.
- `field_predictions` and `detector_diagnostics`: exact offsets, field status, candidate spans and scores.
- `alias_predictions`: pair-level alias proposals; aliases are not reliable resolved identities.
- `review_required` and `review_reasons`: experimental predictions need review. The table's `!` is a review flag, not a statement that the name is definitely wrong.

Offsets are half-open Unicode codepoint ranges into the exact submitted string. Versions can be null; unresolved fields are not inferred negatives. Scores are uncalibrated. Intent/sentiment labels are predictions and must not be presented as verified facts about authors.

CPU is portable; MPS is a local Mac option. Dense lists can trigger expensive pairwise inference. No production latency or concurrency guarantee is offered. Checkpoints 006 and 012 leave the optional boundary-repair policy disabled.
