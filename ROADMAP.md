# Roadmap

## Quality

- Compare 006 and 012 on one frozen evaluation before promoting a default.
- Ablate contradictory empty-label copies with matched optimizer/sample budgets.
- Mix source-balanced, reviewed annotations; preserve unknown-field masks.
- Improve boundaries and version association using competing-link examples and real negatives.
- Expand sentiment and alias supervision across independent papers.

## Evaluation

- Create a human-reviewed, paper-grouped cross-domain lockbox.
- Keep inspected examples in development/regression sets.
- Report per-domain precision/recall, uncertainty and annotation coverage.
- Evaluate registry/project entity resolution separately from extraction.

## Runtime and distribution

- Batch linker and alias candidates; measure candidate recall before pruning.
- Measure same-hardware CPU/GPU latency, failures, peak memory and concurrency.
- Finalize checkpoint reuse/redistribution terms; the source and model repositories are now publicly accessible.
- Validate actual checkpoint inference on an independent machine. Clean local CPU inference and hosted Linux contributor tests now pass; automatic CI triggers remain a maintainer choice.
