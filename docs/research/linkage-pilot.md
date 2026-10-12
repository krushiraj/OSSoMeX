# Paper-to-software linkage pilot

OSSoMeX extracts software names. This pilot measures the next step: linking each paper to the software *entity* it used (a registry record such as a CRAN package, a bio.tools entry or a Wikidata item). Linkage is scored per paper and entity, as precision and recall of `(paper, software)` pairs.

> **Status: provisional.** The gold links were drafted by an AI assistant under the scoped authorization recorded in `approvals.json` and have not yet been fully spot-checked by the maintainer. Treat every number below as an estimate until that review lands. No model weights changed; checkpoints 006 and 012 are used as released.

## Components

- **Registry** (`research registry`): merges papers.ecosyste.ms, bio.tools and Wikidata software items into about 99,000 entities. Records merge only across sources, on a shared package ID, RRID or repository; ambiguous edges are skipped. Optional Wikidata layers (`configs/registry-layers.json`) add programming languages and literature databases.
- **Resolvers** (`research linkage resolve`):
  - `fast-text-match`: exact name, alias or package match; ties go to the most-mentioned entity. This is the baseline.
  - `context`: scores candidates on match type, URLs and package IDs in the text, ecosystem cues (for example "R package"), TF-IDF similarity between the mention context and the entity description, and notability. It abstains below a threshold instead of guessing.
- **Lockbox** (`research linkage lockbox`, `end-to-end`): fresh CC-BY Methods sections from Europe PMC, published after every training snapshot, run through extraction and resolution.

## Results

Paper-level precision and recall (%), 95% bootstrap intervals over papers.

| Evaluation | Pipeline | Precision | Recall |
|---|---|---:|---:|
| SoFAIR held-out, 355 mentions (resolution only) | context | 89.6 (85–94) | 83.6 (77–91) |
| | fast-text-match | 62.0 (56–68) | 63.6 (57–72) |
| Lockbox, 30 fresh Methods sections (end to end) | OSSoMeX 012 + context | 93.8 (90–98) | 70.5 (59–85) |
| | OSSoMeX 006 + context | 80.0 (73–87) | 75.2 (64–89) |
| | OSSoMeX 012 + fast-text-match | 77.0 (71–83) | 63.1 (52–75) |
| | gold mentions + context | 91.7 (85–98) | 81.2 (70–93) |

Registry `2026-10-12-003`. The resolver was frozen (commit `998d80e`) before the held-out set was scored.

## Limitations

- All SoFAIR papers were in detector training, so SoFAIR measures resolution only. The lockbox is the only end-to-end measurement and has 30 papers.
- Recall counts only software present in the registry. Counting unregistered software (vendor, instrument and very new tools), recall falls to roughly 55–57%.
- On the lockbox, 012 finds 73% of gold software names and 006 finds 85%. Extraction, not resolution, limits end-to-end recall.
- Scope calls affect the numbers: model architectures (for example U-Net) are not counted as software; literature databases count only with the optional layer.
- The lockbox has now been used to compare 006 and 012. A new detector should be judged on a fresh lockbox.

## Reproduce

Registry and lockbox data are fetched from public APIs and stay outside Git.

```bash
python -m research registry pull --config configs/registry-layers.json --output data/registry/<run>
python -m research registry build --run data/registry/<run>
python -m research linkage lockbox --config configs/linkage-lockbox-001.json --output data/linkage/<lockbox>
python -m research linkage resolve --tasks <tasks.jsonl> --registry data/registry/<run> --arm context --output <pred.jsonl>
python -m research linkage evaluate --gold <labels.jsonl> --tasks <tasks.jsonl> --predictions <pred.jsonl>
python -m research linkage end-to-end --run data/linkage/<lockbox> --registry data/registry/<run> --gold <gold.jsonl>
```

Resolution takes about 1 ms per paper on one CPU core after a 3-second registry load, using 1.1 GB of memory.
