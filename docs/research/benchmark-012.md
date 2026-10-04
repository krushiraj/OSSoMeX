# Fresh 012 comparison

Run: 2 October 2026. Article analysis updated 3 October. Source release: v0.0.1.

## Exact software-name scores

Same frozen 480-token windows, 64-token overlap. Exact offsets; coverage masks exclude unjudged regions. Softcite primary projection includes explicit software names and language fields, excluding implicit names. These are matched-window evaluations, not whole-document pipeline timings.

| Dataset | Model | Precision % | Recall % | F1 % |
|---|---|---:|---:|---:|
| fresh30 | OSSoMeX 012 | 89.5 | 66.7 | 76.4 |
| fresh30 | Softcite Wapiti | 92.3 | 23.5 | 37.5 |
| fresh30 | Softcite SciBERT | 86.5 | 62.7 | 72.7 |
| exposed50 | OSSoMeX 012 | 86.2 | 42.1 | 56.6 |
| exposed50 | Softcite Wapiti | 90.7 | 29.3 | 44.3 |
| exposed50 | Softcite SciBERT | 70.7 | 30.8 | 42.9 |
| reserve27 | OSSoMeX 012 | 88.9 | 43.8 | 58.7 |
| reserve27 | Softcite Wapiti | 100.0 | 42.5 | 59.6 |
| reserve27 | Softcite SciBERT | 88.6 | 42.5 | 57.4 |
| regression7 | OSSoMeX 012 | 85.7 | 58.8 | 69.8 |
| regression7 | Softcite Wapiti | 75.0 | 17.6 | 28.6 |
| regression7 | Softcite SciBERT | 83.3 | 39.2 | 53.3 |
| softcite-gold-paragraphs | OSSoMeX 012 | 78.4 | 72.2 | 75.1 |
| softcite-gold-paragraphs | Softcite Wapiti | 93.7 | 92.7 | 93.2 |
| softcite-gold-paragraphs | Softcite SciBERT | 91.5 | 92.7 | 92.1 |

## Confidence and scope

The local OpenAlex/regression references are agent-provisional and development-exposed. On fresh30, 012 minus SciBERT is +3.7 F1 points, 95% paired-bootstrap interval -11.7 to +20.5. Gold is -16.9 points, paper-clustered interval -22.6 to -12.0. The 461 positive paragraphs come from 233 articles, not the entire published holdout. Paragraph/document views overlap; do not pool them. Baseline training membership was not independently audited.

Only 012 was freshly evaluated among OSSoMeX checkpoints. Its name does not establish that it is better than 006.

## Other tasks

| Measure | 012 | Wapiti | SciBERT |
|---|---:|---:|---:|
| Corrected SoMeSci10 name recall, 145 references | 71.7% | 36.6% | 51.0% |
| Corrected SoFAIR6 name recall, 120 references | 75.8% | 6.7% | 38.3% |
| Regression7 name-version link F1, 28 reference links | 79.2% | 18.2% | 27.8% |
| 20-paper catalogue coverage, 666 links | 55.1% | 5.3% | 40.1% |

SoMeSci/SoFAIR are previously inspected small subsets with unaudited negative coverage: recall only, not precision/F1. SoFAIR uses the disclosed posthoc whitespace-only normalization sensitivity. Catalogue matches are not mention recall or validated project resolution; the selected papers are heavily biomedical/bioinformatics.

Fresh30 sentiment observed macro-F1 is 20.5%, excluding absent classes. All three selected positive alias pairs were missed. These attributes are not established advantages.

## Runtime

90 warm observations per arm: 012 full pipeline 16.44 snippets/s on Apple MPS (median 37.2 ms, P95 166.3 ms); Wapiti 3.63/s (5.0 ms, 1352.0 ms); SciBERT 1.28/s (625.9 ms, 1882.9 ms). Baselines ran in amd64 CPU containers on an ARM host. A native-CPU 012 control achieved 11.37/s. Hardware/runtime differences prevent architecture-level speed claims. Wapiti had the fastest median and more empty-name responses. A dense list took 012 about 90 seconds because linking/alias candidates were classified individually.

## Bugs and improvement priorities

- Legacy SoMeSci/SoFAIR import offsets were invalid; repaired imports remain partially annotated.
- Softcite response encoding and leading-whitespace offsets needed explicit integration controls. No fuzzy matching was used to hide offset failures.
- 011/012 detector training includes contradictory positive-text/empty-label copies. Ablate this under matched training effort.
- The 012 detector can reject familiar software in certain contexts; [diagnostic](detector-regression.md).
- Batch and constrain pair candidates while measuring true-link candidate recall.

## Evidence and reproduction

[Aggregate score records](benchmark-012-scores.json) are tracked without source text. The [experiment archive](../../experiments/README.md) contains the training and comparison Python helpers. Full raw responses, source documents, manifests, bootstrap receipts and timing controls remain in the local `reports/scibert-v2/threeway-012-2026-10-02/` artifact directory and must be obtained separately for complete reproduction. The source release alone is not the full evidence dataset.

[Roadmap](../../ROADMAP.md) describes the independent evaluation and training work needed before stronger claims.
