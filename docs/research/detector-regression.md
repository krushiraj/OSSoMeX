# 006/012 context-sensitive detector regression

Diagnostic: 4 October 2026. No model changes or retraining.

On an exposed benchmark passage containing four MATLAB mentions, Java and TRNSYS, 006 recovers five genuine name occurrences plus spurious PVTType, and incorrectly links 27 as a version. 012 returns no mentions. Both greedy and constrained decoding return no spans for 012. Adding sentence spaces does not help. Both detectors recognize the three names in a simple usage sentence.

012's token logits favor O (outside an entity): first MATLAB 81.8%, Java 81.5%; these softmax values are uncalibrated. The failure occurs before downstream linking/attributes. Changing the decoder alone does not recover this case.

The four downstream head manifest hashes match. 012 has different detector weights, trained from base SciBERT on Softcite records rather than continued from 006. Both use constrained wordpiece decoding without boundary repair. The checkpoint number therefore does not imply preservation of earlier behavior.

Contradictory empty-label positive copies, context distribution and final-epoch selection are candidate training contributors. Their individual causal effects require controlled retraining, not inference-only diagnosis. The diagnostic ran on CPU and reproduced the recorded 006 and 012 MPS results; no fresh MPS run was performed.

## Next action

Compare both checkpoints on the frozen benchmark. Ablate empty-label copies; then add source-balanced contexts and boundary/linking supervision. Preserve a new untouched paper-grouped evaluation set. This case is now a regression example, not an independent test.

Exact input, token scores, full predictions and script are local at `reports/scibert-v2/matlab-006-vs-012-2026-10-04/`. The source release publishes this aggregate diagnosis, not the third-party passage text.
