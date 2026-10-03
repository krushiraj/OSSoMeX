# Third-party attribution and artifact scope

OSSoMeX uses SciBERT (`allenai/scibert_scivocab_cased`, revision `ddf0be025f8e432a1870e34811997ba6725bf04a`) and the PyTorch/Transformers ecosystem. Cite Beltagy, Lo and Cohan (2019), [SciBERT: A Pretrained Language Model for Scientific Text](https://aclanthology.org/D19-1371/). The [upstream SciBERT repository](https://github.com/allenai/scibert) distributes its code under Apache-2.0; retain its license and notices with a model distribution.

Softcite, SoMeSci, SoFAIR, OpenAlex-derived snippets and ecosyste.ms-linked papers are separately sourced research inputs. Project-code licensing does not replace their licenses or grant rights to redistribute text, annotations, baseline weights or derived assets. Inspect provenance and source terms for the exact artifact before publication. The early 006 model card records its training parents; 012 uses a different Softcite-derived detector pool.

The source release includes aggregate results and Python experiment helpers. It excludes model tensors, raw papers, annotation stores and raw-output HTML outreach packages. The local model archive is an engineering handoff asset whose public distribution remains pending. Manifest provenance is preserved byte-for-byte, including historical workstation paths; no credentials or source text should be added to the bundle.

Repository organization was informed by the public [SciBERT](https://github.com/allenai/scibert), [GLiNER](https://github.com/urchade/GLiNER) and [SetFit](https://github.com/huggingface/setfit) repositories: short installation/example paths, explicit model access, evaluation scope, contributor guidance and citation metadata. Their source code was not copied for this cleanup.
