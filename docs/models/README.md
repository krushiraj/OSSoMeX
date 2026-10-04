# Model access and installation

Release tag **v0.0.12** separates source code from model assets; the Python package and archive filenames retain version 0.0.1. The model repository is [krushiraj/OSSoMeX on Hugging Face](https://huggingface.co/krushiraj/OSSoMeX). It is public and can be downloaded without signing in. Checkpoint reuse/redistribution terms remain separate from the project-code license. The source code is Apache-2.0; this does not automatically assign that license to every model or dataset artifact.

## Download from Hugging Face

After installing the source environment:

```sh
.venv-scibert/bin/hf download krushiraj/OSSoMeX --revision v0.0.12 --local-dir model-assets
.venv-scibert/bin/python -m research full-label predict \
  --model model-assets/checkpoints/scibert-full-label-006 \
  --device cpu --text 'We used NumPy 1.24.3.' --format pretty
```

Choose `scibert-full-label-012` to run the adapted detector. Download the complete snapshot and preserve the relative directory layout; a detector alone cannot run the complete pipeline. The tag fixes the model snapshot; individual stage hashes are checked by the loader.

## Install a received bundle

Use a fresh source checkout. Do not extract over existing checkpoints.

```sh
# Run in the directory containing the received files.
shasum -a 256 -c ossomex-v0.0.1-models.tar.sha256
# Inspect the archive before extraction; use a trusted maintainer bundle.
tar -tf ossomex-v0.0.1-models.tar
# Replace this destination with your fresh OSSoMeX checkout.
tar -xf ossomex-v0.0.1-models.tar -C /path/to/OSSoMeX
```

On Linux, `sha256sum -c` is equivalent. The archive contains both full-label manifests, six distinct stage directories (four shared heads plus two detectors), per-file hashes and notices. The CLI checks stage and file hashes when loading. Tokenizers and configs are included; inference does not require the original Hugging Face cache or training data.

The bundle is 2,642,329,600 bytes (about 2.46 GiB). Its [expected checksum](../../models/v0.0.1-models.sha256) is tracked.

The two models share stages. Keep this relative layout:

```text
checkpoints/
  scibert-full-label-006/manifest.json
  scibert-full-label-012/manifest.json
  scibert-detector-005-wordpiece/
  scibert-detector-012-softcite-wordpiece/
  scibert-linker-004/
  scibert-intent-003/
  scibert-sentiment-003/
  scibert-alias-003/
```

## Selection and provenance

- **006:** earlier selected name-extraction POC; [model card](006.md). It can invent version links and miss names.
- **012:** [current model card](012.md); benchmark adaptation, not a promoted general replacement. It reuses 011 detector weights and constrained decoding.
- [Registry](../../models/registry.json): pipeline/stage manifest hashes and release roles.

Five separate encoders are loaded for inference. CPU is supported. Training and benchmark claims remain experimental; neither pipeline is production-validated. There is no automatic default model and no supported CUDA CLI option.

## Maintainer: package existing local weights

```sh
python3 scripts/package_models.py \
  --model scibert-full-label-006 --model scibert-full-label-012 \
  --output output/releases/v0.0.1/ossomex-v0.0.1-models.tar
```

The packager accepts only registry-pinned pipelines and stage files named in their manifests. It verifies source and archived bytes, deduplicates shared stages, preserves manifest bytes, and refuses an existing destination. It does not include source papers, annotation stores, caches or arbitrary checkpoint-directory contents. Original manifests retain historical workstation paths as provenance; those paths are not needed to load weights.
