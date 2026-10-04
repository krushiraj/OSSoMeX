# Changelog

## v0.0.12 - 2026-10-04 (Python package 0.0.1)

First versioned research handoff. This replaces the earlier unreleased 0.1.0 placeholder in package metadata; it is not a model-quality upgrade.

- Add installation, inference, architecture, contributor and training guides.
- Register 006 and 012 identities and document their different roles and limitations.
- Add a manifest-verified local model packager and bundle installation instructions.
- Track historical benchmark/training Python sources separately from ignored results.
- Include current aggregate three-model scores and the 006/012 detector diagnosis.
- Capture existing optional boundary-repair implementation and its existing tests; repair remains disabled for 006 and 012.
- Add release notes, citation metadata and a manually dispatched CI workflow.

The source tag was renamed to v0.0.12 to identify checkpoint 012. Source and package assets are on the public GitHub repository; weights and model cards are on the public Hugging Face repository. Follow-up documentation records 1,869 passing tests and CPU inference from a clean wheel installation. No retraining or model promotion is implied. See `docs/releases/v0.0.1.md` for verification and remaining distribution work.

### Follow-up validation

- Fix contributor setup and Linux CI to install the model/interactive dependencies required by the full suite. Record CI environment and JUnit output.
- Add pinned Hugging Face model access instructions, local wheel-inference evidence and public model documentation.
- Correct the package documentation link to the actual default branch.
- Make retained research-data prerequisites explicit in tests. Verify reloaded model weights exactly and compare floating-point logits within a declared tolerance; record reload deltas in JUnit.

### Public documentation cleanup

- Remove internal plans, session notes and development handoffs from source history and release archives.
- Keep public setup, model, architecture, training and benchmark documentation.
- Move synthetic annotation fixtures into `examples/` and update their consumers.
- Rename the primary branch to `main`; refresh source release assets without changing model weights.
