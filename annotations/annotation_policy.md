# Annotation policy (v1.0, frozen 2026-09-21)

Ground truth is a claim supported by the supplied text, not a guess about real
software behaviour. This pilot uses **silver** labels (agent-produced,
provisional) because no independent human reviewer has been assigned. They must
not be described as human gold.

## Scope defaults (protocol section 01)

- Include: named applications, libraries, frameworks, interpreters/languages,
  developer tools, operating systems, software services/platforms, named ML
  models.
- Exclude: organizations alone, hardware, datasets, file formats, abstract
  methods/algorithms without a software referent.
- Ambiguous names (U-Net, GSEA, Fiji/ImageJ, SAM/SAM2) keep their raw composite
  span and an ambiguity note; never silently canonicalize.
- Named model families keep embedded digits (SAM2, ImageJ2, CodeT5+) unless the
  text supplies a release version.
- References/table names are mentions only; a citation never implies use.
- Primary language English; non-English is a separate exploratory slice.

## Field rules

- **name**: exact named span, original capitalization, no punctuation, no
  version/citation digits attached.
- **version**: source-supported string or JSON `null`. Preserve raw text; do not
  coerce `3.10` to `3.1`; do not fill missing patch numbers.
- **intents**: multi-label `created`, `used`, `shared`; `mentioned` is the
  exclusive fallback. Author-relative: a cited third party's creation is
  `mentioned`.
- **sentiment**: `positive`, `negative`, `mixed`, `not_expressed`. A numeric
  benchmark win alone is not positive sentiment.
- **context_sentence**: sentence containing the name from the supplied text.

## Silver-label procedure used in this pilot

1. A deterministic candidate pass proposes name spans (software lexicon +
   contextual markers + version patterns).
2. Each candidate is adjudicated against the passage by the annotation agent
   without seeing any arm's predictions where practical.
3. The same `research` matching code that scores models is applied to silver
   labels; label support counts are reported.
4. Known limitations: no human double-annotation, no span-F1 calibration gate
   (protocol section 04 target: span F1 >= 0.90). Results are therefore
   provisional and directional.

## Version statuses (internal evidence only)

`explicit_local`, `explicit_remote`, `ambiguous`, `unreadable`, plus JSON-null
absence. Ambiguous/unreadable never becomes confirmed-null gold.
