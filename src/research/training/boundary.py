"""Deterministic post-decode boundary repair for decoded software spans.

Token decoding is a local decision, so a correctly identified name can still
come back with the wrong character boundaries, and a name can be reported from
inside a longer code identifier. These are tokenizer and orthography artifacts
rather than model beliefs, so they are corrected by explicit, frozen,
individually auditable rules instead of by retraining or by a learned reranker.

Every rule is anchored to the source text, so a repaired span is exact by
construction and can never drift from the document. Each applied rule appends an
event carrying the reason and the removed characters, so a repair is
reviewable rather than silent.

The policy is disabled by default. A detector checkpoint that does not name a
policy keeps its historical behaviour, so existing checkpoints stay valid.
"""

from __future__ import annotations

import re

# Trailing trademark and ornament glyphs that are never part of a tool name.
TRADEMARK_GLYPHS = "®™©℠"

# `Scikit Learn-h, 2021` — a bibliography disambiguator letter. The year that
# confirms it usually sits *outside* the decoded span, so it is checked against
# the full document rather than against the span text.
CITATION_LETTER = re.compile(r"-[a-z]$")
CITATION_YEAR = re.compile(r",\s*\d{4}\b")

# A version the decoder absorbed into the name, e.g. `scikit-learn 12` or
# `numpy 1.21.0`. Both branches need a real preceding separator, so a digit
# glued to a name (`Python3`, `LaTeX2e`) is never split off.
ABSORBED_VERSION = re.compile(r"(?<=[\s(\[])(?:v)?\d+(?:\.\d+)+(?:[-+][\w.]+)*\s*$")
# A bare integer is only promoted from two digits up. A single digit after a
# name (`Python 3`, `MATLAB 9`) is far more often part of the name than a
# release, so promoting it would invent a split rather than fix a boundary.
ABSORBED_BARE_NUMBER = re.compile(r"(?<=[\s(\[])\d{2,4}\s*$")

# A following word the tokenizer glued onto the name, e.g. `Scikit-LearnThe`.
# Only closed-class forms are listed, and the preceding character must be
# lowercase, so a name that legitimately ends in `in` (Berlin) is never cut.
GLUED_FUNCTION_WORD = re.compile(
    r"(?<=[a-z])(?:The|A|An|To|And|For|With|On|At|By|Of|In|As|Is|It|We|Using|That|This)$")

IDENTIFIER_CHARACTER = re.compile(r"[A-Za-z0-9_]")

NONE = "none"
SOFTWARE_SPAN_RULES_V1 = "software-span-rules-v1"

# Frozen policy. Changing any value here is a new policy name, never an edit.
POLICIES: dict[str, dict] = {
    NONE: {"schema_version": "boundary-repair-1", "identifier_suppression": False,
           "trademark_suffix": False, "absorbed_version": False, "citation_suffix": False,
           "glued_function_word": False, "nested_span_resolution": False},
    SOFTWARE_SPAN_RULES_V1: {"schema_version": "boundary-repair-1", "identifier_suppression": True,
                             "trademark_suffix": True, "absorbed_version": True, "citation_suffix": True,
                             "glued_function_word": True, "nested_span_resolution": True},
}


def boundary_repair_policy(manifest) -> str:
    """Read the frozen repair policy, defaulting to no repair."""
    policy = manifest.get("inference", {})
    if not isinstance(policy, dict) or "boundary_repair" not in policy:
        return NONE
    name = policy["boundary_repair"]
    if not isinstance(name, str) or name not in POLICIES:
        raise ValueError("invalid boundary repair policy")
    return name


def _event(reason, label, start, end, text, **extra):
    return {"reason": reason, "label": label, "start": start, "end": end,
            "removed": text, **extra}


def _identifier_internal(text, start, end):
    """True when the span sits strictly inside a longer code identifier."""
    if start <= 0 or end >= len(text):
        return False
    return bool(IDENTIFIER_CHARACTER.match(text[start - 1])
                and IDENTIFIER_CHARACTER.match(text[end]))


def _citation_year_follows(text, end):
    """True when a four-digit year follows the span after a comma."""
    return CITATION_YEAR.match(text, end) is not None


def repair_span(text: str, start: int, end: int, label: str, policy: dict):
    """Repair one decoded span. Returns ([(label, start, end)], events)."""
    if policy["identifier_suppression"] and _identifier_internal(text, start, end):
        return [], [_event("identifier_internal", label, start, end, text[start:end])]

    spans: list[tuple[str, int, int]] = []
    events: list[dict] = []

    def emit(kind, lo, hi, reason=None):
        if hi <= lo:
            return
        spans.append((kind, lo, hi))
        if reason is not None:
            events.append(_event(reason, label, lo, hi, text[lo:hi]))

    if label == "SOFTWARE" and policy["absorbed_version"]:
        body = text[start:end]
        match = ABSORBED_VERSION.search(body) or ABSORBED_BARE_NUMBER.search(body)
        if match is not None and match.start() > 0:
            version_start = start + match.start()
            while version_start < end and text[version_start].isspace():
                version_start += 1
            if version_start < end:
                events.append(_event("absorbed_version", label, start, end, text[start:end],
                                     promoted={"label": "VERSION", "start": version_start,
                                               "end": end, "text": text[version_start:end]}))
                emit("VERSION", version_start, end)
                end = version_start

    if policy["trademark_suffix"]:
        while end > start and text[end - 1] in TRADEMARK_GLYPHS:
            events.append(_event("trademark_suffix", label, start, end, text[start:end],
                                 removed_characters=text[end - 1]))
            end -= 1

    if policy["citation_suffix"]:
        match = CITATION_LETTER.search(text, start, end)
        if match is not None and match.start() > start and _citation_year_follows(text, end):
            events.append(_event("citation_suffix", label, start, end, text[start:end],
                                 removed_characters=text[match.start():end]))
            end = match.start()

    if policy["glued_function_word"]:
        match = GLUED_FUNCTION_WORD.search(text[start:end])
        if match is not None and match.start() > 0:
            events.append(_event("glued_function_word", label, start, end, text[start:end],
                                 removed_characters=text[start + match.start():end]))
            end = start + match.start()

    # The separator that opened a promoted version is not part of the name.
    trimmed = end
    while trimmed > start and text[trimmed - 1].isspace():
        trimmed -= 1
    if trimmed != end:
        events.append(_event("whitespace_trim", label, start, end, text[start:end]))
        end = trimmed

    emit(label, start, end)
    return spans, events


def _resolve_nested(spans):
    """Keep the longest span when one prediction is contained in another."""
    kept: list[tuple[str, int, int]] = []
    for kind, start, end in sorted(spans, key=lambda row: (row[1], row[0], -(row[2] - row[1]))):
        if any(kind == other_kind and other_start <= start and end <= other_end
               and (end - start) < (other_end - other_start) for other_kind, other_start, other_end in kept):
            continue
        kept.append((kind, start, end))
    return kept


def repair_spans(text: str, spans: list[dict], policy_name: str):
    """Repair a whole document's decoded spans.

    Returns `(repaired, events)`. `repaired` entries keep the input keys and
    replace only `start`, `end` and `text`, so an unrepaired span is returned
    byte-identical to its input.
    """
    policy = POLICIES.get(policy_name)
    if policy is None:
        raise ValueError("unknown boundary repair policy")
    repaired: list[dict] = []
    events: list[dict] = []
    if policy_name == NONE:
        return [dict(span) for span in spans], events

    for span in spans:
        start, end, label = span["start"], span["end"], span["label"]
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(text):
            raise ValueError("decoded span is outside the source text")
        if text[start:end] != span.get("text", text[start:end]):
            raise ValueError("decoded span does not match the source text")
        produced, span_events = repair_span(text, start, end, label, policy)
        events.extend(span_events)
        for kind, lo, hi in produced:
            entry = {key: value for key, value in span.items() if key not in ("start", "end", "text", "label")}
            entry.update(label=kind, start=lo, end=hi, text=text[lo:hi])
            repaired.append(entry)

    if policy["nested_span_resolution"]:
        keep = _resolve_nested([(span["label"], span["start"], span["end"]) for span in repaired])
        survivors = set(keep)
        resolved: list[dict] = []
        for span in repaired:
            key = (span["label"], span["start"], span["end"])
            if key in survivors:
                survivors.discard(key)
                resolved.append(span)
            else:
                events.append(_event("nested_span", span["label"], span["start"], span["end"],
                                     span["text"]))
        repaired = sorted(resolved, key=lambda row: (row["start"], row["end"]))
    return repaired, events
