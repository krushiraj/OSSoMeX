"""Native-to-canonical mapping with conservative field coverage and exclusions."""

from copy import deepcopy
import re

from ..contracts import (ContractError, FIELDS, INTENT_BITS, check_span, occurrence_id,
                         text_revision, validate_document, validate_occurrence)


def normalize_with_offsets(text: str) -> dict:
    parts, spans, changes = [], [], []
    pattern = r"\r\n|\r|[ \t]+|\u00a0|\n{3,}|[^\r \t\u00a0]"
    for match in re.finditer(pattern, text):
        raw = match.group()
        value = "\n" if raw.startswith("\r") else " " if raw[0] in " \t\u00a0" else "\n\n" if raw.startswith("\n\n\n") else raw
        parts.append(value)
        spans.extend({"start": match.start(), "end": match.end()} for _ in value)
        if value != raw:
            changes.append({"source_start": match.start(), "source_end": match.end(), "replacement": value})
    return {"text": "".join(parts), "original": text, "source_spans": spans,
            "changes": changes, "normalizer_version": "scibert-v2.0"}


def project_span(span: dict, normalized: dict) -> dict | None:
    start, end = span["start"], span["end"]
    mapping = normalized["source_spans"]
    positions = [i for i, source in enumerate(mapping) if start <= source["start"] and source["end"] <= end]
    if not positions:
        return None
    lo, hi = positions[0], positions[-1] + 1
    if mapping[lo]["start"] != start or mapping[hi-1]["end"] != end:
        return None
    expected = normalize_with_offsets(normalized["original"][start:end])["text"]
    return {"start": lo, "end": hi} if normalized["text"][lo:hi] == expected else None


def map_annotations(parsed: dict, document: dict, mapping: dict) -> dict:
    raw = parsed["text"]
    if document["text"] != raw:
        raise ContractError(document, "text", "native text must match supplied document", "ALIGNMENT_UNRESOLVED")
    if not mapping.get("policy_version"):
        raise ContractError(document, "policy_version", "explicit source mapping version required")
    normalized = normalize_with_offsets(raw)
    doc = validate_document({**deepcopy(document), "text": normalized["text"],
                             "text_revision": text_revision(normalized["text"]),
                             "original_text_revision": text_revision(raw),
                             "normalizer_version": normalized["normalizer_version"]})
    text = doc["text"]
    revision = doc["text_revision"]
    coverage, exclusions = [], []
    for region in mapping.get("coverage", []):
        check_span(region, doc, "coverage", len(raw))
        projected = project_span(region, normalized)
        if projected is None:
            raise ContractError(doc, "coverage", "unalignable coverage boundary", "ALIGNMENT_UNRESOLVED")
        fields = region.get("fields", {})
        if set(fields) - set(FIELDS) or any(type(v) is not bool for v in fields.values()):
            raise ContractError(doc, "coverage.fields", "invalid field masks")
        coverage.append({"document_id": doc["document_id"], "text_revision": revision, **projected,
                         "fields": {field: fields.get(field, False) for field in FIELDS},
                         "status": region.get("status", "complete"),
                         "provenance": deepcopy(region.get("provenance", {}))})

    software_types = mapping.get("software_types", {})
    version_types = set(mapping.get("version_types", []))
    native = parsed["spans"]
    projected_by_id, bad = {}, set()
    issue_ids = {i.get("source_id") for i in parsed["issues"] if i["code"] == "ALIGNMENT_UNRESOLVED"}

    def exclude(span, code):
        bad.add(span["source_id"])
        exclusions.append({"code": code, "source_id": span["source_id"],
                           "segments": deepcopy(span["segments"]), "type": span["type"]})

    for span in native:
        source_id = span["source_id"]
        projected = project_span(span, normalized)
        if projected:
            projected_by_id[source_id] = projected
        if span["type"] in mapping.get("ignored_types", []):
            continue
        if source_id in issue_ids or projected is None or not span["text"].strip():
            exclude(span, "ALIGNMENT_UNRESOLVED")
        elif span["type"] not in software_types and span["type"] not in version_types:
            exclude(span, "UNMAPPED_TYPE")
        elif len(span["segments"]) != 1:
            exclude(span, "DISCONTINUOUS_SPAN")
        elif not any(r["start"] <= projected["start"] < projected["end"] <= r["end"] for r in coverage):
            exclude(span, "OUTSIDE_COVERAGE")
    entities = [s for s in native if s["type"] in software_types or s["type"] in version_types]
    for i, span in enumerate(entities):
        for other in entities[i+1:]:
            if span["start"] < other["end"] and other["start"] < span["end"]:
                for item in (span, other):
                    if item["source_id"] not in bad:
                        exclude(item, "OVERLAPPING_ENTITY")

    version_links = {}
    by_id = {span["source_id"]: span for span in native}
    linked = set()
    for relation in parsed["relations"]:
        if relation["kind"] not in mapping.get("version_relations", []):
            continue
        source, target = relation["source_id"], relation["target_id"]
        if (source in bad or target in bad or source not in by_id or target not in by_id
                or by_id[source]["type"] not in version_types or by_id[target]["type"] not in software_types):
            exclusions.append({"code": "UNRESOLVED_VERSION_RELATION", **relation})
            continue
        span = projected_by_id[source]
        edge = {"text": text[span["start"]:span["end"]], "span": span, "status": "explicit_local"}
        edges = version_links.setdefault(target, [])
        if edge not in edges:
            edges.append(edge)
        linked.add(source)
    unlinked = [span for span in native if span["type"] in version_types and span["source_id"] not in linked]
    for span in unlinked:
        exclusions.append({"code": "UNLINKED_VERSION", "source_id": span["source_id"], "segments": span["segments"]})

    # Split coverage around quarantined entities; their text is not negative supervision.
    masked = []
    for region in coverage:
        boundaries = {region["start"], region["end"]}
        ranges = []
        for span in native:
            if span["source_id"] not in bad:
                continue
            pos = projected_by_id.get(span["source_id"])
            if pos:
                lo, hi = max(region["start"], pos["start"]), min(region["end"], pos["end"])
                if lo < hi:
                    ranges.append((lo, hi))
                    boundaries.update((lo, hi))
            else:
                ranges.append((region["start"], region["end"]))
        points = sorted(boundaries)
        for lo, hi in zip(points, points[1:]):
            item = {**deepcopy(region), "start": lo, "end": hi}
            if any(a < hi and lo < b for a, b in ranges):
                item["fields"] = dict.fromkeys(FIELDS, False)
                item["status"] = "partial"
            if unlinked:
                item["fields"]["versions"] = False
            masked.append(item)

    occurrences = []
    for native_span in native:
        source_id = native_span["source_id"]
        if source_id in bad or native_span["type"] not in software_types:
            continue
        span = projected_by_id[source_id]
        start, end = span["start"], span["end"]
        region = next((r for r in masked if r["start"] <= start < end <= r["end"]), None)
        known = dict.fromkeys(FIELDS, False)
        known["software"] = True
        known["versions"] = bool(region and region["fields"]["versions"])
        bit_values = software_types[native_span["type"]].get("intent_bits", {})
        for label, value in bit_values.items():
            if label not in INTENT_BITS or type(value) is not bool:
                raise ContractError(doc, "intent_bits", "invalid native intent mapping")
            known[label] = True
        intents = [label for label in INTENT_BITS if bit_values.get(label) is True]
        if not bit_values:
            intents = None
        elif all(known[label] for label in INTENT_BITS) and not intents:
            intents = ["mentioned"]
        context_start = text.rfind("\n", 0, start) + 1
        context_end = text.find("\n", end)
        if context_end == -1:
            context_end = len(text)
        context_span = {"start": context_start, "end": context_end}
        links = sorted(version_links.get(source_id, []), key=lambda edge: edge["span"]["start"])
        status = ("explicit" if links else "absent") if known["versions"] else ("ambiguous" if links or unlinked else "unannotated")
        occurrence = {"schema_version": "2.0", "document_id": doc["document_id"], "text_revision": revision,
                      "mention_id": occurrence_id(doc["document_id"], revision, start, end),
                      "name": text[start:end], "name_span": span,
                      "context_sentence": text[context_start:context_end], "context_span": context_span,
                      "context_kind": "paragraph", "version_links": links, "version_status": status,
                      "intents": intents, "sentiment": None, "known": known,
                      "evidence": {"intents": [context_span] if intents and intents != ["mentioned"] else [], "sentiment": []},
                      "review": {"status": "unreviewed", "reasons": ["NATIVE_MAPPING_REVIEW"]},
                      "annotation_provenance": {"kind": "native_import", "source_id": source_id,
                                                "source_type": native_span["type"], "policy_version": mapping["policy_version"]}}
        occurrences.append(validate_occurrence(occurrence, text))
    return {"document": doc, "occurrences": occurrences, "coverage": masked,
            "exclusions": exclusions, "issues": deepcopy(parsed["issues"]),
            "native_annotations": deepcopy(parsed), "normalization": normalized,
            "policy_version": mapping["policy_version"]}
