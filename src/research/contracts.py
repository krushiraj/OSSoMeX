"""Canonical v2 annotations and deterministic v1 public exports."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
import re

from .schema import INTENTS, SENTIMENTS, validate_public_record

FIELDS = ("software", "versions", "created", "used", "shared", "sentiment")
INTENT_BITS = ("created", "used", "shared")


class ContractError(ValueError):
    def __init__(self, record: dict, field: str, message: str, code: str = "INVALID_CONTRACT"):
        self.issues = [{"document_id": record.get("document_id"), "field": field,
                        "code": code, "message": message}]
        super().__init__(f"{field}: {message}")


def text_revision(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def occurrence_id(document_id: str, revision: str, start: int, end: int) -> str:
    return f"{document_id}|{revision}|{start}:{end}"


def check_span(span, record: dict, field: str, length: int | None = None) -> tuple[int, int]:
    if not isinstance(span, dict) or any(type(span.get(k)) is not int for k in ("start", "end")):
        raise ContractError(record, field, "requires integer code-point boundaries")
    start, end = span["start"], span["end"]
    if start < 0 or end <= start or (length is not None and end > length):
        raise ContractError(record, field, "span is empty, reversed, or outside text")
    return start, end


def _nonblank(record: dict, field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(record, field, "requires a nonblank string")
    return value


def validate_document(record: dict) -> dict:
    if not isinstance(record, dict):
        raise ContractError({}, "document", "requires an object")
    _nonblank(record, "document_id")
    text = _nonblank(record, "text")
    try:
        revision = text_revision(text)
    except UnicodeEncodeError as exc:
        raise ContractError(record, "text", "invalid UTF-8 text", "INPUT_FAILED") from exc
    if "text_revision" in record and record["text_revision"] != revision:
        raise ContractError(record, "text_revision", "does not match text SHA-256")
    if record.get("metadata") is not None and not isinstance(record["metadata"], dict):
        raise ContractError(record, "metadata", "requires an object or null")
    for key in ("sections", "page_spans"):
        if record.get(key) is None:
            continue
        if not isinstance(record[key], list):
            raise ContractError(record, key, "requires an array or null")
        for span in record[key]:
            check_span(span, record, key, len(text))
    return {**deepcopy(record), "text_revision": revision}


def _validate_prediction_metadata(record: dict) -> None:
    _nonblank(record, "run_id")
    if record.get("calibration") not in ("uncalibrated", "development_calibrated"):
        raise ContractError(record, "calibration", "unknown calibration state")
    if type(record.get("needs_review")) is not bool:
        raise ContractError(record, "needs_review", "requires a boolean")
    reasons = record.get("review_reasons")
    if not isinstance(reasons, list) or any(not isinstance(r, str) or not r.strip() for r in reasons):
        raise ContractError(record, "review_reasons", "requires nonblank reason strings")
    if record["needs_review"] != bool(reasons):
        raise ContractError(record, "review_reasons", "must agree with needs_review")
    hashes = record.get("checkpoint_hashes")
    if not isinstance(hashes, dict) or not hashes or any(
        not isinstance(key, str) or not key.strip() or not isinstance(value, str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", value) for key, value in hashes.items()
    ):
        raise ContractError(record, "checkpoint_hashes", "requires named SHA-256 checkpoint hashes")
    scores = record.get("scores")
    if not isinstance(scores, dict) or set(scores) != {"software", "version_links", "intents", "sentiment"}:
        raise ContractError(record, "scores", "requires software, version_links, intents and sentiment scores")
    values = [scores["software"]]
    for field, labels in (("intents", INTENT_BITS), ("sentiment", SENTIMENTS)):
        group = scores[field]
        if not isinstance(group, dict) or set(group) != set(labels):
            raise ContractError(record, f"scores.{field}", "requires every class probability")
        values.extend(group.values())
    links = scores["version_links"]
    if not isinstance(links, list) or len(links) != len(record.get("version_links", [])):
        raise ContractError(record, "scores.version_links", "requires one score per confirmed edge")
    values.extend(links)
    if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ContractError(record, "scores", "probabilities must be finite numbers in [0,1]")
    if record.get("version_status") == "ambiguous" and not record["needs_review"]:
        raise ContractError(record, "needs_review", "ambiguous versions require review")


def _validate_occurrence(record: dict, text: str | None = None) -> None:
    if not isinstance(record, dict):
        raise ContractError({}, "occurrence", "requires an object")
    if record.get("schema_version") != "2.0":
        raise ContractError(record, "schema_version", "expected 2.0")
    doc_id = _nonblank(record, "document_id")
    revision = _nonblank(record, "text_revision")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", revision):
        raise ContractError(record, "text_revision", "expected SHA-256 revision")
    name = _nonblank(record, "name")
    length = len(text) if text is not None else None
    start, end = check_span(record.get("name_span"), record, "name_span", length)
    if record.get("mention_id") != occurrence_id(doc_id, revision, start, end):
        raise ContractError(record, "mention_id", "does not match occurrence identity")
    context = _nonblank(record, "context_sentence")
    cs, ce = check_span(record.get("context_span"), record, "context_span", length)
    if not cs <= start < end <= ce or ce - cs != len(context) or context[start-cs:end-cs] != name:
        raise ContractError(record, "context_span", "must contain the exact name and context text")
    if record.get("context_kind") not in ("sentence", "paragraph", "table", "reference"):
        raise ContractError(record, "context_kind", "unsupported context kind")
    if text is not None and (text_revision(text) != revision or text[start:end] != name or text[cs:ce] != context):
        raise ContractError(record, "text_revision", "source hash or span text mismatch")

    known = record.get("known")
    if known is None:
        required = ("scores", "calibration", "needs_review", "review_reasons", "run_id", "checkpoint_hashes")
        if any(key not in record for key in required):
            raise ContractError(record, "known", "annotation masks or prediction provenance required")
        _validate_prediction_metadata(record)
        known = dict.fromkeys(FIELDS, True)
    elif not isinstance(known, dict) or set(known) != set(FIELDS) or any(type(v) is not bool for v in known.values()):
        raise ContractError(record, "known", "requires every field mask as a boolean")
    if not known["software"]:
        raise ContractError(record, "known.software", "an occurrence must have a known software span")

    links = record.get("version_links")
    if not isinstance(links, list):
        raise ContractError(record, "version_links", "requires an array")
    seen = set()
    for edge in links:
        if not isinstance(edge, dict):
            raise ContractError(record, "version_links", "requires edge objects")
        value = _nonblank(edge, "text")
        vs, ve = check_span(edge.get("span"), record, "version_links.span", length)
        if value.lower().strip() in ("null", "na", "n/a", "none") or len(value) != ve-vs:
            raise ContractError(record, "version_links.text", "invalid version text")
        if text is not None and text[vs:ve] != value:
            raise ContractError(record, "version_links.text", "source span mismatch")
        if edge.get("status") not in ("explicit_local", "explicit_remote"):
            raise ContractError(record, "version_links.status", "only confirmed edges belong in version_links")
        key = (vs, ve, value)
        if key in seen:
            raise ContractError(record, "version_links", "duplicate version edge")
        seen.add(key)
    status = record.get("version_status")
    if status not in ("explicit", "absent", "ambiguous", "unannotated"):
        raise ContractError(record, "version_status", "unsupported version state")
    if (status == "explicit" and not links) or (status in ("absent", "unannotated") and links):
        raise ContractError(record, "version_status", "inconsistent version links")
    if "known" in record and known["versions"] != (status in ("explicit", "absent")):
        raise ContractError(record, "known.versions", "inconsistent version supervision")

    intents = record.get("intents")
    bits_known = [known[label] for label in INTENT_BITS]
    if intents is None:
        if any(bits_known):
            raise ContractError(record, "intents", "known decisions require an intent array")
    else:
        if not isinstance(intents, list) or any(i not in INTENTS for i in intents):
            raise ContractError(record, "intents", "invalid label array")
        if intents != [label for label in INTENTS if label in intents]:
            raise ContractError(record, "intents", "labels must be unique and canonically ordered")
        if "mentioned" in intents and (len(intents) != 1 or not all(bits_known)):
            raise ContractError(record, "intents", "mentioned is exclusive and requires all intent bits known")
        if any(label in intents and not known[label] for label in INTENT_BITS):
            raise ContractError(record, "intents", "positive label has unknown mask")
        if all(bits_known) and not intents:
            raise ContractError(record, "intents", "complete negatives must use mentioned")
    sentiment = record.get("sentiment")
    if (known["sentiment"] and sentiment not in SENTIMENTS) or (not known["sentiment"] and sentiment is not None):
        raise ContractError(record, "sentiment", "inconsistent sentiment supervision")
    evidence = record.get("evidence")
    if not isinstance(evidence, dict):
        raise ContractError(record, "evidence", "requires field evidence")
    for field in ("intents", "sentiment"):
        if not isinstance(evidence.get(field), list):
            raise ContractError(record, f"evidence.{field}", "requires an array")
        for span in evidence[field]:
            check_span(span, record, f"evidence.{field}", length)
    if set(intents or []) & set(INTENT_BITS) and not evidence["intents"]:
        raise ContractError(record, "evidence.intents", "positive intent requires evidence")
    if sentiment in ("positive", "negative", "mixed") and not evidence["sentiment"]:
        raise ContractError(record, "evidence.sentiment", "expressed sentiment requires evidence")


def validate_occurrence(record: dict, text: str) -> dict:
    _validate_occurrence(record, text)
    return deepcopy(record)


def unique_occurrences(occurrences: list[dict]) -> list[dict]:
    by_id = {}
    for record in occurrences:
        _validate_occurrence(record)
        key = record["mention_id"]
        if key in by_id and by_id[key] != record:
            raise ContractError(record, "mention_id", "conflicting duplicate occurrence")
        by_id[key] = record
    return deepcopy(list(by_id.values()))


def export_public(occurrences: list[dict]) -> tuple[list[dict], list[dict]]:
    rows, mappings = [], []
    for occ in unique_occurrences(occurrences):
        if "known" in occ and not all(occ["known"].values()):
            raise ContractError(occ, "known", "partial annotation is not a completed prediction")
        edges = list(enumerate(occ["version_links"])) or [(None, None)]
        for ordinal, edge in edges:
            row = {key: deepcopy(occ[key]) for key in ("name", "context_sentence", "intents", "sentiment")}
            row["version"] = edge["text"] if edge else None
            problems = validate_public_record(row)
            if problems:
                raise ContractError(occ, "public_record", "; ".join(problems))
            mappings.append({"export_ordinal": len(rows), "document_id": occ["document_id"],
                             "mention_id": occ["mention_id"], "version_edge_ordinal": ordinal})
            rows.append(row)
    return rows, mappings
