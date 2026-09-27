"""Strict wire validation for frozen comparison inputs, outputs and references."""

from copy import deepcopy
import math
from pathlib import PurePosixPath

from ..contracts import ContractError, check_span, validate_document


SCHEMA_VERSION = "span-comparison-1"
OFFSET_UNIT = "unicode_codepoint_half_open"
LABEL_CAPABILITIES = {"SOFTWARE": "software_spans", "VERSION": "version_spans"}
CAPABILITY_FIELDS = ("software_spans", "version_spans", "version_linking", "aliases",
                     "intent", "sentiment", "full_contract")
RESULT_STATUSES = ("success", "no_mentions", "failure", "unavailable", "unsupported")
COMPLETED_STATUSES = ("success", "no_mentions")


def _require_fields(record: dict, fields: tuple[str, ...], kind: str) -> None:
    if not isinstance(record, dict):
        raise ContractError({}, kind, "requires an object")
    for field in fields:
        if field not in record:
            raise ContractError(record, field, "required field is missing")


def _nonblank(value, record: dict, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(record, field, "requires a nonblank string")


def _object_array(record: dict, field: str) -> list[dict]:
    value = record.get(field)
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ContractError(record, field, "requires an array of objects")
    return value


def _check_identity(document: dict, record: dict) -> None:
    for field in ("document_id", "text_revision"):
        if record.get(field) != document[field]:
            raise ContractError(record, field, "does not match frozen input")
    if not isinstance(record.get("provenance"), dict):
        raise ContractError(record, "provenance", "requires an object")


def _span_key(document: dict, span: dict, field: str) -> tuple[str, int, int]:
    start, end = check_span(span, document, field, len(document["text"]))
    label = span.get("label")
    if label not in tuple(LABEL_CAPABILITIES):
        raise ContractError(document, f"{field}.label", "expected SOFTWARE or VERSION")
    if span.get("text") != document["text"][start:end]:
        raise ContractError(document, f"{field}.text", "does not match exact source slice")
    return label, start, end


def _check_score(span: dict, result: dict) -> None:
    _require_fields(span, ("score", "score_kind", "alignment_method"), "span")
    _nonblank(span["alignment_method"], result, "spans.alignment_method")
    score, kind = span["score"], span["score_kind"]
    if score is None:
        if kind is not None:
            raise ContractError(result, "spans.score_kind", "must be null when score is null")
        return
    if type(score) not in (int, float) or (type(score) is float and not math.isfinite(score)):
        raise ContractError(result, "spans.score", "requires a finite number or null")
    _nonblank(kind, result, "spans.score_kind")


def _check_raw_artifact(result: dict) -> None:
    raw = result["raw_artifact"]
    if raw is None:
        if result["status"] in COMPLETED_STATUSES:
            raise ContractError(result, "raw_artifact", "completed results require raw evidence")
        return
    _nonblank(raw, result, "raw_artifact")
    path = PurePosixPath(raw)
    if path.is_absolute() or not path.parts or ".." in path.parts or any(c in raw for c in ("\\", ":", "\0")):
        raise ContractError(result, "raw_artifact", "requires a path relative to the run directory")


def validate_input(document: dict) -> dict:
    """Verify an explicitly frozen revision without normalizing the source text."""
    _require_fields(document, ("document_id", "text", "text_revision"), "document")
    return validate_document(document)


def validate_result(document: dict, result: dict) -> dict:
    """Reject invalid model output; callers retain it as a failed result with raw evidence."""
    document = validate_input(document)
    _require_fields(result, ("schema_version", "arm_id", "document_id", "text_revision", "status",
                             "offset_unit", "capabilities", "scores_calibrated", "spans", "unresolved",
                             "chunks", "raw_artifact", "provenance"), "result")
    _check_identity(document, result)
    _nonblank(result["arm_id"], result, "arm_id")
    if result["schema_version"] != SCHEMA_VERSION or result["offset_unit"] != OFFSET_UNIT:
        raise ContractError(result, "schema_version/offset_unit", "unsupported comparison schema or offset unit")
    status = result["status"]
    if status not in RESULT_STATUSES:
        raise ContractError(result, "status", "unsupported result status")
    capabilities = result["capabilities"]
    if (not isinstance(capabilities, dict) or set(capabilities) != set(CAPABILITY_FIELDS)
            or any(type(value) is not bool for value in capabilities.values())):
        raise ContractError(result, "capabilities", "requires every capability as a boolean")
    if type(result["scores_calibrated"]) is not bool:
        raise ContractError(result, "scores_calibrated", "requires a boolean")
    spans = _object_array(result, "spans")
    unresolved = _object_array(result, "unresolved")
    chunks = _object_array(result, "chunks")
    seen = set()
    for span in spans:
        key = _span_key(document, span, "spans")
        if key in seen:
            raise ContractError(result, "spans", "duplicate label/start/end span")
        seen.add(key)
        if not capabilities[LABEL_CAPABILITIES[key[0]]]:
            raise ContractError(result, "capabilities", "span label is not supported by this arm")
        _check_score(span, result)
    for chunk in chunks:
        if chunk.get("status") not in RESULT_STATUSES:
            raise ContractError(result, "chunks.status", "unsupported chunk status")
    if status in COMPLETED_STATUSES:
        if (status == "success") != bool(spans):
            raise ContractError(result, "status", "empty completed extraction must use no_mentions")
        if unresolved or any(chunk["status"] not in COMPLETED_STATUSES for chunk in chunks):
            raise ContractError(result, "status", "unresolved or incomplete output must be a failure")
        if not any(capabilities[field] for field in LABEL_CAPABILITIES.values()):
            raise ContractError(result, "capabilities", "completed extraction requires a span capability")
    else:
        _nonblank(result.get("reason"), result, "reason")
        if spans:
            raise ContractError(result, "spans", "non-completed results cannot carry scored spans")
    if status == "unsupported" and any(capabilities[field] for field in LABEL_CAPABILITIES.values()):
        raise ContractError(result, "capabilities", "unsupported extraction cannot claim span capabilities")
    if result["arm_id"] == "scibert-base" and (status != "unsupported" or result.get("reason") != "no_task_head"):
        raise ContractError(result, "status/reason", "plain base SciBERT requires unsupported/no_task_head")
    _check_raw_artifact(result)
    return deepcopy(result)


def validate_reference(document: dict, reference: dict) -> dict:
    """Validate field-complete coverage while retaining uncovered positive labels."""
    document = validate_input(document)
    _require_fields(reference, ("document_id", "text_revision", "spans", "coverage", "provenance"), "reference")
    _check_identity(document, reference)
    spans = _object_array(reference, "spans")
    coverage = _object_array(reference, "coverage")
    unique = {}
    for span in spans:
        key = _span_key(document, span, "spans")
        unique.setdefault(key, span)
    for region in coverage:
        check_span(region, reference, "coverage", len(document["text"]))
        if region.get("label") not in tuple(LABEL_CAPABILITIES):
            raise ContractError(reference, "coverage.label", "expected SOFTWARE or VERSION")
        if region.get("complete") is not True:
            raise ContractError(reference, "coverage.complete", "coverage requires explicit complete review")
        if region.get("review_kind") not in ("human_reviewed", "agent_provisional"):
            raise ContractError(reference, "coverage.review_kind", "unsupported review kind")
    return {**deepcopy(reference), "spans": deepcopy(list(unique.values()))}
