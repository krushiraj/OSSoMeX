"""Public JSON contract and internal evidence contract (protocol section 02)."""

from __future__ import annotations

from typing import Any

INTENTS = ("created", "used", "shared", "mentioned")
SENTIMENTS = ("positive", "negative", "mixed", "not_expressed")
VERSION_STATUSES = (
    "explicit_local",
    "explicit_remote",
    "ambiguous",
    "unreadable",
)

PUBLIC_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "array",
    "items": {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "version", "context_sentence", "intents", "sentiment"],
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "version": {"type": ["string", "null"], "minLength": 1},
            "context_sentence": {"type": "string", "minLength": 1},
            "intents": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"enum": list(INTENTS)},
            },
            "sentiment": {"enum": list(SENTIMENTS)},
        },
    },
}

EVIDENCE_FIELDS = (
    "document_id",
    "text_revision",
    "mention_id",
    "name_span",
    "sentence_span",
    "page_numbers",
    "section_type",
    "entity_type",
    "version_links",
    "assertion",
    "actor",
    "feature_support",
    "alignment_status",
    "run_id",
    "chunk_ids",
)


class SemanticError(ValueError):
    """Raised when a record is schema-valid but semantically invalid."""


def _is_blank(value: Any) -> bool:
    return isinstance(value, str) and value.strip() == ""


def validate_public_record(record: Any) -> list[str]:
    """Return a list of problems for one public record (empty means valid)."""
    problems: list[str] = []
    if not isinstance(record, dict):
        return ["record is not an object"]
    allowed = {"name", "version", "context_sentence", "intents", "sentiment"}
    extra = set(record) - allowed
    if extra:
        problems.append(f"unexpected keys: {sorted(extra)}")
    for key in ("name", "version", "context_sentence", "intents", "sentiment"):
        if key not in record:
            problems.append(f"missing key: {key}")
    name = record.get("name")
    if not isinstance(name, str) or _is_blank(name):
        problems.append("name must be a non-blank string")
    version = record.get("version")
    if version is not None and (not isinstance(version, str) or _is_blank(version)):
        problems.append("version must be a non-blank string or JSON null")
    if isinstance(version, str) and version.strip().lower() in {"null", "na", "n/a", "none"}:
        problems.append("version must not be a sentinel string")
    context = record.get("context_sentence")
    if not isinstance(context, str) or _is_blank(context):
        problems.append("context_sentence must be a non-blank string")
    intents = record.get("intents")
    if not isinstance(intents, list) or not intents:
        problems.append("intents must be a non-empty array")
    else:
        seen = [i for i in intents if i in INTENTS]
        if len(seen) != len(intents):
            problems.append("intents contains unknown label")
        if len(set(intents)) != len(intents):
            problems.append("intents must be unique")
        if "mentioned" in intents and len(intents) > 1:
            problems.append("mentioned must appear alone")
    sentiment = record.get("sentiment")
    if sentiment not in SENTIMENTS:
        problems.append("sentiment is not an allowed value")
    return problems


def validate_public_records(records: Any) -> tuple[list[dict], list[dict]]:
    """Split records into (valid, invalid_with_reasons)."""
    if not isinstance(records, list):
        return [], [{"record": records, "reasons": ["top level is not an array"]}]
    valid: list[dict] = []
    invalid: list[dict] = []
    for rec in records:
        problems = validate_public_record(rec)
        if problems:
            invalid.append({"record": rec, "reasons": problems})
        else:
            valid.append(rec)
    return valid, invalid
