"""Deterministic aggregation of raw extraction records (protocol section 16).

Counts are computed in ordinary code from distinct occurrence IDs, never by a
model. A mention with two version edges contributes once to total mention count
and once to each relevant version bucket.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from .contracts import unique_occurrences


def aggregate_occurrences(occurrences: list[dict]) -> dict:
    """Count canonical occurrences, not scalar-version public rows."""
    records = unique_occurrences(occurrences)
    intents, sentiments, versions = Counter(), Counter(), Counter()
    export_rows = 0
    for record in records:
        intents.update(record.get("intents") or [])
        if record.get("sentiment") is not None:
            sentiments.update([record["sentiment"]])
        values = {edge["text"] for edge in record["version_links"]}
        versions.update(values or {"@null"})
        export_rows += max(1, len(record["version_links"]))
    return {"mention_count": len(records), "export_row_count": export_rows,
            "intent_counts": dict(intents), "sentiment_counts": dict(sentiments),
            "version_buckets": dict(versions)}


def aggregate_documents(records: list[dict]) -> dict:
    """records: list of prediction rows combining public + evidence fields."""
    docs: dict[str, dict] = {}
    for rec in records:
        doc_id = rec.get("document_id", "<unknown>")
        doc = docs.setdefault(
            doc_id,
            {
                "document_id": doc_id,
                "distinct_mention_count": 0,
                "export_row_count": 0,
                "mention_ids": set(),
                "intent_counts": Counter(),
                "sentiment_counts": Counter(),
                "version_buckets": Counter(),
                "version_mention_ids": defaultdict(list),
            },
        )
        doc["export_row_count"] += 1
        mid = rec.get("mention_id") or _fallback_mention_id(rec)
        if mid:
            doc["mention_ids"].add(mid)
        for intent in rec.get("intents") or []:
            doc["intent_counts"][intent] += 1
        sentiment = rec.get("sentiment")
        if sentiment:
            doc["sentiment_counts"][sentiment] += 1

        raw_version = rec.get("version")
        version_key = raw_version if raw_version is not None else "@null"
        doc["version_buckets"][version_key] += 1
        if mid:
            doc["version_mention_ids"][version_key].append(mid)

    out = {}
    for doc_id, doc in docs.items():
        out[doc_id] = {
            "document_id": doc_id,
            "distinct_mention_count": len(doc["mention_ids"]),
            "export_row_count": doc["export_row_count"],
            "intent_counts": dict(doc["intent_counts"]),
            "sentiment_counts": dict(doc["sentiment_counts"]),
            "version_buckets": {
                k: {"export_rows": v, "distinct_mentions": len(set(doc["version_mention_ids"][k]))}
                for k, v in doc["version_buckets"].items()
            },
        }
    return out


def _fallback_mention_id(rec: dict) -> str | None:
    span = rec.get("name_span")
    if span and isinstance(span, dict):
        return f"{rec.get('document_id')}:{span.get('start')}:{span.get('end')}"
    return None
