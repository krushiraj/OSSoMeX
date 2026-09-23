"""Convert frozen v1 rows for evaluation only, without inventing annotations."""

from copy import deepcopy

from ..contracts import ContractError, FIELDS, occurrence_id, validate_document
from ..schema import INTENTS, SENTIMENTS


def convert_legacy(rows: list[dict], documents: list[dict], capabilities: dict, *, reference: bool = False) -> dict:
    docs = {d["document_id"]: validate_document(d) for d in documents}
    occurrences, unresolved, issues = {}, {}, []
    for ordinal, source in enumerate(rows):
        row = deepcopy(source)
        doc_id = row.get("document_id")
        if doc_id not in docs:
            raise ContractError(row, "document_id", "not in fixed document population")
        doc = docs[doc_id]
        revision = doc["text_revision"]
        if row.get("text_revision") is not None and row["text_revision"] != revision:
            raise ContractError(row, "text_revision", "legacy revision differs from frozen source")
        span = row.get("name_span")
        aligned = (isinstance(span, dict) and type(span.get("start")) is int and type(span.get("end")) is int
                   and 0 <= span["start"] < span["end"] <= len(doc["text"])
                   and isinstance(row.get("name"), str)
                   and doc["text"][span["start"]:span["end"]].casefold() == row["name"].casefold())
        if aligned and doc["text"][span["start"]:span["end"]] != row["name"]:
            issues.append({"document_id": doc_id, "source_ordinal": ordinal, "code": "SOURCE_CASE_RESTORED",
                           "reported_name": row["name"]})
            row["name"] = doc["text"][span["start"]:span["end"]]
        labels = row.get("intents")
        if labels and (not isinstance(labels, list) or any(label not in INTENTS for label in labels)
                       or ("mentioned" in labels and len(labels) != 1)):
            raise ContractError(row, "intents", "invalid legacy labels")
        known = dict.fromkeys(FIELDS, False)
        known["software"] = True
        known["versions"] = bool(capabilities.get("versions", capabilities.get("version", False))
                                 and any(key in row for key in ("version", "versions", "version_links")))
        for label in ("created", "used", "shared"):
            known[label] = bool(capabilities.get("intents") and labels)
        known["sentiment"] = bool(capabilities.get("sentiment") and row.get("sentiment") in SENTIMENTS)
        values = row.get("versions", [row.get("version")])
        if not isinstance(values, list):
            raise ContractError(row, "versions", "expected a list")
        links = []
        for value in values:
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip() or value.lower().strip() in ("null", "na", "n/a", "none"):
                raise ContractError(row, "version", "invalid legacy version value")
            native = next((edge for edge in row.get("version_links", []) if edge.get("text") == value), {})
            pos = native.get("span")
            if not (isinstance(pos, dict) and type(pos.get("start")) is int and type(pos.get("end")) is int
                    and 0 <= pos["start"] < pos["end"] <= len(doc["text"])
                    and doc["text"][pos["start"]:pos["end"]] == value):
                pos = None
            links.append({"text": value, "span": pos, "status": "explicit_local" if pos else "legacy_value_only"})
        converted = {"record_kind": "legacy_evaluation_only", "document_id": doc_id, "text_revision": revision,
                     "name": row.get("name"), "name_span": span if aligned else None,
                     "mention_id": occurrence_id(doc_id, revision, span["start"], span["end"]) if aligned else f"{doc_id}|unaligned:{ordinal}",
                     "version_links": links, "known": known,
                     "intents": [label for label in INTENTS if label in labels] if labels and capabilities.get("intents") else None,
                     "sentiment": row.get("sentiment") if known["sentiment"] else None,
                     "source_ordinals": [ordinal]}
        destination = occurrences
        if not aligned:
            converted["alignment_status"] = "unaligned"
            converted["reported_name_span"] = deepcopy(span)
            converted["reported_mention_id"] = row.get("mention_id")
            if isinstance(row.get("mention_id"), str) and row["mention_id"].strip():
                converted["mention_id"] = f"{doc_id}|{revision}|reported:{row['mention_id']}"
            destination = unresolved
            issues.append({"document_id": doc_id, "source_ordinal": ordinal, "code": "UNALIGNED_PREDICTION"})
        key = converted["mention_id"]
        if key not in destination:
            destination[key] = converted
        else:
            previous = destination[key]
            if previous["name"] != converted["name"] or previous.get("reported_name_span") != converted.get("reported_name_span"):
                raise ContractError(row, "mention_id", "conflicting reported occurrence identity")
            conflicts = [field for field in ("intents", "sentiment") if previous[field] != converted[field]]
            if reference and (conflicts or previous["known"] != converted["known"]):
                raise ContractError(row, "mention_id", "conflicting legacy reference labels at one occurrence")
            for field in conflicts:
                previous[field] = None
                previous.setdefault("invalid_fields", [])
                if field not in previous["invalid_fields"]:
                    previous["invalid_fields"].append(field)
                issues.append({"document_id": doc_id, "source_ordinal": ordinal,
                               "code": "CONFLICTING_LEGACY_LABELS", "field": field})
            previous["source_ordinals"].append(ordinal)
            for link in links:
                if link not in previous["version_links"]:
                    previous["version_links"].append(link)
    return {"occurrences": list(occurrences.values()), "unresolved_predictions": list(unresolved.values()), "issues": issues}
