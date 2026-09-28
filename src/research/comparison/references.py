"""Project verified review snapshots without expanding their reviewed coverage."""

from copy import deepcopy
import json
from pathlib import Path

from ..annotations.review_workflow import project_review
from ..annotations.snapshots import read_review_snapshot
from ..contracts import FIELDS, check_span
from ..data.manifest import digest
from .contracts import validate_input, validate_reference


REVIEW_KINDS = ("human_reviewed", "agent_provisional")


def _permitted(record: dict) -> None:
    for metadata in (record, record.get("provenance", {}), record.get("metadata", {})):
        if not isinstance(metadata, dict):
            continue
        if (metadata.get("private") is True or metadata.get("public") is False
                or metadata.get("role") in ("test", "private", "heldout")
                or metadata.get("split") in ("test", "private", "heldout")):
            raise ValueError("private or test reference not permitted")


def _documents(documents: list[dict]) -> dict[str, dict]:
    indexed = {}
    for document in documents:
        document = validate_input(document)
        if document["document_id"] in indexed:
            raise ValueError("duplicate document_id")
        indexed[document["document_id"]] = document
    return indexed


def _review_kind(projection: dict, occurrence: dict, field: str) -> str:
    state = projection["fields"][occurrence["mention_id"]][field]["state"]
    return "human_reviewed" if state == "confirmed" else "agent_provisional"


def _coverage(item: dict, projection: dict) -> list[dict]:
    task, annotation = item["task"], item["annotation"]
    owned = task["annotation_region"]
    result = []
    for region in annotation.get("covered_regions", []):
        start, end = check_span(region, task, "covered_regions")
        fields = region.get("fields")
        if (not owned["start"] <= start < end <= owned["end"]
                or not isinstance(fields, dict) or set(fields) != set(FIELDS)
                or any(type(value) is not bool for value in fields.values())
                or region.get("status") not in ("complete", "partial")):
            raise ValueError("invalid snapshot field coverage")
        for field, label in (("software", "SOFTWARE"), ("versions", "VERSION")):
            if not fields[field]:
                continue
            human = (projection["name_audit"] == "confirmed" and not projection["source_issues"]
                     and (field == "software" or projection["workflow_status"] == "approved"))
            if field == "versions":
                relevant = [o for o in annotation["occurrences"]
                            if start < o["name_span"]["end"] and o["name_span"]["start"] < end]
                if any(not occurrence["known"]["versions"] for occurrence in relevant):
                    continue
                human = human and all(_review_kind(projection, o, field) == "human_reviewed"
                                      for o in relevant)
            result.append({"start": start, "end": end, "label": label, "complete": True,
                           "review_kind": "human_reviewed" if human else "agent_provisional"})
    return result


def references_from_snapshot(snapshot: Path, documents: list[dict]) -> list[dict]:
    """Load only public train/dev snapshots, retaining field and review provenance."""
    indexed = _documents(documents)
    for document in indexed.values():
        _permitted(document)
    snapshot = Path(snapshot)
    manifest_bytes = (snapshot / "manifest.json").read_bytes()
    header = json.loads(manifest_bytes)
    if not isinstance(header, dict) or header.get("role") not in ("train", "dev"):
        raise ValueError("snapshot role not permitted")
    _permitted(header)
    manifest, data = read_review_snapshot(snapshot)
    if manifest != header:
        raise ValueError("snapshot manifest changed during read")
    histories = {}
    for record in data["decision-records.jsonl"]:
        histories.setdefault(record["task_id"], []).append({
            "payload": json.loads(record["payload"]), "result": json.loads(record["result"]),
            "recorded_at": record["recorded_at"]})
    references = []
    for item in data["items.jsonl"]:
        task, annotation = item["task"], item["annotation"]
        _permitted(task)
        if task.get("public") is not True:
            raise ValueError("nonpublic snapshot task not permitted")
        if task["document_id"] not in indexed:
            continue
        document = indexed[task["document_id"]]
        if task["text_revision"] != document["text_revision"]:
            raise ValueError("snapshot text_revision does not match frozen input")
        start, end = check_span(task["context_span"], task, "context_span", len(document["text"]))
        if task["text"] != document["text"][start:end]:
            raise ValueError("snapshot context does not match frozen input")
        history = histories.get(task["task_id"], [])
        projection = project_review(item, history)
        spans, unscored_versions = {}, []
        for occurrence in annotation["occurrences"]:
            candidates = [{"label": "SOFTWARE", "text": occurrence["name"], **occurrence["name_span"],
                           "review_kind": _review_kind(projection, occurrence, "software")}]
            if occurrence['known']['versions']:
                candidates.extend({"label": "VERSION", "text": edge["text"], **edge["span"],
                                   "review_kind": _review_kind(projection, occurrence, "versions")}
                                  for edge in occurrence["version_links"])
            elif occurrence['version_links']:
                unscored_versions.append({'mention_id': occurrence['mention_id'], 'known': False,
                    'version_status': occurrence['version_status'],
                    'version_links': deepcopy(occurrence['version_links'])})
            for span in candidates:
                key = (span["label"], span["start"], span["end"])
                if key not in spans or span["review_kind"] == "human_reviewed":
                    spans[key] = span
        provenance = {"kind": "validated_review_snapshot", "snapshot": str(snapshot),
                      "snapshot_manifest_sha256": digest(manifest_bytes), "role": manifest["role"],
                      "task_id": task["task_id"], "annotation_revision": item["annotation_revision"],
                      "task": deepcopy(task), "review_projection": projection,
                      "covered_regions": deepcopy(annotation.get("covered_regions", [])),
                      "unresolved_regions": deepcopy(annotation.get("unresolved_regions", [])),
                      "annotator": deepcopy(annotation.get("annotator")),
                      "unscored_version_candidates": unscored_versions,
                      "decision_ids": [row["payload"]["decision_id"] for row in history]}
        for key in ("exposed", "training_overlap", "selection_bias"):
            if key in task or key in document:
                provenance[key] = deepcopy(task.get(key, document.get(key)))
        references.append(validate_reference(document, {
            "document_id": document["document_id"], "text_revision": document["text_revision"],
            "spans": list(spans.values()), "coverage": _coverage(item, projection), "provenance": provenance}))
    return references
