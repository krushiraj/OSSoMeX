"""Exact-span agreement and coverage-scoped operational reference diagnostics."""

from copy import deepcopy
from itertools import combinations

from ..evaluation.metrics import prf
from .contracts import CAPABILITY_FIELDS, COMPLETED_STATUSES, LABEL_CAPABILITIES, validate_reference, validate_result
from .references import REVIEW_KINDS, _documents, _permitted


LABELS = tuple(LABEL_CAPABILITIES)


def _prepare(documents, results):
    indexed = _documents(documents)
    arms = {}
    for raw in results:
        if (not isinstance(raw, dict) or raw.get("document_id") not in indexed
                or not isinstance(raw.get("arm_id"), str) or not raw["arm_id"].strip()):
            raise ValueError("result requires a known document_id and arm_id")
        rows = arms.setdefault(raw["arm_id"], {})
        ident = raw["document_id"]
        if ident in rows:
            raise ValueError("duplicate result for arm/document")
        capabilities = raw.get("capabilities")
        if (not isinstance(capabilities, dict) or set(capabilities) != set(CAPABILITY_FIELDS)
                or any(type(value) is not bool for value in capabilities.values())):
            raise ValueError("arm requires explicit boolean capabilities")
        if rows and next(iter(rows.values()))["capabilities"] != capabilities:
            raise ValueError("arm capabilities changed between documents")
        try:
            row = validate_result(indexed[ident], raw)
            row.update(invalid=False, missing=False, discarded_predictions=0)
        except ValueError as exc:
            if raw["arm_id"] == "scibert-base":
                raise
            row = {"status": "failure", "spans": [], "reason": str(exc), "invalid": True,
                   "missing": False, "capabilities": deepcopy(capabilities),
                   "discarded_predictions": len(raw["spans"]) if isinstance(raw.get("spans"), list) else 0}
        rows[ident] = row
    prepared = {}
    for arm, rows in arms.items():
        started = any(row["status"] not in ("unavailable", "unsupported") for row in rows.values())
        capabilities = deepcopy(next(iter(rows.values()))["capabilities"])
        for ident in indexed:
            if ident not in rows:
                rows[ident] = {"status": "failure" if started else "unavailable", "spans": [],
                               "reason": "missing_result", "invalid": False, "missing": True,
                               "capabilities": capabilities, "discarded_predictions": 0}
        counts = {"population_documents": len(indexed),
                  "completed_documents": sum(row["status"] in COMPLETED_STATUSES for row in rows.values()),
                  "failed_documents": sum(row["status"] == "failure" for row in rows.values()),
                  "missing_documents": sum(row["missing"] for row in rows.values()),
                  "unavailable_documents": sum(row["status"] == "unavailable" for row in rows.values()),
                  "unsupported_documents": sum(row["status"] == "unsupported" for row in rows.values()),
                  "invalid_documents": sum(row["invalid"] for row in rows.values())}
        prepared[arm] = {"inference_started": started, "capabilities": capabilities,
                         "operations": counts, "rows": rows}
    return indexed, prepared


def _span_set(row, label):
    return {(span["start"], span["end"]) for span in row["spans"] if span["label"] == label}


def _details(indexed, arm):
    return [{"document_id": ident, "text_revision": doc["text_revision"],
             "status": arm["rows"][ident]["status"], "reason": arm["rows"][ident].get("reason"),
             "discarded_predictions": arm["rows"][ident]["discarded_predictions"]}
            for ident, doc in indexed.items()]


def agreement(documents: list[dict], results: list[dict]) -> dict:
    """Measure overlap only on jointly valid outputs; never emit accuracy metrics."""
    indexed, arms = _prepare(documents, results)
    report = {"mode": "unlabelled_agreement", "population_documents": len(indexed), "arms": {}, "pairs": []}
    for name, arm in arms.items():
        report["arms"][name] = {key: deepcopy(arm[key]) for key in ("inference_started", "capabilities", "operations")}
        report["arms"][name].update(candidate_counts={label: sum(len(_span_set(row, label))
            for row in arm["rows"].values()) for label in LABELS}, per_document=_details(indexed, arm))
    for left, right in combinations(arms, 2):
        joint = [ident for ident in indexed if all(arms[name]["rows"][ident]["status"] in COMPLETED_STATUSES
                                                   for name in (left, right))]
        pair = {"left_arm_id": left, "right_arm_id": right, "population_documents": len(indexed),
                "jointly_valid_documents": len(joint), "jointly_valid_document_ids": joint,
                "label_jointly_valid_documents": {}, "labels": {}}
        for label, capability in LABEL_CAPABILITIES.items():
            eligible = [ident for ident in joint if all(arms[name]["rows"][ident]["capabilities"][capability]
                                                        for name in (left, right))]
            sets = [{(ident, *span) for ident in eligible
                     for span in _span_set(arms[name]["rows"][ident], label)} for name in (left, right)]
            a, b = sets
            pair["label_jointly_valid_documents"][label] = len(eligible)
            pair["labels"][label] = {"intersection": len(a & b), "union": len(a | b),
                                     "left_only": len(a - b), "right_only": len(b - a),
                                     "jaccard": len(a & b) / len(a | b) if a | b else None}
        report["pairs"].append(pair)
    return report


def _region_state(span, regions):
    start, end = span
    if any(lo <= start < end <= hi for lo, hi in regions):
        return "eligible"
    return "boundary_crossing" if any(lo < end and start < hi for lo, hi in regions) else "uncovered"


def _covered_size(regions):
    total = end = 0
    for lo, hi in sorted(regions):
        total += max(0, hi - max(lo, end))
        end = max(end, hi)
    return total


def _reference_index(indexed, references):
    by_document = {ident: {"spans": {}, "regions": {kind: {label: set() for label in LABELS}
                                                    for kind in REVIEW_KINDS}} for ident in indexed}
    provenance = []
    for ref in references:
        _permitted(ref)
        if ref.get("document_id") not in indexed:
            raise ValueError("reference document_id not in frozen input")
        doc = indexed[ref["document_id"]]
        validated = validate_reference(doc, ref)
        target = by_document[doc["document_id"]]
        for span in validated["spans"]:
            kind = span.get("review_kind", validated["provenance"].get("review_kind"))
            if kind is not None and kind not in REVIEW_KINDS:
                raise ValueError("invalid reference span review_kind")
            key = (span["label"], span["start"], span["end"])
            target["spans"].setdefault(key, set()).add(kind)
        for region in validated["coverage"]:
            target["regions"][region["review_kind"]][region["label"]].add((region["start"], region["end"]))
        provenance.append({key: deepcopy(value) for key, value in validated.items() if key not in ("spans", "coverage")})
    return by_document, provenance


def _null_scores():
    return dict.fromkeys(("tp", "fp", "fn", "precision", "recall", "f1"))


def _add(counts, gold, predictions):
    counts["tp"] += len(gold & predictions)
    counts["fp"] += len(predictions - gold)
    counts["fn"] += len(gold - predictions)


def _label_score(indexed, arm, refs, label, kind):
    supported = arm["capabilities"][LABEL_CAPABILITIES[label]]
    active = supported and arm["inference_started"]
    primary, secondary = (dict.fromkeys(("tp", "fp", "fn"), 0) for _ in range(2))
    coverage = {"eligible_documents": 0, "eligible_gold_spans": 0, "covered_codepoints": 0}
    exclusions = dict.fromkeys(("gold_boundary_crossing", "gold_uncovered",
                                "prediction_boundary_crossing", "prediction_uncovered"), 0)
    recovery = {"eligible_positives": 0, "recovered": 0, "missed": 0,
                "recovery_rate": None, "eligible_documents": 0}
    denominators = {"operational_documents": 0, "completed_only_documents": 0}
    details = {}
    for ident in indexed:
        ref, row = refs[ident], arm["rows"][ident]
        regions = ref["regions"][kind][label]
        gold, candidates = set(), set()
        for (span_label, start, end), kinds in ref["spans"].items():
            if span_label != label:
                continue
            state = _region_state((start, end), regions)
            if state == "eligible":
                gold.add((start, end))
            else:
                exclusions["gold_" + state] += 1
                if kind in kinds:
                    candidates.add((start, end))
        predictions = _span_set(row, label)
        eligible_predictions = set()
        for span in predictions:
            state = _region_state(span, regions)
            if state == "eligible":
                eligible_predictions.add(span)
            else:
                exclusions["prediction_" + state] += 1
        coverage["eligible_documents"] += bool(regions)
        coverage["eligible_gold_spans"] += len(gold)
        coverage["covered_codepoints"] += _covered_size(regions)
        recovery["eligible_positives"] += len(candidates)
        recovery["eligible_documents"] += bool(candidates)
        recovered = len(candidates & predictions) if active else None
        if active:
            _add(primary, gold, eligible_predictions)
            denominators["operational_documents"] += bool(regions)
            if row["status"] in COMPLETED_STATUSES:
                _add(secondary, gold, eligible_predictions)
                denominators["completed_only_documents"] += bool(regions)
            recovery["recovered"] += recovered
            recovery["missed"] += len(candidates) - recovered
        details[ident] = {"eligible_gold_spans": len(gold), "eligible_predictions": len(eligible_predictions),
                          "operational": prf(len(gold & eligible_predictions), len(eligible_predictions - gold),
                                             len(gold - eligible_predictions)) if active else _null_scores()}
    if active:
        recovery["recovery_rate"] = (recovery["recovered"] / recovery["eligible_positives"]
                                     if recovery["eligible_positives"] else None)
    else:
        recovery.update(recovered=None, missed=None)
    return {"supported": supported, "operational": prf(**primary) if active else _null_scores(),
            "secondary_completed_only": prf(**secondary) if active else _null_scores(),
            "positive_recovery": recovery, "coverage": coverage, "exclusions": exclusions,
            "denominators": denominators}, details


def score_reference(documents: list[dict], results: list[dict], references: list[dict], *, review_kind: str) -> dict:
    """Score each field only inside complete regions of the requested review kind."""
    if review_kind not in REVIEW_KINDS:
        raise ValueError("invalid review_kind")
    indexed, arms = _prepare(documents, results)
    for document in indexed.values():
        _permitted(document)
    refs, provenance = _reference_index(indexed, references)
    report = {"mode": "reference_diagnostic", "review_kind": review_kind,
              "population_documents": len(indexed), "reference_provenance": provenance,
              "population": [{key: deepcopy(value) for key, value in doc.items() if key != "text"}
                             for doc in indexed.values()], "coverage": {}, "arms": {}}
    for kind in REVIEW_KINDS:
        report["coverage"][kind] = {label: {
            "documents": sum(bool(ref["regions"][kind][label]) for ref in refs.values()),
            "regions": sum(len(ref["regions"][kind][label]) for ref in refs.values()),
            "covered_codepoints": sum(_covered_size(ref["regions"][kind][label]) for ref in refs.values())}
            for label in LABELS}
    for name, arm in arms.items():
        result = {key: deepcopy(arm[key]) for key in ("inference_started", "capabilities", "operations")}
        result.update(labels={}, per_document=_details(indexed, arm))
        for label in LABELS:
            result["labels"][label], details = _label_score(indexed, arm, refs, label, review_kind)
            for doc in result["per_document"]:
                doc.setdefault("labels", {})[label] = details[doc["document_id"]]
        report["arms"][name] = result
    return report
