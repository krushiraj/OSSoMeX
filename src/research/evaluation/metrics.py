"""End-to-end counts over a fixed, explicitly covered document population."""

from collections import Counter
from copy import deepcopy
import hashlib
import json

from ..contracts import ContractError, FIELDS, INTENT_BITS, check_span, validate_document
from ..schema import INTENTS, SENTIMENTS

VERSION = "scibert-v2.0"


def prf(tp: int, fp: int, fn: int) -> dict:
    if any(type(v) is not int or v < 0 for v in (tp, fp, fn)):
        raise ValueError("counts must be nonnegative integers")
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def _counts() -> dict:
    return {"tp": 0, "fp": 0, "fn": 0}


def _values(record) -> set:
    return {edge["text"] for edge in record.get("version_links", [])} if record else set()


def _offset_edges(record) -> set:
    return {(edge["span"]["start"], edge["span"]["end"]) for edge in record.get("version_links", [])
            if edge.get("span")} if record else set()


def _add_edges(counts, gold, pred):
    counts["tp"] += len(gold & pred)
    counts["fp"] += len(pred - gold)
    counts["fn"] += len(gold - pred)


def count_value_edges(gold: list[dict], predictions: list[dict]) -> dict:
    def edges(records):
        return {(r["document_id"], r.get("text_revision"), r["name_span"]["start"], r["name_span"]["end"], value)
                for r in records for value in _values(r)}
    counts = _counts()
    _add_edges(counts, edges(gold), edges(predictions))
    return counts


def _digest(value):
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _metric(counts, supported=True):
    result = {**prf(**counts), "supported": supported}
    if not supported:
        result.update(precision=None, recall=None, f1=None)
    return result


def _class_metrics(counts, labels, supported):
    per_label = {label: _metric(counts[label], supported) for label in labels}
    missing = [label for label in labels if counts[label]["tp"] + counts[label]["fn"] == 0]
    defined = [v["f1"] for v in per_label.values() if v["f1"] is not None]
    observed = sum(defined) / len(defined) if defined else None
    return per_label, (None if missing else observed), observed, missing


def _span_key(record, text):
    span = record.get("name_span")
    if (isinstance(span, dict) and type(span.get("start")) is int and type(span.get("end")) is int
            and 0 <= span["start"] < span["end"] <= len(text)
            and text[span["start"]:span["end"]] == record.get("name")):
        return span["start"], span["end"]
    return None


def _covered(regions, span, field, length):
    start, end = span if span is not None else (0, length)
    cursor = start
    for region in sorted(regions, key=lambda row: (row["start"], row["end"])):
        if not region["fields"].get(field, False):
            continue
        if region["start"] > cursor:
            break
        if region["end"] > cursor:
            cursor = region["end"]
        if cursor >= end:
            return True
    return False


def _intent_known(gold, label):
    if label == "mentioned":
        return all(gold["known"][bit] for bit in INTENT_BITS) or any(
            gold["known"][bit] and bit in (gold.get("intents") or []) for bit in INTENT_BITS)
    return gold["known"][label]


def _intent_coverage(regions, span, label, length):
    bits = INTENT_BITS if label == "mentioned" else (label,)
    return all(_covered(regions, span, bit, length) for bit in bits)


def _validate_rows(records, docs, reference):
    grouped = {doc_id: {} for doc_id in docs}
    for ordinal, record in enumerate(records):
        doc_id = record.get("document_id")
        if doc_id not in docs:
            raise ContractError(record, "document_id", "outside fixed population")
        doc = docs[doc_id]
        if record.get("text_revision") != doc["text_revision"]:
            raise ContractError(record, "text_revision", "revision differs from frozen document")
        key = _span_key(record, doc["text"])
        if key is None and reference:
            raise ContractError(record, "name_span", "unaligned reference requires adjudication")
        if reference:
            masks = record.get("known")
            if not isinstance(masks, dict) or set(masks) != set(FIELDS) or any(type(v) is not bool for v in masks.values()):
                raise ContractError(record, "known", "reference requires explicit boolean field masks")
            if not masks["software"]:
                raise ContractError(record, "known.software", "reference occurrence must be known software")
            if any(masks[bit] for bit in INTENT_BITS) and record.get("intents") is None:
                raise ContractError(record, "intents", "known reference labels cannot be null")
            if masks["sentiment"] and record.get("sentiment") not in SENTIMENTS:
                raise ContractError(record, "sentiment", "known reference label cannot be null")
        if record.get("intents") is not None:
            labels = record["intents"]
            if not isinstance(labels, list) or any(label not in INTENTS for label in labels) or ("mentioned" in labels and len(labels) != 1):
                raise ContractError(record, "intents", "invalid intent set")
        if record.get("sentiment") is not None and record["sentiment"] not in SENTIMENTS:
            raise ContractError(record, "sentiment", "invalid sentiment label")
        for edge in record.get("version_links", []):
            if not isinstance(edge.get("text"), str) or not edge["text"].strip():
                raise ContractError(record, "version_links", "invalid version value")
            if edge.get("span") is not None:
                start, end = check_span(edge["span"], record, "version_links.span", len(doc["text"]))
                if doc["text"][start:end] != edge["text"]:
                    raise ContractError(record, "version_links.span", "version source text mismatch")
        key = key if key is not None else ("unaligned", ordinal)
        previous = grouped[doc_id].get(key)
        if previous is not None:
            compared = ("name", "version_links", "intents", "sentiment") + (("known",) if reference else ())
            if any(previous.get(field) != record.get(field) for field in compared):
                raise ContractError(record, "name_span", "conflicting duplicate occurrence")
        else:
            grouped[doc_id][key] = deepcopy(record)
    return grouped


def evaluate_v2(documents: list[dict], gold: list[dict], predictions: list[dict],
                coverage: list[dict], statuses: list[dict], capabilities: dict) -> dict:
    docs = {}
    for raw in documents:
        doc = validate_document(raw)
        if doc["document_id"] in docs:
            raise ContractError(doc, "document_id", "duplicate fixed-population document")
        docs[doc["document_id"]] = doc
    if not docs:
        raise ContractError({}, "documents", "empty evaluation population")
    gold_docs, pred_docs = _validate_rows(gold, docs, True), _validate_rows(predictions, docs, False)
    regions = {doc_id: [] for doc_id in docs}
    for region in coverage:
        doc = docs.get(region.get("document_id"))
        if doc is None or region.get("text_revision") != doc["text_revision"]:
            raise ContractError(region, "coverage", "unknown document or mismatched revision")
        check_span(region, region, "coverage", len(doc["text"]))
        fields = region.get("fields")
        if not isinstance(fields, dict) or set(fields) - set(FIELDS) or any(type(v) is not bool for v in fields.values()):
            raise ContractError(region, "coverage.fields", "invalid field masks")
        for existing in regions[doc["document_id"]]:
            if region["start"] < existing["end"] and existing["start"] < region["end"]:
                raise ContractError(region, "coverage", "overlapping coverage must be materialized disjointly")
        regions[doc["document_id"]].append(deepcopy(region))
    states = {}
    for state in statuses:
        if state.get("document_id") not in docs or state["document_id"] in states:
            raise ContractError(state, "status", "unknown or duplicate document status")
        for key in ("chunks_expected", "chunks_succeeded", "chunks_failed"):
            if key in state and (type(state[key]) is not int or state[key] < 0):
                raise ContractError(state, key, "chunk counts must be nonnegative integers")
        states[state["document_id"]] = state
    supported = {"software": capabilities.get("software", capabilities.get("name", False)) is True,
                 "versions": capabilities.get("versions", capabilities.get("version", False)) is True,
                 "intents": capabilities.get("intents") is True, "sentiment": capabilities.get("sentiment") is True}
    strict = supported["versions"] and capabilities.get("version_offsets", True) is True and all(
        edge.get("span") is not None for row in gold + predictions for edge in row.get("version_links", []))
    per_document, exclusions, failure_counts, confusion = [], Counter(), Counter(), Counter()
    for doc_id, doc in docs.items():
        gs, ps, rs = gold_docs[doc_id], pred_docs[doc_id], regions[doc_id]
        counts = {key: _counts() for key in ("mentions", "versions", "strict_versions", "complete")}
        counts.update(intents={label: _counts() for label in INTENTS}, sentiment={label: _counts() for label in SENTIMENTS})
        auxiliary = {"intent_exact_correct": 0, "intent_exact_eligible": 0, "false_opinions": 0, "not_expressed_gold": 0,
                     "matched_mentions": 0, "matched_intent_eligible": 0, "matched_intent_correct": 0,
                     "matched_sentiment_eligible": 0, "matched_sentiment_correct": 0,
                     "zero_mention_eligible": 0, "zero_mention_false_positive": 0}
        for key in gs.keys() | ps.keys():
            g, p = gs.get(key), ps.get(key)
            span = key if key[0] != "unaligned" else None
            if g is None and not _covered(rs, span, "software", len(doc["text"])):
                exclusions["prediction_outside_software_coverage"] += 1
                continue
            counts["mentions"]["tp" if g and p else "fn" if g else "fp"] += 1
            if g and p:
                auxiliary["matched_mentions"] += 1
            if p:
                for field in p.get("invalid_fields", []):
                    exclusions[f"invalid_prediction_{field}"] += 1
            versions_known = g["known"]["versions"] if g else _covered(rs, span, "versions", len(doc["text"]))
            if versions_known:
                _add_edges(counts["versions"], _values(g), _values(p))
                if strict:
                    _add_edges(counts["strict_versions"], _offset_edges(g), _offset_edges(p))
            else:
                exclusions["versions_unknown"] += 1
            g_intents = set(g.get("intents") or []) if g else set()
            p_intents = set(p.get("intents") or []) if p else set()
            for label in INTENTS:
                known = _intent_known(g, label) if g else _intent_coverage(rs, span, label, len(doc["text"]))
                if known:
                    _add_edges(counts["intents"][label], {label} & g_intents, {label} & p_intents)
                else:
                    exclusions[f"intent_{label}_unknown"] += 1
            if g and all(g["known"][bit] for bit in INTENT_BITS):
                auxiliary["intent_exact_eligible"] += 1
                auxiliary["intent_exact_correct"] += int(p is not None and p_intents == g_intents)
                if p:
                    auxiliary["matched_intent_eligible"] += 1
                    auxiliary["matched_intent_correct"] += int(p_intents == g_intents)
            sentiment_known = g["known"]["sentiment"] if g else _covered(rs, span, "sentiment", len(doc["text"]))
            g_sent, p_sent = g.get("sentiment") if g else None, p.get("sentiment") if p else None
            if sentiment_known:
                if g and p:
                    auxiliary["matched_sentiment_eligible"] += 1
                    auxiliary["matched_sentiment_correct"] += int(g_sent == p_sent)
                for label in SENTIMENTS:
                    _add_edges(counts["sentiment"][label], {label} if g_sent == label else set(), {label} if p_sent == label else set())
                confusion[(g_sent or "@spurious", p_sent or "@missed")] += 1
                if g_sent == "not_expressed":
                    auxiliary["not_expressed_gold"] += 1
                    auxiliary["false_opinions"] += int(p_sent in ("positive", "negative", "mixed"))
            else:
                exclusions["sentiment_unknown"] += 1
            full_known = all(g["known"].values()) if g else all(_covered(rs, span, field, len(doc["text"])) for field in FIELDS)
            if full_known:
                correct = bool(g and p and _values(g) == _values(p) and g_intents == p_intents and g_sent == p_sent)
                counts["complete"]["tp"] += int(correct)
                counts["complete"]["fn"] += int(g is not None and not correct)
                counts["complete"]["fp"] += int(p is not None and not correct)
        if not gs and _covered(rs, None, "software", len(doc["text"])):
            auxiliary["zero_mention_eligible"] = 1
            auxiliary["zero_mention_false_positive"] = int(counts["mentions"]["fp"] > 0)
        state = states.get(doc_id, {"status": "missing"})
        status = state.get("status", state.get("final_status", "missing"))
        failed = status not in ("success", "no_mentions", "completed") or state.get("chunks_failed", 0) > 0
        failure_counts["failed_documents"] += int(failed)
        failure_counts["failed_chunks"] += state.get("chunks_failed", 0)
        failure_counts["missing_status_documents"] += int(status == "missing")
        per_document.append({"document_id": doc_id, "work_group_id": doc.get("work_group_id", doc_id),
                             "source": (doc.get("metadata") or {}).get("source"), "text_revision": doc["text_revision"],
                             "counts": counts, "auxiliary": auxiliary, "status": status})

    def pooled(key):
        return {field: sum(row["counts"][key][field] for row in per_document) for field in ("tp", "fp", "fn")}

    def pooled_labels(key, labels):
        return {label: {field: sum(row["counts"][key][label][field] for row in per_document)
                        for field in ("tp", "fp", "fn")} for label in labels}

    aux = {key: sum(row["auxiliary"][key] for row in per_document) for key in per_document[0]["auxiliary"]}
    intents, macro, observed, missing = _class_metrics(pooled_labels("intents", INTENTS), INTENTS, supported["intents"])
    sentiment, sent_macro, sent_observed, sent_missing = _class_metrics(pooled_labels("sentiment", SENTIMENTS), SENTIMENTS, supported["sentiment"])
    return {"evaluator_version": VERSION, "policy_version": VERSION,
            "input_hashes": {key: _digest(value) for key, value in
                             (("documents", documents), ("reference", gold), ("predictions", predictions),
                              ("coverage", coverage), ("statuses", statuses), ("capabilities", capabilities))},
            "eligible_documents": list(docs), "per_document": per_document,
            "mention_detection": _metric(pooled("mentions"), supported["software"]),
            "version": {"supported": supported["versions"], "value_edges": _metric(pooled("versions"), supported["versions"]),
                        "strict_edges": _metric(pooled("strict_versions"), strict)},
            "intents": {"supported": supported["intents"], "per_label": intents, "macro": {"f1": macro},
                        "observed_macro_f1": observed, "missing_classes": missing,
                        "legacy_three_label_macro_f1": _class_metrics(pooled_labels("intents", INTENT_BITS), INTENT_BITS, supported["intents"])[1],
                        "exact_set_accuracy": aux["intent_exact_correct"] / aux["intent_exact_eligible"] if supported["intents"] and aux["intent_exact_eligible"] else None},
            "sentiment": {"supported": supported["sentiment"], "per_class": sentiment, "macro_f1": sent_macro,
                          "observed_macro_f1": sent_observed, "missing_classes": sent_missing,
                          "false_opinion_rate": aux["false_opinions"] / aux["not_expressed_gold"] if supported["sentiment"] and aux["not_expressed_gold"] else None,
                          "confusion": [{"gold": g, "prediction": p, "count": count} for (g, p), count in sorted(confusion.items())]},
            "complete_occurrence": _metric(pooled("complete"), all(supported.values())),
            "diagnostics": {"matched_only": {
                "intent_exact_accuracy": aux["matched_intent_correct"] / aux["matched_intent_eligible"] if supported["intents"] and aux["matched_intent_eligible"] else None,
                "sentiment_accuracy": aux["matched_sentiment_correct"] / aux["matched_sentiment_eligible"] if supported["sentiment"] and aux["matched_sentiment_eligible"] else None,
                "intent_eligible": aux["matched_intent_eligible"], "sentiment_eligible": aux["matched_sentiment_eligible"]}},
            "zero_mention_documents": {"eligible": aux["zero_mention_eligible"],
                                       "false_positive_rate": aux["zero_mention_false_positive"] / aux["zero_mention_eligible"] if aux["zero_mention_eligible"] else None},
            "failures": {**failure_counts, "total_documents": len(docs)}, "exclusions": dict(exclusions),
            "support": {"gold_mentions": sum(len(rows) for rows in gold_docs.values()),
                        "pred_mentions": sum(len(rows) for rows in pred_docs.values()),
                        "matched_mentions": aux["matched_mentions"]}}
