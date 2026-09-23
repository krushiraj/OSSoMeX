"""Occurrence-level matching and feature metrics (protocol section 13).

Matching is by document ID plus exact source name span, one-to-one after
overlap deduplication. Version edges, intents and sentiment are scored only on
matched mentions. The same code scores the silver gold; unsupported fields are
excluded, never zero-filled guesses.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from . import schema as schema_mod

INTENT_LABELS = ("created", "used", "shared")
SENTIMENT_LABELS = ("positive", "negative", "mixed", "not_expressed")


def _span(rec: dict) -> tuple[int, int] | None:
    span = rec.get("name_span")
    if isinstance(span, dict) and "start" in span and "end" in span:
        return (int(span["start"]), int(span["end"]))
    return None


def dedupe_predictions(records: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    out = []
    for rec in records:
        span = _span(rec)
        key = (
            rec.get("document_id"),
            span,
            rec.get("version"),
            tuple(sorted(rec.get("intents") or [])),
            rec.get("sentiment"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def group_by_doc(records: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        groups[rec.get("document_id", "<none>")].append(rec)
    return groups


class Evaluator:
    def __init__(self, score_sentiment: bool = True, metric_version: str = "legacy") -> None:
        if metric_version not in ("legacy", "v2"):
            raise ValueError(f"unknown metric version: {metric_version}")
        self.score_sentiment = score_sentiment
        self.metric_version = metric_version

    def evaluate(self, gold: list[dict], predictions: list[dict], **context) -> dict:
        if self.metric_version == "v2":
            from .evaluation.metrics import evaluate_v2
            return evaluate_v2(gold=gold, predictions=predictions, **context)
        if context:
            raise ValueError("fixed-population context requires metric_version='v2'")
        pred = dedupe_predictions(predictions)
        gold_docs = group_by_doc(gold)
        pred_docs = group_by_doc(pred)

        mention_tp = mention_fp = mention_fn = 0
        version_correct_explicit = version_pred_explicit = version_gold_explicit = 0
        null_correct = null_pred = null_gold = 0
        name_correct = 0
        matched_total = 0
        intent_tp: Counter = Counter()
        intent_fp: Counter = Counter()
        intent_fn: Counter = Counter()
        sentiment_conf: Counter = Counter()
        exact_set_correct = 0
        complete_core_correct = 0
        complete_full_correct = 0
        relaxed_tp = 0
        details: list[dict] = []

        all_docs = sorted(set(gold_docs) | set(pred_docs))
        for doc_id in all_docs:
            g_rows = gold_docs.get(doc_id, [])
            p_rows = pred_docs.get(doc_id, [])
            g_by_key: dict[tuple, list[dict]] = defaultdict(list)
            for g_row in g_rows:
                span = _span(g_row)
                if span:
                    g_by_key[span].append(g_row)
            p_by_key: dict[tuple, list[dict]] = defaultdict(list)
            for p_row in p_rows:
                span = _span(p_row)
                if span:
                    p_by_key[span].append(p_row)

            used_gold_keys: set[tuple] = set()
            for span, g_mentions in g_by_key.items():
                p_rows_for_key = p_by_key.get(span, [])
                matched = bool(p_rows_for_key)
                if matched:
                    mention_tp += 1
                    used_gold_keys.add(span)
                else:
                    mention_fn += 1
                for g_row in g_mentions:
                    versions = g_row.get("versions") or [g_row.get("version")]
                    gold_explicit = {str(v) for v in versions if v is not None}
                    gold_has = bool(gold_explicit)
                    pred_versions = {p.get("version") for p in p_rows_for_key}
                    pred_explicit = {str(v) for v in pred_versions if v is not None}
                    pred_has = bool(pred_explicit)

                    if matched and _same_name(g_mentions, p_rows_for_key):
                        name_correct += 1
                        matched_total += 1

                    if gold_has:
                        version_gold_explicit += 1
                        if pred_has and pred_explicit == gold_explicit:
                            version_correct_explicit += 1
                        if pred_has:
                            version_pred_explicit += 1
                    else:
                        null_gold += 1
                        if not pred_has:
                            null_correct += 1
                        if not pred_has:
                            null_pred += 1

                    if not matched:
                        continue
                    g_intents = set(g_row.get("intents") or [])
                    p_intents = set()
                    for p_row in p_rows_for_key:
                        for label in p_row.get("intents") or []:
                            if label in INTENT_LABELS:
                                p_intents.add(label)
                    for label in INTENT_LABELS:
                        if label in g_intents and label in p_intents:
                            intent_tp[label] += 1
                        elif label in p_intents:
                            intent_fp[label] += 1
                        elif label in g_intents:
                            intent_fn[label] += 1
                    if p_intents == g_intents:
                        exact_set_correct += 1

                    p_sent = {p.get("sentiment") for p in p_rows_for_key}
                    p_sent.discard(None)
                    if g_row.get("sentiment") and p_sent:
                        g_sent = g_row["sentiment"]
                        p_sent_val = list(p_sent)[0] if len(p_sent) == 1 else "mixed_ambiguous"
                        sentiment_conf[(g_sent, p_sent_val)] += 1

                    if (p_intents == g_intents
                            and ((not gold_has and not pred_has) or (gold_has and pred_has and pred_explicit == gold_explicit))):
                        complete_core_correct += 1
                    if (p_intents == g_intents
                            and ((not gold_has and not pred_has) or (gold_has and pred_has and pred_explicit == gold_explicit))
                            and (not self.score_sentiment
                                 or (g_row.get("sentiment") == _majority_sent(p_rows_for_key)))):
                        complete_full_correct += 1

                    details.append(
                        {
                            "document_id": doc_id,
                            "mention_id": g_row.get("mention_id"),
                            "name": g_row.get("name"),
                            "gold_versions": sorted(gold_explicit),
                            "gold_has_version": gold_has,
                            "pred_versions": sorted(pred_explicit),
                            "pred_has_version": pred_has,
                            "gold_intents": sorted(g_intents),
                            "pred_intents": sorted(p_intents),
                            "gold_sentiment": g_row.get("sentiment"),
                            "pred_sentiments": sorted(s for s in p_sent if s is not None),
                            "matched": matched,
                        }
                    )

            for span, p_rows in p_by_key.items():
                if span not in used_gold_keys:
                    mention_fp += 1
                    details.append(
                        {
                            "document_id": doc_id,
                            "mention_id": p_rows[0].get("mention_id"),
                            "name": p_rows[0].get("name"),
                            "gold_versions": [],
                            "gold_has_version": None,
                            "pred_versions": sorted(v for v in {p.get("version") for p in p_rows} if v is not None),
                            "pred_has_version": any(p.get("version") for p in p_rows),
                            "gold_intents": [],
                            "pred_intents": [],
                            "gold_sentiment": None,
                            "pred_sentiments": [],
                            "matched": False,
                            "false_positive": True,
                        }
                    )

        metrics = _prf(mention_tp, mention_fp, mention_fn)
        intent_metrics = {}
        for label in INTENT_LABELS:
            intent_metrics[label] = _prf(intent_tp[label], intent_fp[label], intent_fn[label])
        intent_micro = _prf(
            sum(intent_tp.values()), sum(intent_fp.values()), sum(intent_fn.values())
        )
        intent_macro = {}
        for key in ("precision", "recall", "f1"):
            values = [intent_metrics[label][key] for label in INTENT_LABELS if intent_metrics[label][key] is not None]
            intent_macro[key] = sum(values) / len(values) if values else None

        sent_metrics = _sentiment_metrics(sentiment_conf)

        return {
            "support": {
                "gold_mentions": mention_tp + mention_fn,
                "pred_mentions": mention_tp + mention_fp,
                "matched_total": matched_total,
                "gold_explicit_versions": version_gold_explicit,
                "pred_explicit_versions": version_pred_explicit,
                "gold_nulls": null_gold,
                "pred_nulls": null_pred,
                "intent_gold_support": {k: intent_tp[k] + intent_fn[k] for k in INTENT_LABELS},
                "sentiment_gold_support": sum(c for (_, _), c in sentiment_conf.items()),
            },
            "mention_detection": {
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "tp": mention_tp,
                "fp": mention_fp,
                "fn": mention_fn,
            },
            "name_correct_on_matched": name_correct / matched_total if matched_total else None,
            "version": {
                "explicit_edge_precision": version_correct_explicit / version_pred_explicit if version_pred_explicit else None,
                "explicit_edge_recall": version_correct_explicit / version_gold_explicit if version_gold_explicit else None,
                "explicit_edge_f1": _f1(version_correct_explicit / version_pred_explicit if version_pred_explicit else None,
                                        version_correct_explicit / version_gold_explicit if version_gold_explicit else None),
                "null_precision": null_correct / null_pred if null_pred else None,
                "null_recall": null_correct / null_gold if null_gold else None,
            },
            "intents": {
                "per_label": intent_metrics,
                "micro": intent_micro,
                "macro": intent_macro,
                "exact_set_accuracy": exact_set_correct / matched_total if matched_total else None,
            },
            "sentiment": sent_metrics,
            "complete_occurrence": {
                "core_accuracy": complete_core_correct / matched_total if matched_total else None,
                "full_accuracy": complete_full_correct / matched_total if matched_total else None,
            },
            "details": details,
        }


def _same_name(g_rows: list[dict], p_rows: list[dict]) -> bool:
    return any(g.get("name") == p.get("name") for g in g_rows for p in p_rows)


def _majority_sent(p_rows: list[dict]) -> str | None:
    values = Counter(p.get("sentiment") for p in p_rows if p.get("sentiment"))
    if not values:
        return None
    return values.most_common(1)[0][0]


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    return {"precision": precision, "recall": recall, "f1": _f1(precision, recall), "tp": tp, "fp": fp, "fn": fn}


def _f1(precision, recall):
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def _sentiment_metrics(conf: Counter) -> dict:
    tp = Counter()
    fp = Counter()
    fn = Counter()
    for (gold, pred), count in conf.items():
        if pred in SENTIMENT_LABELS:
            if gold == pred:
                tp[gold] += count
            else:
                fp[pred] += count
                fn[gold] += count
    per_class = {label: _prf(tp[label], fp[label], fn[label]) for label in SENTIMENT_LABELS}
    gold_total = sum(fn.values()) + sum(tp.values())
    false_opinion = sum(
        count for (gold, pred), count in conf.items()
        if gold == "not_expressed" and pred in ("positive", "negative", "mixed")
    )
    false_opinion_gold_ne = sum(
        count for (gold, pred), count in conf.items() if gold == "not_expressed"
    )
    macro_f1 = sum(
        per_class[l]["f1"] for l in SENTIMENT_LABELS if per_class[l]["f1"] is not None
    ) / sum(1 for l in SENTIMENT_LABELS if per_class[l]["f1"] is not None) if any(
        per_class[l]["f1"] is not None for l in SENTIMENT_LABELS
    ) else None
    return {
        "per_class": per_class,
        "macro_f1": macro_f1,
        "false_opinion_rate": false_opinion / false_opinion_gold_ne if false_opinion_gold_ne else None,
        "matched_with_sentiment": gold_total,
    }


def validate_gold_rows(rows: list[dict]) -> list[str]:
    problems: list[str] = []
    for i, row in enumerate(rows):
        if not row.get("document_id"):
            problems.append(f"row {i}: missing document_id")
        span = row.get("name_span")
        if not isinstance(span, dict) or "start" not in span or "end" not in span:
            problems.append(f"row {i}: missing/odd name_span")
        intents = row.get("intents")
        if not intents or not set(intents).issubset(schema_mod.INTENTS):
            problems.append(f"row {i}: intents outside {schema_mod.INTENTS}")
        if row.get("sentiment") not in schema_mod.SENTIMENTS:
            problems.append(f"row {i}: sentiment invalid")
    return problems
