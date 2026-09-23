from research import aggregate as a
from research import evaluate as ev


def _gold(records):
    return records


def _mention(doc, start, end, name, versions, intents, sentiment):
    return {
        "document_id": doc,
        "mention_id": f"{doc}:{start}:{end}",
        "name": name,
        "name_span": {"start": start, "end": end},
        "versions": versions,
        "intents": intents,
        "sentiment": sentiment,
    }


def _pred(doc, start, end, name, version, intents, sentiment):
    return {
        "document_id": doc,
        "mention_id": f"{doc}:{start}:{end}",
        "name": name,
        "name_span": {"start": start, "end": end},
        "version": version,
        "intents": intents,
        "sentiment": sentiment,
    }


def test_perfect_matches():
    g = [_mention("d1", 10, 16, "NumPy", [None], ["used"], "not_expressed")]
    p = [_pred("d1", 10, 16, "NumPy", None, ["used"], "not_expressed")]
    r = ev.Evaluator().evaluate(g, p)
    assert r["mention_detection"]["precision"] == 1.0
    assert r["mention_detection"]["recall"] == 1.0
    assert r["intents"]["exact_set_accuracy"] == 1.0
    assert r["complete_occurrence"]["core_accuracy"] == 1.0


def test_all_missing_is_zero_recall():
    g = [_mention("d1", 10, 16, "NumPy", [None], ["used"], "not_expressed")]
    r = ev.Evaluator().evaluate(g, [])
    assert r["mention_detection"]["recall"] == 0.0
    assert r["mention_detection"]["precision"] is None


def test_hallucination_is_false_positive():
    g = [_mention("d1", 4, 8, "snake", [None], ["mentioned"], "not_expressed")]
    p = [_pred("d1", 0, 9, "GhostTool", None, ["mentioned"], "not_expressed")]
    r = ev.Evaluator().evaluate(g, p)
    # the hallucinated name does not overlap the gold span -> FP, not TP
    assert r["mention_detection"]["precision"] == 0.0
    assert r["mention_detection"]["recall"] == 0.0


def test_dedupe_overlapping_chunks():
    rows = [
        _pred("d1", 10, 16, "NumPy", None, ["used"], "not_expressed"),
        _pred("d1", 10, 16, "NumPy", None, ["used"], "not_expressed"),
        _pred("d1", 30, 36, "SciPy", None, ["used"], "not_expressed"),
    ]
    out = ev.dedupe_predictions(rows)
    assert len(out) == 2