from research import aggregate as a


def _row(doc, mid, version, intents, sentiment):
    return {
        "document_id": doc,
        "mention_id": mid,
        "name_span": {"start": 0, "end": 1},
        "version": version,
        "intents": intents,
        "sentiment": sentiment,
    }


def test_multiversion_one_mention_two_export_rows():
    rows = [
        _row("d1", "d1:0:5", "R2018a", ["used"], "not_expressed"),
        _row("d1", "d1:0:5", "R2021b", ["used"], "not_expressed"),
    ]
    out = a.aggregate_documents(rows)["d1"]
    assert out["distinct_mention_count"] == 1
    assert out["export_row_count"] == 2
    assert out["version_buckets"]["R2018a"]["distinct_mentions"] == 1
    assert out["version_buckets"]["R2021b"]["distinct_mentions"] == 1


def test_repeated_occurrences_counted_separately():
    rows = [
        _row("d1", "d1:0:5", None, ["used"], "not_expressed"),
        _row("d1", "d1:30:35", None, ["used"], "not_expressed"),
    ]
    out = a.aggregate_documents(rows)["d1"]
    assert out["distinct_mention_count"] == 2
    assert out["version_buckets"]["@null"]["distinct_mentions"] == 2


def test_intent_counts_can_exceed_mentions():
    rows = [
        _row("d1", "d1:0:5", None, ["created", "shared"], "not_expressed"),
    ]
    out = a.aggregate_documents(rows)["d1"]
    assert out["distinct_mention_count"] == 1
    assert out["intent_counts"]["created"] == 1
    assert out["intent_counts"]["shared"] == 1